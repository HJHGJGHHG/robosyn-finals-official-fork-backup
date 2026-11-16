"""Observation-owned planning: static URDF, predicted geometry, no simulator state."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import torch
from embodichain.lab.sim.solvers import OPWSolverCfg
from embodichain.lab.sim.planners.toppra_planner import (
    _toppra_solve_one_env, TrajectorySampleMethod,
)
from embodichain.lab.gym.envs.action_bank.configurable_action import ActionBank, get_func_tag

ROOT = Path(__file__).resolve().parent


class RobotKinematics:
    """The action-bank math API backed only by a static arm and observed joints."""
    def __init__(self):
        self.device = torch.device('cpu')
        urdf = ROOT / 'assets/CobotMagicWithGripperV100.urdf'
        joints = {j.attrib['name']: j.find('limit').attrib
                  for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
        self.limits = np.array([[float(joints[f'joint{i}'][k]) for k in ('lower', 'upper')]
                                for i in range(1, 7)], dtype=np.float32)
        self.solver = OPWSolverCfg(
            urdf_path=str(urdf), joint_names=[f'joint{i}' for i in range(1, 7)],
            root_link_name='arm_base', end_link_name='link6',
            tcp=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,.143],[0,0,0,1]]),
            user_qpos_limits=self.limits.tolist(),
        ).init_solver(device=self.device)
        self.bases = {}
        for arm, y in [('left_arm', .3), ('right_arm', -.3)]:
            pose = torch.eye(4); pose[:3, 3] = torch.tensor([.233, y, .835])
            self.bases[arm] = pose
        limits = np.concatenate([self.limits, [[0, .05]], self.limits, [[0, .05]]])
        self.body_data = SimpleNamespace(qpos_limits=torch.tensor(limits, dtype=torch.float32)[None])
        self._qpos = None

    def observe(self, state):
        q = torch.as_tensor(state, dtype=torch.float32, device='cpu').reshape(1, -1)
        if q.shape != (1, 14) or not torch.isfinite(q).all():
            raise ValueError('Planning requires finite observed qpos [1,14]')
        self._qpos = q.clone()

    def get_qpos(self):
        if self._qpos is None: raise RuntimeError('Observe qpos before planning')
        return self._qpos.clone()

    def get_joint_ids(self, name, **kwargs):
        return {'left_arm': list(range(6)), 'left_eef': [6],
                'right_arm': list(range(7,13)), 'right_eef': [13]}[name]

    def compute_fk(self, qpos, name, to_matrix=True):
        if not to_matrix: raise ValueError('Planner requires matrix FK')
        q = torch.as_tensor(qpos, dtype=torch.float32, device='cpu').reshape(-1, 6)
        return self.bases[name] @ self.solver.get_fk(q)

    def compute_ik(self, pose, joint_seed=None, name=None):
        target = torch.as_tensor(pose, dtype=torch.float32, device='cpu').reshape(-1, 4, 4)
        if not torch.isfinite(target).all(): raise ValueError('Nonfinite IK target')
        local = torch.linalg.inv(self.bases[name]) @ target
        seed = None if joint_seed is None else torch.as_tensor(joint_seed, dtype=torch.float32, device='cpu').reshape(-1,6)
        success, qpos = self.solver.get_ik(local, qpos_seed=seed)
        return success, qpos.reshape(-1, 6)


def joint_trajectory(waypoints, steps, velocity=.2, acceleration=.5):
    q = np.asarray(waypoints, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all() or int(steps) != steps or steps < 2:
        raise ValueError('Invalid joint trajectory input')
    if all(np.linalg.norm(a-b) <= 1e-3 for a,b in zip(q,q[1:])):
        return np.repeat(q[:1], int(steps), axis=0).astype(np.float32)
    result = _toppra_solve_one_env(q, velocity, acceleration,
                                  TrajectorySampleMethod.QUANTITY, int(steps))
    frames = result['positions']
    if not result['success'] or frames.shape != (steps, q.shape[1]) or not np.isfinite(frames).all():
        raise RuntimeError('Independent TOPPRA planning failed')
    return frames.astype(np.float32)


class MotionGenerator:
    """Joint-only adapter for the existing three direct-IK templates."""
    def __init__(self, robot):
        self.robot = robot

    def generate(self, target_states, options):
        if any(s.qpos is None for s in target_states):
            raise ValueError('This trajectory adapter accepts joint waypoints only')
        q = np.stack([s.qpos.detach().cpu().numpy().reshape(-1) for s in target_states])
        cfg = options.plan_opts
        frames = joint_trajectory(q, cfg.sample_interval, **cfg.constraints)
        return SimpleNamespace(positions=torch.from_numpy(frames)[None])


def _graph_trajectory(context, agent_uid, keypose_names, duration, edge_name=''):
    return joint_trajectory([context.affordance_datas[k] for k in keypose_names], duration).T


def _graph_hold(context, agent_uid, keypose_names, duration):
    q = np.asarray(context.affordance_datas[keypose_names[0]], dtype=np.float32)
    return np.repeat(q[:, None], duration, axis=1)


class PlanningContext:
    """Fresh per-episode data; never receives an environment or expert cache."""
    def __init__(self, task, robot, state):
        self.robot = robot
        self.robot.observe(state)
        self.device = torch.device('cpu')
        self.task = task
        self.cfg = SimpleNamespace(control_parts=['left_arm', 'left_eef', 'right_arm', 'right_eef'])
        self.affordance_datas = {f'{arm}_base_pose': pose.numpy().copy()
                                 for arm, pose in robot.bases.items()}
        for arm in robot.bases:
            self.affordance_datas[f'{arm}_init_qpos'] = robot.get_qpos()[0, robot.get_joint_ids(arm)].numpy()
        self.agent_qpos_flip_ids = [3, 4]
        self.agent_qpos_flip_threshold = 3.455751918948773
        self.agent_qpos_flip_mode = 'delta'
        constants = json.loads((ROOT/'assets/planning_constants.json').read_text()).get(task, {})
        for key, value in constants.items():
            array = np.asarray(value)
            self.affordance_datas[key] = array[0] if array.shape == (1,4,4) else value
        config = ROOT/'action_config.json'
        self.action_config = json.loads(config.read_text()) if config.exists() else None

    def action_bank_compute_ik(self, target_xpos, qpos_seed, control_part):
        return self.robot.compute_ik(target_xpos, qpos_seed, control_part)

    def initialize_graph(self, bank_class, config):
        # Failed candidate graphs must not leave derived waypoints for the next candidate.
        if not hasattr(self, "_input_keys"):
            self._input_keys = set(self.affordance_datas)
        self.affordance_datas = {k: deepcopy(v) for k, v in self.affordance_datas.items() if k in self._input_keys}
        self.action_bank = bank_class(deepcopy(config))
        node_functions, edge_functions = {}, {}
        # Task gripper functions use 0=closed, 1=open; generic SDK functions reverse it.
        for name in (ActionBank.__name__, bank_class.__name__):
            node_functions.update(get_func_tag('node').functions.get(name, {}))
            edge_functions.update(get_func_tag('edge').functions.get(name, {}))
        edge_functions['plan_trajectory'] = _graph_trajectory
        edge_functions['stand_still'] = _graph_hold
        self.graph_compose, jobs, indices = self.action_bank.parse_network(node_functions, edge_functions, vis_graph=False)
        self.packages = self.action_bank.gantt(tasks_data=jobs, taskkey2index=indices, vis=False)

    def plan_graph(self):
        ret = self.action_bank.create_action_list(self, self.graph_compose, self.packages)
        if ret is None: return None
        keys = ('left_arm','left_eef','right_arm','right_eef')
        frames = np.concatenate([ret[k] for k in keys], axis=0).T.astype(np.float32)
        # Graph jaws are normalized; official env.step accepts physical joint positions.
        frames[:, [6, 13]] *= .05
        if frames.ndim != 2 or frames.shape[1] != 14 or not np.isfinite(frames).all():
            raise RuntimeError('Action graph returned invalid joint commands')
        return frames
