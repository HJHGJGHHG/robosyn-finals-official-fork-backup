"""Static CobotMagic kinematics and observation-to-action joint conversion."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from embodichain.lab.sim.solvers import OPWSolverCfg
from .motion import Kinematics


def physical_qpos(obs, *, contact=False):
    q = torch.as_tensor(obs['robot']['qpos']).detach().cpu().numpy()
    if q.shape != (1, 14) or not np.isfinite(q).all():
        raise ValueError('Expected finite (1,14) observation qpos')
    q = q[0].astype(float).copy()
    # Reset physics can overshoot the jaw limit; allow at most 5 micrometers.
    low,high=(-.1,1.1) if contact else (-1e-4,1.0001)
    if ((q[[6,13]] < low) | (q[[6,13]] > high)).any():
        raise ValueError('Observation grippers must be normalized to [0,1]')
    # Contact can deflect observed jaws beyond command limits; retain it for geometry.
    q[[6,13]] = (q[[6,13]] if contact else np.clip(q[[6,13]], 0, 1)) * .05
    return q


def build_kinematics():
    urdf = Path(__file__).parent/'assets/CobotMagicWithGripperV100.urdf'
    joints = {j.attrib['name']: j.find('limit').attrib for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    limits = np.array([[float(joints[f'joint{i}'][k]) for k in ('lower','upper')] for i in range(1,7)])
    velocity = np.minimum([float(joints[f'joint{i}']['velocity']) for i in range(1,7)], [2.,2.,2.,2.,2.,1.5])
    cfg = dict(urdf_path=str(urdf), joint_names=[f'joint{i}' for i in range(1,7)],
               root_link_name='arm_base', end_link_name='link6',
               tcp=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,.143],[0,0,0,1]]),
               user_qpos_limits=limits.tolist())
    solver = OPWSolverCfg(**cfg).init_solver(device=torch.device('cpu'))
    kin = {}
    for arm, y in [('left_arm', .3), ('right_arm', -.3)]:
        base = np.eye(4); base[:3,3] = [.233,y,.835]
        kin[arm] = Kinematics(solver, base, limits.copy(), velocity.copy())
    return kin
