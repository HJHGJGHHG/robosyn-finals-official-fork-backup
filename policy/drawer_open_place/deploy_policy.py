"""Official get_model/eval/reset_model interface; env is used only for step()."""
import time
import hashlib
from pathlib import Path
import torch
import numpy as np
from .geometry import LABELS
from .model import DrawerPerception
from .observation import encode_obs
from .kinematics import build_kinematics, physical_qpos
from .planner import plan
from .close_tail import append_close_tail
from .refinement import refine, retreat, Suffix
from .motion import PlanningError
from .clearance import check as check_clearance


class Policy:
    def __init__(self, config):
        self.config = dict(config)
        self.device = torch.device(config.get('pytorch_device','cpu'))
        checkpoint = torch.load(Path(config['checkpoint_path']),map_location='cpu',weights_only=False)
        if checkpoint.get('task') != 'drawer_open_place' or tuple(checkpoint.get('label_names',[])) != LABELS:
            raise ValueError('Checkpoint task/label contract mismatch')
        self.network = DrawerPerception()
        self.network.load_state_dict(checkpoint['model'],strict=True)
        self.network.to(self.device).eval()
        self.kin = build_kinematics()
        self.max_steps = int(config.get('max_steps',900))
        self.pull_distance = float(config.get('pull_distance_m',.19))
        self.timing = config.get('timing','baseline')
        self.close_enabled = bool(config.get('close_enabled',False))
        self.close_epsilon = float(config.get('close_epsilon_m',.01))
        self.action_device = config.get('action_device','cpu')
        self.reset()

    def reset(self):
        self.result, self.parameters, self.cursor, self.done = None,None,0,False
        self.replanned = False
        self.gates = {}
        self.execution_failure = None
        self.p6_history = []
        self.last_action, self.last_stage = None, None
        self._submission_steps = 0

    def predict_parameters(self, images, qpos):
        """Return the three predicted world XYZ points for submission logging."""
        return {'parameters': self.network(images,qpos)[0].float().cpu().numpy()}


def hold_one(model):
    r, i = model.result, model.cursor
    if len(r['actions']) >= model.max_steps:
        raise PlanningError('runtime_wait:step_budget')
    r['actions'] = np.insert(r['actions'], i, r['actions'][i-1], axis=0)
    for s in r['stages']:
        if s['start'] >= i:s['start'] += 1
    r['stages'].append(dict(name='tracking_gate_wait',start=i,steps=1))
    r['stages'].sort(key=lambda s:s['start'])
    if r.get('close_start_step',1001) > i:r['close_start_step'] += 1
    if r.get('p5_runtime_replan_at',-1) >= i:r['p5_runtime_replan_at'] += 1


def update_suffix(model, obs):
    mode = model.config.get('p5_mode','baseline')
    if mode == 'baseline' or model.execution_failure:return
    q = physical_qpos(obs,contact=True)
    if mode == 'p6':
        from .p6 import update
        update(model,q)
        return
    r, i = model.result, model.cursor
    if not model.replanned and r.get('p5_runtime_replan_at') == i:
        model.result = retreat(r,model.parameters,model.kin,model.config,i,q)
        model.replanned = True
        r = model.result
    stage = next((s['name'] for s in r['stages'] if s['start']==i),None)
    placement_gate = mode in ('placement','combined') and stage == 'left_release'
    park_gate = mode in ('retreat','combined') and stage == 'right_push_coarse'
    if not (placement_gate or park_gate):return
    state = model.gates.setdefault(stage,dict(waited=0,consecutive=0))
    error = float(np.rad2deg(abs(q[:6]-r['actions'][i-1,:6]).max()))
    if placement_gate:
        actual = model.kin['left_arm'].fk(q[None,:6])[0]
        zerror = abs(actual[2,3]-r['targets']['left_place'][2,3])
        ok = error<=2 and zerror<=.01
    else:
        try:
            check_clearance(q[:7],model.kin['left_arm'].base,model.parameters[6:9],model.config['asset_source'])
            ok = error<=1
        except PlanningError:ok=False
    state['consecutive'] = state['consecutive']+1 if ok else 0
    if state['consecutive'] >= 5:return
    if state['waited'] >= (5 if placement_gate else 20):
        raise PlanningError(stage+':actual_tracking_or_clearance_timeout',dict(joint_error_deg=error))
    hold_one(model);state['waited'] += 1


def get_model(config):
    """Load the hash-pinned candidate; resolve assets relative to this task package."""
    if config.get('task_name') != 'drawer_open_place':
        raise ValueError('This policy supports drawer_open_place only')
    config = dict(config)
    root = Path(__file__).resolve().parent
    for key in ('checkpoint_path', 'asset_source', 'drawer_asset'):
        path = Path(config[key]).expanduser()
        path = path if path.is_absolute() else root/path
        if not path.exists(): raise FileNotFoundError(path)
        config[key] = str(path)
    expected = config.get('checkpoint_sha256')
    if not expected: raise ValueError('checkpoint_sha256 is required')
    with Path(config['checkpoint_path']).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected: raise ValueError('Checkpoint hash mismatch')
    if not 1 <= int(config.get('max_steps', 1000)) <= 1000:
        raise ValueError('max_steps must be in [1,1000]')
    return Policy(config)


def reset_model(model, **kwargs):
    model.reset()


def eval(env, model, obs):
    """One physical action per call; planner exceptions remain explicit failures."""
    if model.done or model.cursor >= model.max_steps:
        raise RuntimeError('Reset required')
    if model.device.type == 'cuda': torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    if model.result is None:
        images,qpos = encode_obs(obs,model.device)
        with torch.inference_mode():
            model.parameters = model.predict_parameters(images,qpos)['parameters']
        model.result = plan(model.parameters,model.kin,physical_qpos(obs),model.max_steps,model.pull_distance,timing=model.timing)
        if model.close_enabled:
            try:
                model.result = refine(model.result,model.parameters,model.kin,model.config)
            except PlanningError:
                if model.config.get('p5_mode')!='p6':raise
                from .p6 import initial_grasp
                model.result=initial_grasp(model.result,model.parameters,model.kin,model.config,physical_qpos(obs),model.max_steps)
    try:
        update_suffix(model,obs)
    except PlanningError as exc:
        if model.config.get('p5_mode')=='p6':
            from .p6 import escape
            escape(model,physical_qpos(obs,contact=True),exc)
        else:
            model.execution_failure = dict(step=model.cursor+1,reason=str(exc),diagnostics=exc.diagnostics)
            try:
                suffix=Suffix(model.result,model.cursor,model.kin,model.max_steps)
                suffix.append('execution_failure_unload',20,joint=physical_qpos(obs,contact=True)[:6])
                model.result=suffix.finish()
                model.result['close_start_step']=1001
            except PlanningError as unload_error:
                model.execution_failure['unload_error']=str(unload_error)
                model.result['actions']=model.result['actions'][:model.cursor].copy()
    action = model.result['actions'][min(model.cursor,len(model.result['actions'])-1)]
    stage = next((s['name'] for s in model.result['stages'] if s['start']<=model.cursor<s['start']+s['steps']),'terminal_hold')
    if model.execution_failure and model.config.get('p5_mode')!='p6':
        if stage != 'execution_failure_unload':stage = 'execution_failure_hold'
    command=torch.as_tensor(action[None],dtype=torch.float32,device=model.action_device)
    if command.shape != (1, 14) or not torch.isfinite(command).all():
        raise ValueError('Joint command must be finite [1,14]')
    model.last_action,model.last_stage = command.detach().cpu().numpy()[0].copy(),stage
    if model.device.type == 'cuda': torch.cuda.synchronize(model.device)
    timings = [time.perf_counter()-start]
    obs,_,terminated,truncated,info = env.step(command)
    model.cursor += 1
    model._submission_steps = model.cursor
    model.done = bool(torch.as_tensor(truncated).any())
    return obs,info,model.done,timings
