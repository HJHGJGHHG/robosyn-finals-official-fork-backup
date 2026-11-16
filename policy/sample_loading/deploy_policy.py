"""Initial RGB prediction -> bounded template -> official 14D action interface.

Only RGB, robot proprioception/kinematics and fixed task geometry enter planning.
Object ground truth and success checks belong to the external evaluator.
This baseline predicts once per episode; it is not stage-wise visual feedback.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .model_v4 import PerceptionV4
from .geometry import CAMERAS, decode
from .planner import plan, PlanningError
from .kinematics import build_kinematics, physical_qpos


def numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def encode_obs(obs, device):
    frames = []
    for camera in CAMERAS:
        x = torch.as_tensor(obs['sensor'][camera]['color']).detach()
        if x.ndim == 3:
            x = x[None]
        if x.shape[0] != 1 or x.shape[1:3] != (480, 640) or x.shape[-1] not in (3, 4):
            raise ValueError(f'Invalid camera shape {camera}: {tuple(x.shape)}')
        is_byte = x.dtype == torch.uint8
        x = x[..., :3].to(device='cpu', dtype=torch.float32)
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 255:
            raise ValueError(f'Invalid RGB values: {camera}')
        # Match saved training RGB bytes also for a float simulator image.
        # Byte input always divides by 255, even for an all-dark frame.
        if not is_byte:
            x = (x * (255 if x.max() <= 1.5 else 1)).clamp(0, 255).to(torch.uint8).float()
        frames.append((x / 255).permute(0, 3, 1, 2).to(device))
    q = torch.as_tensor(obs['robot']['qpos'], device=device, dtype=torch.float32)
    if q.shape != (1, 14) or not torch.isfinite(q).all():
        raise ValueError('Expected finite normalized observation qpos (1,14)')
    return torch.stack(frames, 1), q


class Policy:
    def __init__(self, config):
        self.device = torch.device(config.get('pytorch_device', 'cpu'))
        checkpoint = Path(config['checkpoint_path']).resolve()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != config['checkpoint_sha256']:
            raise ValueError('Checkpoint does not match deployment configuration')
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.network = PerceptionV4(residual=bool(weights.get('residual_enabled',False)))
        self.network.load_state_dict(weights['model'], strict=True)
        self.network.to(self.device).eval()
        geometry_path = Path(config['geometry_path'])
        if not geometry_path.is_absolute():
            geometry_path = Path(__file__).resolve().parents[1] / geometry_path
        self.geometry = json.loads(geometry_path.read_text())['geometry']
        self.geometry['right_repair_enabled']=config.get('right_repair_enabled',True)
        self.kin = build_kinematics()
        self.control_variant = config.get('control_variant','C0')
        self.max_steps = int(config.get('max_steps', 1000))
        self.settle_steps = int(config.get('settle_steps', 25))
        if self.max_steps not in (500, 700, 1000) or self.settle_steps not in (24, 25):
            raise ValueError('Unsupported experiment horizon')
        self.transport_steps = int(config.get('transport_steps', 40))
        self.action_device = torch.device(config.get('action_device', 'cpu'))
        self.checkpoint_sha256 = digest
        self.template = config.get('template', 'nominal_v1')
        if self.template not in ('nominal_v1', 'fixed_grasp_v2'):
            raise ValueError('Unknown template')
        self.calibration = json.loads(Path(config['calibration_path']).read_text()) if self.template == 'fixed_grasp_v2' else None
        self.reset()

    def reset(self):
        self.result = None
        self.parameters = None
        self.cursor = 0
        self.plan_attempted = False
        self.done = False
        self.initial_observation = None
        self.initial_observation_receipt = None
        self._submission_steps = 0

    def prepare_initial_observation(self, env, obs):
        if self.initial_observation is None:
            from .initial_observation import synchronized_initial_observation
            self.initial_observation, self.initial_observation_receipt = synchronized_initial_observation(env, obs)
        return self.initial_observation

    def make_plan(self, env, obs, *, diagnostic_targets=None, diagnostic_mode='predicted'):
        """Plan from live observations; explicitly named oracle inputs are diagnostic-only 10D labels."""
        if self.plan_attempted:
            raise RuntimeError('Planning already attempted; reset is required')
        self.plan_attempted = True
        obs = self.prepare_initial_observation(env, obs)
        images, qpos = encode_obs(obs, self.device)
        with torch.inference_mode():
            values = self.network(images, qpos)['parameters'][0]
        from .diagnostics import mix_parameters
        self.raw_parameters = numpy(values).astype(float)
        self.parameters = mix_parameters(self.raw_parameters, diagnostic_targets, diagnostic_mode)
        self.geometry['prediction_bias_xy_m'] = [0., 0.] if diagnostic_mode in ('rack','both') else [-.00182, -.00201]
        # Outputs already are meters. No handover-style centimeter conversion.
        tube, rack = decode(self.parameters)
        kin = self.kin
        qinit = physical_qpos(obs)
        if getattr(self, 'template', 'nominal_v1') == 'fixed_grasp_v2':
            from .calibrated_planner import plan as calibrated_plan
            self.result = calibrated_plan(tube, rack, kin, qinit, self.geometry, self.calibration)
        else:
            try:
                self.result = plan(tube, rack, kin, qinit, self.geometry, max_steps=self.max_steps, settle_steps=self.settle_steps,transport_steps=self.transport_steps,control_variant=self.control_variant)
            except PlanningError as exc:
                self.result = {'actions':qinit[None], 'stages':[], 'planning_error':str(exc), 'rejections':getattr(exc, 'rejections', [])}
        actions = self.result['actions']
        if actions.ndim != 2 or actions.shape[1] != 14 or not 0 < len(actions) <= self.max_steps or not np.isfinite(actions).all():
            raise ValueError('Invalid action plan')


def get_model(config):
    if config.get('task_name') != 'sample_loading':
        raise ValueError('This policy supports sample_loading only')
    config = dict(config)
    root = Path(__file__).resolve().parent
    for key in ('checkpoint_path', 'geometry_path', 'calibration_path'):
        if key in config:
            path = Path(config[key]).expanduser()
            config[key] = str(path if path.is_absolute() else root / path)
    return Policy(config)


def reset_model(model, seed=None, **kwargs):
    model.reset()


def eval(env, model, obs):
    if model.done or model.cursor >= model.max_steps:
        raise RuntimeError('Episode ended; reset_model must precede another step')
    times = []
    if model.result is None:
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        model.make_plan(env, obs)
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)
        times.append(time.perf_counter() - started)
    action = model.result['actions'][min(model.cursor, len(model.result['actions']) - 1)]
    obs, _, terminated, truncated, info = env.step(torch.as_tensor(action[None], dtype=torch.float32, device=model.action_device))
    model.cursor += 1
    model._submission_steps = model.cursor
    model.done = bool(numpy(truncated).any())
    # The generic official evaluator calls its own success predicate after return.
    return obs, info, model.done, times
