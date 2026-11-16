"""Portable loading and one joint command per official evaluator call.

Images: three uint8 [1,H,W,3/4] RGB observations, or float RGB in [0,1].
Proprioception/action: [1,14], left arm6 + gripper + right arm6 + gripper.
Planning uses a private context built from obs and static policy assets.
"""
import hashlib
from pathlib import Path
import time
import numpy as np
import torch

CAMS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')

def load_model(args, model_class, task, policy_dir):
    """Load the task's hash-pinned checkpoint, relative to its policy directory."""
    if args.get('task_name') != task:
        raise ValueError(f'Expected task_name={task!r}, got {args.get("task_name")!r}')
    path = Path(args.get('checkpoint_path', 'checkpoint/best.pt')).expanduser()
    if not path.is_absolute():
        path = Path(policy_dir) / path
    expected = args.get('checkpoint_sha256')
    if not expected:
        raise ValueError('checkpoint_sha256 is required')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    budget = int(args.get('max_steps', 1000))
    if not 1 <= budget <= 1000:
        raise ValueError('max_steps must be in [1,1000]')
    if args.get('parapred_step', 1) != 1:
        raise ValueError('This adapter requires parapred_step=1')
    model = model_class(False, args.get('pytorch_device', 'cpu'), 1,
                        int(args.get('parapred_ik_attempts', 5)))
    model.load_checkpoint(path)
    model.eval()
    from .planning import RobotKinematics
    model._planning_robot = RobotKinematics()
    model._submission_task = task
    model._action_device = torch.device(args.get("action_device", "cpu"))
    model._submission_budget = budget
    model._submission_steps = 0
    return model

def encode_obs(obs, device):
    """Validate the public RGB/qpos fields and return [1,3,3,H,W], [1,14]."""
    frames = []
    for camera in CAMS:
        frame = torch.as_tensor(obs['sensor'][camera]['color'])
        is_uint8 = frame.dtype == torch.uint8
        if not is_uint8 and not frame.is_floating_point():
            raise ValueError(f'{camera}: expected uint8 RGB or float RGB in [0,1]')
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[-1] not in (3,4):
            raise ValueError(f'{camera}: expected [1,H,W,3/4] RGB')
        frame = frame[..., :3].permute(0,3,1,2).to(device=device, dtype=torch.float32)
        if not torch.isfinite(frame).all() or frame.min() < 0 or frame.max() > (255 if is_uint8 else 1):
            raise ValueError(f'{camera}: nonfinite or invalid RGB range')
        frames.append(frame / 255 if is_uint8 else frame)
    state = torch.as_tensor(obs['robot']['qpos'], device=device, dtype=torch.float32)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape != (1,14) or not torch.isfinite(state).all():
        raise ValueError('robot.qpos must be finite [1,14] in official joint order')
    return torch.stack(frames, 1), state

def evaluate(env, model, obs):
    """Predict/plan once, then send one 14-D absolute joint command per call.

Returns (observation, info, truncated, inference_times), matching the pinned official
code. The evaluator checks task success separately; termination is not timeout.
Never queries hidden success/scene state
in this transport adapter. The SDK/evaluator owns execution timing.
"""
    if model._submission_steps >= model._submission_budget:
        raise RuntimeError('Episode command budget exhausted; reset_model is required')
    if str(model.device).startswith('cuda'):
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        if not model._plan_attempted:
            model.set_language(model._submission_task)
            images, state = encode_obs(obs, model.device)
            model.update_observation_window(images, state)
            from .planning import PlanningContext
            context = PlanningContext(model._submission_task, model._planning_robot, state)
            model.plan_trajectory(context, model.predict_parameters(images, state))
        if model.has_pending_actions():
            actions = np.asarray(model.get_action(count=1), dtype=np.float32)
            if actions.shape != (1,14):
                raise ValueError(f'Planner returned {actions.shape}, expected (1,14)')
            action = actions[0]
        else:
            if model._last_action is None:
                raise RuntimeError('Planner produced no command')
            action = np.asarray(model._last_action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Joint command must be finite [14]')
        command = torch.as_tensor(action, device=model._action_device).unsqueeze(0)
    if str(model.device).startswith('cuda'):
        torch.cuda.synchronize(model.device)
    inference_times = [time.perf_counter()-started]
    observation, _, _, truncated, info = env.step(command)
    model._submission_steps += 1
    return observation, info, truncated, inference_times

def reset_model(model, seed=None, **kwargs):
    """Start a new episode with empty trajectory and a fresh command counter."""
    model.reset()
    model._submission_steps = 0
