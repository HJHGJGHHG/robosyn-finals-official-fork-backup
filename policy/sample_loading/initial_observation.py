"""Freeze the initial RGB/qpos bundle after two matching pure observation reads."""
import copy
import hashlib
import numpy as np
import torch

CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
VERSION = 'initial-sync-v52-r1'


def clone_observation(value):
    """Copy simulator buffers before another render can reuse their storage."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, 'items'):
        return {k: clone_observation(v) for k, v in value.items()}
    return copy.deepcopy(value)


def rgb_bytes(obs, camera):
    value = torch.as_tensor(obs['sensor'][camera]['color']).detach().cpu()
    if value.ndim != 4 or value.shape[:3] != (1, 480, 640) or value.shape[-1] not in (3, 4):
        raise ValueError(f'Invalid RGB shape: {camera} {tuple(value.shape)}')
    value = value[0, :, :, :3]
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 255:
        raise ValueError(f'Invalid RGB range: {camera}')
    if value.dtype != torch.uint8:
        value = (value * (255 if value.max() <= 1.5 else 1)).clamp(0, 255).to(torch.uint8)
    return value.numpy().copy()


def fingerprint(obs):
    qpos = torch.as_tensor(obs['robot']['qpos']).detach().cpu().numpy()
    if qpos.shape != (1, 14) or not np.isfinite(qpos).all():
        raise ValueError('Expected finite qpos (1,14)')
    return dict(rgb_sha256={c: hashlib.sha256(rgb_bytes(obs, c).tobytes()).hexdigest()
                           for c in CAMERAS}, qpos=qpos.copy())


def synchronized_initial_observation(env, reset_obs, observer=None):
    """Return (owned observation, receipt); read RGB/qpos only, never step physics.

    observer(index, owned_observation) is optional evaluator instrumentation.
    Two extra reads are required; at most four are attempted before failing.
    """
    get_obs = getattr(env, 'get_obs', None)
    if not callable(get_obs):
        get_obs = env.get_wrapper_attr('get_obs')
    raw = clone_observation(reset_obs)
    initial = fingerprint(raw)
    if observer:
        observer(0, raw)
    previous = None
    frames = [initial['rgb_sha256']]
    for index in range(1, 5):
        current = clone_observation(get_obs())
        state = fingerprint(current)
        delta = float(np.abs(state['qpos'] - initial['qpos']).max())
        if delta > 1e-7:
            raise RuntimeError(f'initial_sync_qpos_changed:{delta}')
        frames.append(state['rgb_sha256'])
        if observer:
            observer(index, current)
        if previous is not None and previous == state['rgb_sha256']:
            return current, dict(version=VERSION, stable=True, extra_reads=index,
                                 frame_hashes=frames, qpos_max_delta=delta)
        previous = state['rgb_sha256']
    raise RuntimeError('observation_unstable_after_four_reads')
