"""Reused three-camera RGB byte conversion; normalized observed 14D qpos."""
import torch
from .geometry import CAMERAS

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

