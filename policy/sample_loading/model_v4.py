"""Existing angle spatial features supply a zero-initialized rack XY residual."""
import torch
from torch import nn
from .base_model import SampleLoadingPerception


class PerceptionV4(SampleLoadingPerception):
    def __init__(self, residual=False):
        super().__init__(pretrained=False)
        self.residual_enabled = residual
        if residual:
            self.rack_residual = nn.Sequential(nn.Linear(1292, 128), nn.ReLU(), nn.Linear(128, 2))
            nn.init.zeros_(self.rack_residual[-1].weight)
            nn.init.zeros_(self.rack_residual[-1].bias)

    def forward(self, images, qpos):
        if not self.residual_enabled:
            return super().forward(images, qpos)
        if images.ndim != 5 or images.shape[1:3] != (3, 3) or qpos.shape[-1] != 14:
            raise ValueError('Expected Bx3x3xHxW RGB and Bx14 qpos')
        images = (images-self.rgb_mean)/self.rgb_std
        feature = torch.cat([torch.relu(enc(images[:, i])) for i, enc in enumerate(self.encoders)], dim=1)
        joints = (torch.cat([qpos[:, :6], qpos[:, 7:13]], dim=1)-self.qpos_mean)/self.qpos_std
        positions = self.position(torch.cat([feature.mean((2, 3)), joints], dim=1))*self.position_std+self.position_mean
        spatial = torch.cat([self.spatial(feature).flatten(1), joints], dim=1)
        delta = self.rack_residual(spatial)
        positions = torch.cat([positions[:, :3], positions[:, 3:5]+delta, positions[:, 5:]], dim=1)
        raw = self.angles(spatial).reshape(-1, 2, 2)
        unit = nn.functional.normalize(raw, dim=-1, eps=1e-6)
        return {'positions':positions, 'angle_raw':raw, 'angles':unit,
                'parameters':torch.cat([positions, unit.flatten(1)], dim=1)}


def configure(model, unfreeze_blocks):
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    heads = list(model.position.parameters())
    if model.residual_enabled:
        heads += list(model.rack_residual.parameters())
    blocks = [p for enc in model.encoders for p in enc.denseblock4.parameters()]
    for p in heads:
        p.requires_grad_(True)
    for p in blocks:
        p.requires_grad_(unfreeze_blocks)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
    return heads, [p for p in blocks if p.requires_grad]


def task_loss(output, target, weight):
    p = output['positions'].float()
    huber = lambda x: nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none').mean(1)
    losses = huber((p[:, :3]-target[:, :3])/.005)
    losses += 2*huber((p[:, 3:5]-target[:, 3:5])/.002)
    losses += .5*huber((p[:, 5:6]-target[:, 5:6])/.002)
    losses += .25*(output['angle_raw'].float().flatten(1)-target[:, 6:]).square().mean(1)
    return (losses*weight).mean()


def augment(images):
    count = images.shape[0]
    shape = (count, 1, 1, 1, 1)
    uniform = lambda low, high: torch.empty(shape, device=images.device).uniform_(low, high)
    active = torch.rand(shape, device=images.device) < .5
    value = images * uniform(.8, 1.2)
    mean = value.mean((-1, -2), keepdim=True)
    value = (value-mean)*uniform(.8, 1.2)+mean
    value = value.clamp(0, 1).pow(uniform(.9, 1.1))
    gray = (value*torch.tensor([.299,.587,.114], device=value.device)[None,None,:,None,None]).sum(2, keepdim=True)
    value = gray+(value-gray)*uniform(.8, 1.2)
    exposure = torch.empty((count, 3, 1, 1, 1), device=value.device).uniform_(.95, 1.05)
    value = value*exposure+torch.randn_like(value)*uniform(0, .01)
    return torch.where(active, value.clamp(0, 1), images)
