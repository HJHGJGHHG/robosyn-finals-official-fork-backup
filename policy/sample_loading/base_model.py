"""Three-view ParaPred adaptation: six positions, two independent angles, qpos."""
import torch
from torch import nn
from torchvision.models import densenet121
try:
    from torchvision.models import DenseNet121_Weights
except ImportError:  # torchvision < 0.13
    DenseNet121_Weights = None

class SampleLoadingPerception(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        if DenseNet121_Weights is None:
            make_encoder = lambda: densenet121(pretrained=pretrained).features
        else:
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            make_encoder = lambda: densenet121(weights=weights).features
        self.encoders=nn.ModuleList([make_encoder() for _ in range(3)])
        self.position=nn.Sequential(nn.Linear(3072+12,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,6))
        self.spatial=nn.Sequential(nn.Conv2d(3072,128,1),nn.ReLU(),nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,5)))
        self.angles=nn.Sequential(nn.Linear(1280+12,256),nn.ReLU(),nn.Linear(256,4))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,None,:,None,None])
        self.register_buffer('position_mean',torch.zeros(6));self.register_buffer('position_std',torch.ones(6))
        self.register_buffer('qpos_mean',torch.zeros(12));self.register_buffer('qpos_std',torch.ones(12))
    def set_normalization(self,stats):
        assert stats['fit_split']=='train'
        for name in ('position_mean','position_std'):
            getattr(self,name).copy_(torch.tensor(stats[name]))
        idx=[0,1,2,3,4,5,7,8,9,10,11,12]
        self.qpos_mean.copy_(torch.tensor(stats['qpos_mean'])[idx])
        self.qpos_std.copy_(torch.tensor(stats['qpos_std'])[idx].clamp_min(1e-4))
    def forward(self,images,qpos):
        if images.ndim!=5 or images.shape[1:3]!=(3,3) or qpos.shape[-1]!=14:raise ValueError('Expected Bx3x3xHxW RGB and Bx14 qpos')
        images=(images-self.rgb_mean)/self.rgb_std
        feature=torch.cat([torch.relu(enc(images[:,i])) for i,enc in enumerate(self.encoders)],dim=1)
        joints=torch.cat([qpos[:,:6],qpos[:,7:13]],dim=1)
        joints=(joints-self.qpos_mean)/self.qpos_std
        positions=self.position(torch.cat([feature.mean((2,3)),joints],dim=1))*self.position_std+self.position_mean
        raw=self.angles(torch.cat([self.spatial(feature).flatten(1),joints],dim=1)).reshape(-1,2,2)
        unit=nn.functional.normalize(raw,dim=-1,eps=1e-6)
        return {'positions':positions,'angle_raw':raw,'angles':unit,'parameters':torch.cat([positions,unit.flatten(1)],dim=1)}

def supervised_loss(output,target):
    # Position scale is centimeters; keep unnormalized angle supervision to
    # discourage an arbitrarily small vector despite pair normalization.
    position=nn.functional.smooth_l1_loss(output['positions']*100,target[:,:6]*100)
    angle=nn.functional.mse_loss(output['angle_raw'],target[:,6:].reshape(-1,2,2))
    return {'total':position+angle,'position_cm':position,'angle':angle}
