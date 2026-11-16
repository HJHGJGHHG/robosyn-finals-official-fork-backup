"""Three DenseNet-121 views + joint state -> three world points in meters."""
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.checkpoint import checkpoint


class DrawerPerception(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.encoders = nn.ModuleList([densenet121(weights=weights).features for _ in range(3)])
        self.position = nn.Sequential(nn.Linear(3072+12,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,9))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,None,:,None,None])
        self.register_buffer('position_mean',torch.zeros(9))
        self.register_buffer('position_std',torch.ones(9))
        self.register_buffer('qpos_mean',torch.zeros(12))
        self.register_buffer('qpos_std',torch.ones(12))
        self.activation_checkpoint = False

    def set_normalization(self, stats):
        """Load train-only 9D position and 12D arm-qpos statistics."""
        if stats.get('fit_split') != 'train':
            raise ValueError('Normalization must be fitted on training data only')
        for name in ('position_mean','position_std','qpos_mean','qpos_std'):
            value=torch.as_tensor(stats[name],dtype=torch.float32)
            if value.shape!=getattr(self,name).shape or not torch.isfinite(value).all():
                raise ValueError('Invalid normalization: '+name)
            if name.endswith('std'): value=value.clamp_min(1e-4)
            getattr(self,name).copy_(value)

    def features(self, images, qpos):
        """Return B,3084 visual/joint features; used also by frozen-head preflight."""
        if images.ndim != 5 or images.shape[1:3] != (3,3) or qpos.shape != (len(images),14):
            raise ValueError('Wrong camera or joint layout')
        images = (images-self.rgb_mean)/self.rgb_std
        def encode(encoder, image):
            return torch.relu(encoder(image)).mean((2,3))
        views=[]
        for i,e in enumerate(self.encoders):
            if self.activation_checkpoint and self.training and next(e.parameters()).requires_grad:
                views.append(checkpoint(encode,e,images[:,i],use_reentrant=False))
            else: views.append(encode(e,images[:,i]))
        features = torch.cat(views,1)
        joints = torch.cat([qpos[:,:6],qpos[:,7:13]],1)
        joints = (joints-self.qpos_mean)/self.qpos_std.clamp_min(1e-4)
        return torch.cat([features,joints],1)

    def predict_features(self, features):
        """B,3084 -> B,9 meters, retaining float32 precision in world coordinates."""
        return self.position(features).float()*self.position_std+self.position_mean

    def forward(self, images, qpos):
        """B,3,3,H,W RGB [0,1], B,14 observed qpos -> B,9 world meters."""
        return self.predict_features(self.features(images,qpos))


def position_loss(prediction, target):
    """Equal-weight centimeter Smooth-L1 on 9D meter-valued predictions/labels."""
    return nn.functional.smooth_l1_loss(prediction*100,target*100,beta=1.)
