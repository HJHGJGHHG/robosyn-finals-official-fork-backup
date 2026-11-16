from collections import deque
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights

PREDICTION_DIM = 10
INPUT_HEIGHT, INPUT_WIDTH = 480, 640
WORLD_TO_CM = 100.0


def independent_parameters_from_targets(targets):
    return targets["button_pose"][:, :3, 3] * WORLD_TO_CM


def restore_trajectory_points(absolute_cm):
    return absolute_cm / WORLD_TO_CM


class ParaPred(nn.Module):
    """Three-view DenseNet-121 regressor plus policy-owned assembly program."""
    def __init__(self, pretrained=False, device="cpu", action_chunk=10, ik_attempts=5):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbones = nn.ModuleList([densenet121(weights=weights).features for _ in range(3)])
        # Preserve spatial layout for point localization. Global average
        # pooling made the previous model translation-invariant and reduced it
        # to predicting the dataset mean button position.
        self.head = nn.Sequential(
            nn.Conv2d(3072, 256, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(),
            nn.Linear(128 * 4 * 5, 512), nn.LeakyReLU(0.1), nn.Dropout(.1),
            nn.Linear(512, 3),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([63.0, 0.0, 83.0]))
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.register_buffer("left_base_pose", self._base_pose([.233, .300, .835]))
        self.register_buffer("right_base_pose", self._base_pose([.233, -.300, .835]))
        self.action_chunk, self.ik_attempts = int(action_chunk), int(ik_attempts)
        self.observation_window = self.language = None
        self._action_queue, self._event_queues = deque(), deque()
        self._event_lengths = None
        self._plan_attempted = False
        self._last_action = None
        self.to(device)

    @staticmethod
    def _base_pose(xyz):
        pose = torch.eye(4); pose[:3, 3] = torch.tensor(xyz); return pose

    @property
    def device(self): return next(self.parameters()).device

    def set_language(self, value): self.language = value
    def update_observation_window(self, images, state=None): self.observation_window = {"images": images, "state": state}
    def reset(self):
        self.observation_window = self.language = None
        self._action_queue.clear(); self._event_queues.clear()
        self._event_lengths = None; self._plan_attempted = False; self._last_action = None
    def reset_obsrvationwindows(self): self.reset()

    @staticmethod
    def state_from_targets(targets):
        left, right = targets["left_arm_init_qpos"], targets["right_arm_init_qpos"]
        z = torch.zeros((*left.shape[:-1], 1), device=left.device, dtype=left.dtype)
        return torch.cat((left, z, right, z), -1)

    def forward(self, images, state):
        if images.ndim == 4: images = images[:, None]
        if images.ndim != 5 or tuple(images.shape[1:3]) != (3, 3):
            raise ValueError(f"expected [B,3,3,H,W], got {tuple(images.shape)}")
        b, _, _, h, w = images.shape
        if (h, w) != (INPUT_HEIGHT, INPUT_WIDTH): raise ValueError(f"expected 640x480, got {w}x{h}")
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if state.ndim == 1: state = state[None]
        x = images.to(self.device, dtype=torch.float32)
        if x.max() > 1.5: x = x / 255.0
        x = (x - self.image_mean) / self.image_std
        maps = [torch.relu(self.backbones[i](x[:, i])) for i in range(3)]
        raw = self.head(torch.cat(maps, dim=1))
        point = restore_trajectory_points(raw)
        button_pose = torch.eye(4, device=self.device).expand(b, -1, -1).clone()
        button_pose[:, :3, 3] = point
        return {
            "left_arm_init_qpos": state[:, :6], "right_arm_init_qpos": state[:, 7:13],
            "button_pose": button_pose,
            "left_arm_base_pose": self.left_base_pose.expand(b, -1, -1),
            "right_arm_base_pose": self.right_base_pose.expand(b, -1, -1),
            "_trajectory_points": point, "_independent_parameters": raw,
        }

    def training_loss(self, images, targets, state=None):
        from .losses import parameter_supervision_loss
        if state is None: state = self.state_from_targets(targets)
        return parameter_supervision_loss(self(images, state), targets)

    @torch.inference_mode()
    def predict_parameters(self, images, state): self.eval(); return self(images, state)

    def plan_trajectory(self, env, prediction):
        """Build the task program from the predicted button xyz only.

        The official action graph is used as a deterministic motion template.
        Before graph expansion its object-pose inputs are replaced by network
        predictions inside a fresh policy-owned planning context.
        """
        from robosynchallenge.tasks.click_bell.action_bank import ClickBellActionBank

        base = env
        self._plan_attempted = True
        predicted = {
            "button_pose": prediction["button_pose"][0].detach().cpu().numpy().astype(np.float32)
        }
        saved = {key: base.affordance_datas.get(key) for key in predicted}
        try:
            base.affordance_datas.update(predicted)
            base.button_pose_orig = predicted["button_pose"].copy()
            base.initialize_graph(ClickBellActionBank, base.action_config)
            actions = base.plan_graph()
        finally:
            for key, value in saved.items():
                if value is None:
                    base.affordance_datas.pop(key, None)
                else:
                    base.affordance_datas[key] = value
        if actions is None:
            raise RuntimeError("ParaPred click_bell template was infeasible")
        actions = torch.as_tensor(actions).detach().cpu().numpy()
        if actions.ndim == 3:
            actions = actions[:, 0]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"unexpected click_bell action shape {actions.shape}")
        self._action_queue.clear()
        self._action_queue.extend(np.asarray(actions, np.float32))
        self._last_action = np.asarray(actions[-1], np.float32).copy()
        self._event_lengths = (len(actions),)
        print(
            f"ParaPred click_bell button={predicted['button_pose'][:3,3].tolist()} actions={len(actions)}",
            flush=True,
        )
        return np.asarray(actions, np.float32)

    def plan_expert_trajectory(self, env, prediction): return self.plan_trajectory(env, prediction)
    def has_pending_actions(self): return bool(self._action_queue or self._event_queues)
    def get_action(self, count=None):
        out, count = [], self.action_chunk if count is None else count
        while self._action_queue and len(out) < count:
            self._last_action = np.asarray(self._action_queue.popleft(), np.float32).copy(); out.append(self._last_action)
        return np.asarray(out, np.float32)
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint.get("model", checkpoint))
