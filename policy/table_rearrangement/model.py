from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights

PREDICTION_DIM = 15
INPUT_HEIGHT, INPUT_WIDTH = 480, 640
WORLD_TO_CM = 100.0


def independent_parameters_from_targets(targets):
    return torch.stack((targets["fork_grasp_pose"][:, :3, 3], targets["spoon_grasp_pose"][:, :3, 3], targets["plate_pose"][:, :3, 3]), 1) * WORLD_TO_CM


def restore_trajectory_points(absolute_cm):
    return absolute_cm / WORLD_TO_CM


def orientation_sincos_from_targets(targets):
    values=[]
    for key in ("fork_grasp_pose", "spoon_grasp_pose", "plate_pose"):
        rotation=targets[key][:, :3, :3]
        yaw=torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        values.append(torch.stack((torch.sin(yaw), torch.cos(yaw)), -1))
    return torch.stack(values, 1)


class ParaPred(nn.Module):
    """Three-view DenseNet-121 regressor plus policy-owned assembly program."""
    def __init__(self, pretrained=False, device="cpu", action_chunk=10, ik_attempts=5):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbones = nn.ModuleList([densenet121(weights=weights).features for _ in range(3)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(3072, 2048), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(1024, 9),
        )
        self.orientation_head = nn.Sequential(
            nn.Conv2d(3072, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(),
            nn.Linear(64 * 4 * 5, 256), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(256, 6),
        )
        nn.init.zeros_(self.orientation_head[-1].weight)
        with torch.no_grad(): self.orientation_head[-1].bias.copy_(torch.tensor([0., 1., 0., 1., 0., 1.]))
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
        fused_map=torch.cat(maps, 1)
        raw = self.head(torch.cat([self.pool(f).flatten(1) for f in maps], -1))
        points = restore_trajectory_points(raw.view(b, 3, 3))
        orientation_raw=self.orientation_head(fused_map).view(b, 3, 2)
        orientation_sincos=torch.nn.functional.normalize(orientation_raw, dim=-1, eps=1e-6)
        theta=torch.atan2(orientation_sincos[..., 0], orientation_sincos[..., 1])
        poses = []
        for index in range(3):
            pose = torch.eye(4, device=self.device).expand(b, -1, -1).clone()
            c, sn=torch.cos(theta[:, index]), torch.sin(theta[:, index])
            if index < 2:
                # Fork and spoon grasp frames point the tool z-axis downward.
                pose[:, :3, :3] = 0
                pose[:, 0, 0]=c; pose[:, 0, 1]=sn
                pose[:, 1, 0]=sn; pose[:, 1, 1]=-c
                pose[:, 2, 2]=-1
            else:
                # Plate is upright and only its planar yaw is independent.
                pose[:, 0, 0]=c; pose[:, 0, 1]=-sn
                pose[:, 1, 0]=sn; pose[:, 1, 1]=c
            pose[:, :3, 3] = points[:, index]
            poses.append(pose)
        return {
            "left_arm_init_qpos": state[:, :6], "right_arm_init_qpos": state[:, 7:13],
            "fork_grasp_pose": poses[0], "spoon_grasp_pose": poses[1], "plate_pose": poses[2],
            "left_arm_base_pose": self.left_base_pose.expand(b, -1, -1),
            "right_arm_base_pose": self.right_base_pose.expand(b, -1, -1),
            "_trajectory_points": points, "_independent_parameters": raw.view(b, 3, 3),
            "_orientation_raw": orientation_raw, "_orientation_sincos": orientation_sincos,
            "_orientation_theta": theta,
        }

    def training_loss(self, images, targets, state=None):
        from .losses import parameter_supervision_loss
        if state is None: state = self.state_from_targets(targets)
        return parameter_supervision_loss(self(images, state), targets)

    @torch.inference_mode()
    def predict_parameters(self, images, state): self.eval(); return self(images, state)

    def plan_trajectory(self, env, prediction):
        """Build the task program from predicted fork, spoon and plate poses only.

        The official action graph is used as a deterministic motion template.
        Before graph expansion its object-pose inputs are replaced by network
        predictions inside a fresh policy-owned planning context.
        """
        from robosynchallenge.tasks.table_rearrangement.action_bank import TableRearrangementActionBank

        base = env
        self._plan_attempted = True
        predicted = {
            key: prediction[key][0].detach().cpu().numpy().astype(np.float32)
            for key in ("fork_grasp_pose", "spoon_grasp_pose", "plate_pose")
        }
        saved = {key: base.affordance_datas.get(key) for key in predicted}
        # A utensil's long-axis yaw has a pi ambiguity, while only one of the
        # two equivalent wrist postures may be reachable. Keep the official
        # action graph and try the raw prediction first. If only its outer
        # pre-approach is outside the workspace, retain every official stage
        # but shorten pre2's 5 cm clearance to 3 cm. No object truth or expert
        # trajectory is read.
        short_approach_config = deepcopy(base.action_config)
        for arm in ("left_arm", "right_arm"):
            grasp_name = "fork_grasp" if arm == "left_arm" else "spoon_grasp"
            for node in short_approach_config["node"][arm]:
                if grasp_name in node:
                    infos = node[grasp_name]["kwargs"]["affordance_infos"]
                    infos[2]["valid_funcs_name_kwargs_proc"][0]["pass_processes"][0]["kwargs"]["offset_value"] = 0.03
                    break
        candidates = [("raw", predicted, base.action_config),
                      ("raw_pre2_3cm", predicted, short_approach_config)]
        for name, keys in (("fork_pi", ("fork_grasp_pose",)),
                           ("spoon_pi", ("spoon_grasp_pose",)),
                           ("both_pi", ("fork_grasp_pose", "spoon_grasp_pose"))):
            candidate = {key: value.copy() for key, value in predicted.items()}
            for key in keys:
                candidate[key][:3, 0] *= -1.0
                candidate[key][:3, 1] *= -1.0
            candidates.append((name, candidate, base.action_config))
            candidates.append((name + "_pre2_3cm", candidate, short_approach_config))
        actions = None
        selected = None
        try:
            for candidate_name, candidate, action_config in candidates:
                base.affordance_datas.update(candidate)
                base.initialize_graph(TableRearrangementActionBank, action_config)
                actions = base.plan_graph()
                if actions is not None:
                    selected = candidate_name
                    break
        finally:
            for key, value in saved.items():
                if value is None:
                    base.affordance_datas.pop(key, None)
                else:
                    base.affordance_datas[key] = value
        if actions is None:
            raise RuntimeError("ParaPred table_rearrangement template was infeasible")
        actions = torch.as_tensor(actions).detach().cpu().numpy()
        if actions.ndim == 3:
            actions = actions[:, 0]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"unexpected table_rearrangement action shape {actions.shape}")
        self._action_queue.clear()
        self._action_queue.extend(np.asarray(actions, np.float32))
        self._last_action = np.asarray(actions[-1], np.float32).copy()
        self._event_lengths = (len(actions),)
        print(
            f"ParaPred table_rearrangement fork={predicted['fork_grasp_pose'][:3,3].tolist()} "
            f"spoon={predicted['spoon_grasp_pose'][:3,3].tolist()} plate={predicted['plate_pose'][:3,3].tolist()} "
            f"ik_variant={selected} actions={len(actions)}",
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
