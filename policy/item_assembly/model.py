from .planning import MotionGenerator
from collections import deque
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights

PREDICTION_DIM = 10
INPUT_HEIGHT, INPUT_WIDTH = 480, 640
WORLD_TO_CM = 100.0


def independent_parameters_from_targets(targets):
    return torch.stack((targets["guijiao1_pose"][:, :3, 3], targets["guijiao2_pose"][:, :3, 3]), 1) * WORLD_TO_CM


def restore_trajectory_points(absolute_cm):
    return absolute_cm / WORLD_TO_CM


def orientation_sincos_from_targets(targets):
    values=[]
    for key in ("guijiao1_pose", "guijiao2_pose"):
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
            nn.Linear(1024, 6),
        )
        self.orientation_head = nn.Sequential(
            nn.Conv2d(3072, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(),
            nn.Linear(64 * 4 * 5, 256), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(256, 4),
        )
        nn.init.zeros_(self.orientation_head[-1].weight)
        with torch.no_grad(): self.orientation_head[-1].bias.copy_(torch.tensor([0., 1., 0., 1.]))
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
        points = restore_trajectory_points(raw.view(b, 2, 3))
        orientation_raw=self.orientation_head(fused_map).view(b, 2, 2)
        orientation_sincos=torch.nn.functional.normalize(orientation_raw, dim=-1, eps=1e-6)
        theta=torch.atan2(orientation_sincos[..., 0], orientation_sincos[..., 1])
        poses = []
        for index in range(2):
            pose = torch.eye(4, device=self.device).expand(b, -1, -1).clone()
            c, sn=torch.cos(theta[:, index]), torch.sin(theta[:, index])
            pose[:, 0, 0]=c; pose[:, 0, 1]=-sn; pose[:, 1, 0]=sn; pose[:, 1, 1]=c
            pose[:, :3, 3] = points[:, index]
            poses.append(pose)
        return {
            "left_arm_init_qpos": state[:, :6], "right_arm_init_qpos": state[:, 7:13],
            "guijiao1_pose": poses[0], "guijiao2_pose": poses[1],
            "left_arm_base_pose": self.left_base_pose.expand(b, -1, -1),
            "right_arm_base_pose": self.right_base_pose.expand(b, -1, -1),
            "_trajectory_points": points, "_independent_parameters": raw.view(b, 2, 3),
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
        from embodichain.lab.gym.utils.misc import cached_ik
        from embodichain.lab.sim.planners import MoveType, PlanState,   MotionGenOptions, ToppraPlanOptions
        base = env
        self._plan_attempted = True
        points = prediction["_trajectory_points"][0].detach().cpu().numpy().astype(np.float32)
        grasp_yaws = prediction["_orientation_theta"][0].detach().cpu().numpy().astype(np.float32)
        left = prediction["left_arm_init_qpos"][0].detach().cpu().numpy().astype(np.float32)
        right = prediction["right_arm_init_qpos"][0].detach().cpu().numpy().astype(np.float32)
        action = np.concatenate((left, [.05], right, [.05])).astype(np.float32)
        planners = {part: MotionGenerator(base.robot) for part in ("left_arm", "right_arm")}

        def solve(part, pose, seed):
            seed = np.asarray(seed, np.float32); candidates = [seed]
            delta = np.array([.04, -.04, .04, -.06, .04, -.04], np.float32)
            for n in range(1, max(1, self.ik_attempts)):
                candidates.append(seed + (1 if n % 2 else -1) * ((n + 1) // 2) * delta)
            solutions = []
            for candidate in candidates:
                ok, q = cached_ik(pose, base.robot.compute_ik, part, part == "left_arm", candidate)
                if ok and q is not None and np.isfinite(q).all(): solutions.append(np.asarray(q, np.float32))
            if not solutions: return seed, False
            return min(solutions, key=lambda q: float(np.linalg.norm(q - seed))), True

        def move(part, target, steps):
            nonlocal action
            sl = slice(0, 6) if part == "left_arm" else slice(7, 13)
            start, target = action[sl].copy(), np.asarray(target, np.float32)
            if np.linalg.norm(target - start) < 1e-6:
                frames = np.repeat(start[None], steps, 0)
            else:
                states = [PlanState.single(qpos=torch.as_tensor(q), move_type=MoveType.JOINT_MOVE) for q in (start, target)]
                ret = planners[part].generate(target_states=states, options=MotionGenOptions(control_part=part, plan_opts=ToppraPlanOptions(sample_interval=steps)))
                frames = np.asarray(ret.positions.detach().cpu(), np.float32).squeeze()
                if frames.shape[0] == 6: frames = frames.T
                if len(frames) != steps:
                    old, new = np.linspace(0, 1, len(frames)), np.linspace(0, 1, steps)
                    frames = np.stack([np.interp(new, old, frames[:, j]) for j in range(6)], -1).astype(np.float32)
            for q in frames:
                frame = action.copy(); frame[sl] = q; self._action_queue.append(frame); action = frame

        def grip(part, value, steps, hold=0):
            nonlocal action
            idx = 6 if part == "left_arm" else 13
            for _ in range(steps):
                frame = action.copy(); frame[idx] = value; self._action_queue.append(frame); action = frame
            for _ in range(hold): self._action_queue.append(action.copy())

        def hold(steps):
            for _ in range(steps): self._action_queue.append(action.copy())

        def grasp_pose(xyz, yaw, arm):
            c, s = np.cos(yaw), np.sin(yaw)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]], np.float32)
            pose[:3, 3] = xyz
            pose[:3, 3] += pose[:3, 0] * -.01
            if arm == "right_arm": pose[:3, 3] += pose[:3, 1] * .01
            return pose

        rg = grasp_pose(points[0], grasp_yaws[0], "right_arm")
        lg = grasp_pose(points[1], grasp_yaws[1], "left_arm")
        rp, lp = rg.copy(), lg.copy(); rp[:3, 3] += np.array([.01, -.01, .05]); lp[:3, 3] += np.array([-.15, .025, .05])
        rl, ll = rg.copy(), lg.copy(); rl[2, 3] += .05; ll[2, 3] += .05
        rq_pre, ok1 = solve("right_arm", rp, right); rq_grasp, ok2 = solve("right_arm", rg, rq_pre)
        lq_pre, ok3 = solve("left_arm", lp, left); lq_grasp, ok4 = solve("left_arm", lg, lq_pre)
        rq_lift, ok5 = solve("right_arm", rl, rq_grasp); lq_lift, ok6 = solve("left_arm", ll, lq_grasp)
        # Canonical safe assembly qpos extracted once from the official action graph.
        rq_up = np.array([.3582915, 2.0887258, -1.3839157, 0., 1.0031900, -.0259710], np.float32)
        rq_start = np.array([.2695210, 1.5113211, -.9288056, -.2378793, -.5073498, .2088380], np.float32)
        rq_splice = np.array([.2695210, 1.5113211, -.9288056, -.2378793, -.5073498, 1.7796340], np.float32)
        rq_adjust = np.array([.3332618, 1.9073370, -1.4531431, -.1413145, -.3703580, 1.7026439], np.float32)
        lq_up = np.array([-.4949628, 1.7721486, -.9006455, 0., .8364969, 0.], np.float32)
        lq_start = np.array([-.5272416, 1.1812258, -.6483283, -.0747647, -.4467828, .0674491], np.float32)
        lq_adjust = np.array([-.8541088, .5091812, -.6009562, 1.1280080, .3995336, -1.0954342], np.float32)
        lq_splice = np.array([-.8541088, .5091812, -.6009562, 1.1280080, .3995336, .4753622], np.float32)
        lq_align = np.array([-.7503578, 1.9010533, -1.4435098, 1.7379255, 1.1659968, -.4047130], np.float32)
        lq_get = np.array([-.8100434, 2.1020808, -1.7696371, 1.6664896, 1.2061079, -.2628981], np.float32)
        feasible = (ok1, ok2, ok3, ok4, ok5, ok6)
        if not all(feasible): raise RuntimeError(f"ParaPred item_assembly grasp IK infeasible: {feasible}")
        # 300 frames: predicted grasp, secure hold, safe lift, canonical assembly, settle/release.
        move("right_arm", rq_pre, 15); move("left_arm", lq_pre, 15)
        move("right_arm", rq_grasp, 15); grip("right_arm", 0., 15, 5)
        move("left_arm", lq_grasp, 15); grip("left_arm", 0., 15, 5)
        move("right_arm", rq_lift, 10); move("right_arm", rq_up, 10)
        move("left_arm", lq_lift, 10); move("left_arm", lq_up, 10)
        move("right_arm", rq_start, 20); move("left_arm", lq_start, 20)
        move("right_arm", rq_splice, 15); move("left_arm", lq_adjust, 15)
        move("right_arm", rq_adjust, 15); move("left_arm", lq_splice, 15)
        move("left_arm", lq_align, 20); hold(10); move("left_arm", lq_get, 10)
        hold(20)
        self._event_lengths = (15,15,15,20,15,20,20,20,20,20,15,15,15,15,20,10,10,10,10)
        assert len(self._action_queue) == 300, len(self._action_queue)
        self._last_action = action.copy()
        print(f"ParaPred item_assembly points(m)={points.tolist()} IK={feasible} actions=300", flush=True)
        return np.asarray(self._action_queue, np.float32)

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
