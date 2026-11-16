from .planning import MotionGenerator
from collections import deque
from copy import deepcopy
import os
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from .schema import POSE_KEYS, QPOS_KEYS

PREDICTION_DIM=8
INPUT_HEIGHT=480
INPUT_WIDTH=640
WORLD_TO_CM=100.0


def restore_trajectory_points(absolute_cm):
    """Convert absolute world XYZ centimetres to the metre trajectory interface."""
    return absolute_cm/WORLD_TO_CM


def independent_parameters_from_targets(targets):
    pen_grasp_cm=targets["pen_grasp_pose"][:,:3,3]*WORLD_TO_CM
    release_cm=targets["holder_pose"][:,:3,3]*WORLD_TO_CM
    return torch.stack((pen_grasp_cm,release_cm),dim=1)

def orientation_sincos_from_targets(targets,canonical_rotation):
    """World-Z yaw of the grasp frame relative to the canonical frame."""
    rotation=targets["pen_grasp_pose"][:,:3,:3]
    canonical_rotation=canonical_rotation.to(device=rotation.device,dtype=rotation.dtype)
    relative=rotation@canonical_rotation.transpose(0,1)
    theta=torch.atan2(relative[:,1,0],relative[:,0,0])
    return torch.stack((torch.sin(theta),torch.cos(theta)),dim=-1)

class ParaPred(nn.Module):
    def __init__(self,pretrained=False,device="cpu",action_chunk=10,ik_attempts=5):
        super().__init__()
        weights=DenseNet121_Weights.DEFAULT if pretrained else None
        # DenseNet-121 provides compact, feature-reusing ImageNet representations.
        self.backbones=nn.ModuleList([
            densenet121(weights=weights).features
            for _ in range(3)
        ])
        self.spatial_pool=nn.AdaptiveAvgPool2d((1,1))
        self.head=nn.Sequential(
            nn.Linear(3072,2048),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(2048,1024),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(1024,PREDICTION_DIM),
        )
        # Dedicated yaw branch keeps the 15x20 spatial feature layout.
        # The position head remains unchanged and can be frozen independently.
        self.orientation_head=nn.Sequential(
            nn.Conv2d(3072,128,kernel_size=1),nn.ReLU(),
            nn.Conv2d(128,64,kernel_size=3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,5)),nn.Flatten(),
            nn.Linear(64*4*5,256),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(256,2),
        )
        nn.init.zeros_(self.orientation_head[-1].weight)
        with torch.no_grad(): self.orientation_head[-1].bias.copy_(torch.tensor([0.0,1.0]))
        self.register_buffer("image_mean",torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("image_std",torch.tensor([.229,.224,.225]).view(1,3,1,1))
        left=torch.eye(4); left[:3,3]=torch.tensor([.233,.300,.835])
        right=torch.eye(4); right[:3,3]=torch.tensor([.233,-.300,.835])
        pen=torch.tensor([[.98806864,-.07361907,.13527976,.61716634],[.15393580,.44403064,-.88268954,-.09851620],[.00491441,.89298218,.45006531,.81789851],[0.,0.,0.,1.]])
        pen_grasp=torch.tensor([[.98806864,-.07361907,.13527976,.64816117],[.15393580,.44403064,-.88268954,-.10272503],[.00491441,.89298218,.45006531,.82254660],[0.,0.,0.,1.]])
        holder=torch.tensor([[1.,.00000568,-.00000030,.61330009],[-.00000030,.00004536,-1.,.13134208],[-.00000568,1.,.00004536,.81030339],[0.,0.,0.,1.]])
        self.register_buffer("left_base_pose",left)
        self.register_buffer("right_base_pose",right)
        self.register_buffer("canonical_pen_pose",pen)
        self.register_buffer("canonical_pen_grasp_pose",pen_grasp)
        self.register_buffer("canonical_holder_pose",holder)
        self.action_chunk=int(action_chunk)
        self.ik_attempts=int(ik_attempts)
        self.observation_window=None
        self.language=None
        self._action_queue=deque()
        self._event_queues=deque()
        self._event_lengths=None
        self._plan_attempted=False
        self._last_action=None
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_language(self,x):
        self.language=x

    def update_observation_window(self,images,state=None):
        self.observation_window={"images":images,"state":state}

    def reset(self):
        self.observation_window=None
        self.language=None
        self._action_queue.clear()
        self._event_queues.clear()
        self._event_lengths=None
        self._plan_attempted=False
        self._last_action=None

    def reset_obsrvationwindows(self):
        self.reset()

    @staticmethod
    def state_from_targets(targets):
        left=targets["left_arm_init_qpos"]
        right=targets["right_arm_init_qpos"]
        z=torch.zeros((*left.shape[:-1],1),device=left.device,dtype=left.dtype)
        return torch.cat((left,z,right,z),dim=-1)

    def forward(self,images,state):
        if images.ndim==4:
            images=images[:,None]
        if images.ndim!=5:
            raise ValueError(f"expected [B,V,3,H,W], got {images.shape}")
        b,v,c,h,w=images.shape
        if (v,c)!=(3,3):
            raise ValueError(f"ParaPred requires [B,3,3,H,W], got {images.shape}")
        if (h,w)!=(INPUT_HEIGHT,INPUT_WIDTH):
            raise ValueError(f"ParaPred requires native 640x480 RGB, got {w}x{h}")
        state=torch.as_tensor(state,device=self.device,dtype=torch.float32)
        if state.ndim==1:
            state=state[None]
        if state.shape[0]!=b or state.shape[-1]<13:
            raise ValueError(f"expected robot state [B,>=13], got {state.shape}")

        x=images.to(self.device,dtype=torch.float32)
        x=x/255 if x.max()>1.5 else x
        x=(x-self.image_mean)/self.image_std
        feature_maps=[torch.relu(self.backbones[i](x[:,i])) for i in range(3)]
        features=[self.spatial_pool(f).flatten(1) for f in feature_maps]
        raw=self.head(torch.cat(features,dim=-1))
        absolute_cm=raw[:,:6].view(b,2,3)
        orientation_raw=self.orientation_head(torch.cat(feature_maps,dim=1))
        orientation_sincos=torch.nn.functional.normalize(orientation_raw,dim=-1,eps=1e-6)
        theta=torch.atan2(orientation_sincos[:,0],orientation_sincos[:,1])
        points=restore_trajectory_points(absolute_cm)
        pen=self.canonical_pen_pose.expand(b,-1,-1).clone()
        pen_grasp=self.canonical_pen_grasp_pose.expand(b,-1,-1).clone()
        holder=self.canonical_holder_pose.expand(b,-1,-1).clone()
        zero=torch.zeros_like(theta); one=torch.ones_like(theta)
        rotz=torch.stack((
            torch.cos(theta),-torch.sin(theta),zero,
            torch.sin(theta), torch.cos(theta),zero,
            zero,zero,one),dim=-1).view(b,3,3)
        pen[:,:3,:3]=rotz@self.canonical_pen_pose[:3,:3]
        pen_grasp[:,:3,:3]=rotz@self.canonical_pen_grasp_pose[:3,:3]
        pen[:,:3,3]=points[:,0]+(self.canonical_pen_pose[:3,3]-self.canonical_pen_grasp_pose[:3,3])
        pen_grasp[:,:3,3]=points[:,0]
        holder[:,:3,3]=points[:,1]
        return {
            "left_arm_init_qpos":state[:,:6],
            "right_arm_init_qpos":state[:,7:13],
            "pen_pose":pen,
            "pen_grasp_pose":pen_grasp,
            "holder_pose":holder,
            "left_arm_base_pose":self.left_base_pose.expand(b,-1,-1),
            "right_arm_base_pose":self.right_base_pose.expand(b,-1,-1),
            "_trajectory_points":points,
            "_independent_parameters":absolute_cm,
            "_orientation_raw":orientation_raw,
            "_orientation_sincos":orientation_sincos,
            "_orientation_theta":theta,
        }

    def training_loss(self,images,targets,state=None):
        from .losses import parameter_supervision_loss
        if state is None:
            state=self.state_from_targets(targets)
        return parameter_supervision_loss(self(images,state),targets)

    @torch.inference_mode()
    def predict_parameters(self,images,state):
        self.eval()
        return self(images,state)

    def plan_trajectory(self,env,pred):
        """Build the policy-owned fixed-duration trajectory from model outputs."""
        from embodichain.lab.gym.utils.misc import (cached_ik, get_fk_xpos, get_offset_pose, get_aligned_pose,
            get_replaced_pose, get_rotated_pose, get_rotation_replaced_pose)
        from embodichain.lab.sim.planners import (MoveType,PlanState,
            MotionGenOptions,ToppraPlanOptions,)
        base=env
        toppra_planners={part:MotionGenerator(base.robot)
            for part in ("left_arm","right_arm")}
        self._plan_attempted=True
        points=pred["_trajectory_points"][0].detach().cpu().numpy()
        left=np.asarray(pred["left_arm_init_qpos"][0].detach().cpu(),dtype=np.float32)
        right=np.asarray(pred["right_arm_init_qpos"][0].detach().cpu(),dtype=np.float32)
        action=np.concatenate((left,[0.05],right,[0.05])).astype(np.float32)
        right_grasp_pose=np.asarray(pred["pen_grasp_pose"][0].detach().cpu(),dtype=np.float32)
        right_grasp_pose=get_aligned_pose(right_grasp_pose,align_vector=[0,0,-1],pose_axis="z")
        def pose_at(qpos,part,xyz,template=None):
            pose=(np.asarray(template,dtype=np.float32).copy() if template is not None else get_fk_xpos(qpos,part,base.robot.compute_fk).astype(np.float32))
            pose[:3,3]=np.asarray(xyz,dtype=np.float32); return pose
        def solve(part,pose,seed):
            seed=np.asarray(seed,dtype=np.float32).copy()
            seeds=[seed]
            delta=np.asarray([.04,-.04,.04,-.06,.04,-.04],dtype=np.float32)
            for attempt in range(1,max(1,self.ik_attempts)):
                scale=(attempt+1)//2; sign=1. if attempt%2 else -1.
                seeds.append(seed+sign*scale*delta)
            solutions=[]
            for candidate_seed in seeds:
                ok,qpos=cached_ik(pose,base.robot.compute_ik,part,part=="left_arm",candidate_seed)
                if ok and qpos is not None and np.isfinite(qpos).all():
                    solutions.append(np.asarray(qpos,dtype=np.float32))
            if not solutions:
                print(f"ParaPred warning: unreachable {part} waypoint {pose[:3,3].tolist()} after {len(seeds)} attempts; holding last valid target",flush=True)
                return seed,False
            return min(solutions,key=lambda value:float(np.linalg.norm(value-seed))),True
        def move(part,target,steps):
            # Use the same velocity/acceleration-constrained TOPPRA generator as
            # the expert action bank; only our model-derived endpoints differ.
            nonlocal action
            sl=slice(0,6) if part=="left_arm" else slice(7,13); start=action[sl].copy()
            target=np.asarray(target,dtype=np.float32)
            if float(np.linalg.norm(target-start))<1e-6:
                print(f"ParaPred TOPPRA fallback: zero-distance {part} move; holding {steps} frames",flush=True)
                for _ in range(steps):
                    self._action_queue.append(action.copy())
                return
            states=[PlanState.single(qpos=torch.as_tensor(q,dtype=torch.float32),move_type=MoveType.JOINT_MOVE)
                for q in (start,target)]
            ret=toppra_planners[part].generate(target_states=states,options=MotionGenOptions(
                control_part=part,plan_opts=ToppraPlanOptions(sample_interval=steps)))
            trajectory=np.asarray(ret.positions.detach().cpu(),dtype=np.float32).squeeze()
            if trajectory.ndim!=2:
                raise RuntimeError(f"unexpected TOPPRA trajectory shape {trajectory.shape}")
            if trajectory.shape[-1]==len(start):
                frames=trajectory
            elif trajectory.shape[0]==len(start):
                frames=trajectory.T
            else:
                raise RuntimeError(f"unexpected TOPPRA trajectory shape {trajectory.shape}")
            if len(frames)!=steps:
                if len(frames)<2:
                    raise RuntimeError(f"TOPPRA returned unusable trajectory with {len(frames)} frames")
                source_time=np.linspace(0.,1.,len(frames),dtype=np.float32)
                target_time=np.linspace(0.,1.,steps,dtype=np.float32)
                frames=np.stack([np.interp(target_time,source_time,frames[:,joint])
                    for joint in range(frames.shape[1])],axis=-1).astype(np.float32)
                print(f"ParaPred TOPPRA fallback: resampled {part} trajectory to {steps} frames",flush=True)
            for q in frames:
                frame=action.copy(); frame[sl]=q
                self._action_queue.append(frame); action=frame
        def move_cartesian(part,start_pose,end_pose,seed,steps):
            nonlocal action
            sl=slice(0,6) if part=="left_arm" else slice(7,13)
            q=np.asarray(seed,dtype=np.float32).copy()
            for alpha in np.linspace(1/steps,1,steps,dtype=np.float32):
                pose=np.asarray(start_pose,dtype=np.float32).copy()
                pose[:3,3]=start_pose[:3,3]+(end_pose[:3,3]-start_pose[:3,3])*alpha
                # Interpolate orientation while projecting back to SO(3).
                # Translation is prescribed independently, so equal endpoint Z
                # truly means constant world Z throughout the whole segment.
                blended=(1.-alpha)*start_pose[:3,:3]+alpha*end_pose[:3,:3]
                u,_,vt=np.linalg.svd(blended)
                rotation=u@vt
                if np.linalg.det(rotation)<0:
                    u[:,-1]*=-1; rotation=u@vt
                pose[:3,:3]=rotation
                q,ok=solve(part,pose,q)
                if not ok: break
                frame=action.copy(); frame[sl]=q
                self._action_queue.append(frame); action=frame
            return q
        def grip(part,value,move_steps,hold_steps,smooth=True):
            nonlocal action
            idx=6 if part=="left_arm" else 13; start=float(action[idx])
            for step in range(move_steps):
                frame=action.copy()
                alpha=float(step+1)/float(move_steps) if smooth else 1.0
                frame[idx]=start+(float(value)-start)*alpha
                self._action_queue.append(frame); action=frame
            # Let contacts settle while every other action dimension stays fixed.
            for _ in range(hold_steps):
                self._action_queue.append(action.copy())
        def hold(steps):
            for _ in range(steps):
                self._action_queue.append(action.copy())
        # Fixed trajectory template; only the two network points affect execution.
        grasp=points[0].astype(np.float32).copy()
        release=points[1].astype(np.float32).copy()
        print(f"ParaPred restored execution points (m): grasp={grasp.tolist()} release={release.tolist()}",flush=True)
        right_grasp_pose[:3,3]=grasp
        right_nominal_grasp_pose=right_grasp_pose.copy()
        # Six-stage low grasp: start 5 mm low, then raise 3 mm five times.
        right_grasp_pose=get_offset_pose(right_grasp_pose,-.005,direction="z",mode="extrinsic")
        right_unblock_pose=get_offset_pose(right_grasp_pose.copy(),.003,direction="z",mode="extrinsic")
        right_regrasp_pose=get_offset_pose(right_grasp_pose.copy(),.006,direction="z",mode="extrinsic")
        right_regrasp2_pose=get_offset_pose(right_grasp_pose.copy(),.009,direction="z",mode="extrinsic")
        right_regrasp3_pose=get_offset_pose(right_grasp_pose.copy(),.012,direction="z",mode="extrinsic")
        right_regrasp4_pose=get_offset_pose(right_grasp_pose.copy(),.015,direction="z",mode="extrinsic")

        right_pre_pose=get_offset_pose(right_grasp_pose.copy(),-.10,direction="z",mode="intrinsic")
        right_pre_pose=get_rotated_pose(right_pre_pose,30,rot_axis="y",mode="intrinsic",degrees=True)
        print(f"ParaPred first segment pregrasp={right_pre_pose[:3,3].tolist()} second segment grasp={right_grasp_pose[:3,3].tolist()}",flush=True)
        # Input-independent general handover template. The network controls only
        # the grasp point and final release point.
        right_up_pose=get_replaced_pose(right_nominal_grasp_pose,[.60,-.15],axis_str_replace="xy")
        # Raise the complete handover platform by 10 cm.  The lift happens on
        # the approach, so neither arm waits until the handover point to rise.
        right_up_pose=get_offset_pose(right_up_pose,.30,direction="z",mode="extrinsic")
        right_up_pose=get_rotated_pose(right_up_pose,45,rot_axis="y",mode="intrinsic",degrees=True)
        right_handover_pose=get_replaced_pose(right_up_pose.copy(),[.65,.05],axis_str_replace="xy")
        right_handover_pose=get_offset_pose(right_handover_pose,.02,direction="z",mode="extrinsic")
        right_handover_pose=get_rotation_replaced_pose(right_handover_pose,-135,rot_axis="z",mode="extrinsic",degrees=True)
        right_handover_pose=get_rotated_pose(right_handover_pose,-90,rot_axis="y",mode="intrinsic",degrees=True)
        right_handover_pose=get_rotated_pose(right_handover_pose,45,rot_axis="z",mode="intrinsic",degrees=True)
        # Match the expert post-handover retreat: clear upward first and move
        # away in world Y before the left arm starts transporting the pen.
        right_back_pose=get_offset_pose(right_handover_pose.copy(),.10,direction="z",mode="extrinsic")
        right_back_pose=get_offset_pose(right_back_pose,-.05,direction="y",mode="extrinsic")

        left_takeover_pose=get_offset_pose(right_handover_pose.copy(),-.08,direction="x",mode="intrinsic")
        left_takeover_pose=get_rotated_pose(left_takeover_pose,-90,rot_axis="x",mode="intrinsic",degrees=True)
        left_takeover_pose=get_offset_pose(left_takeover_pose,.02,direction="z",mode="intrinsic")
        left_pre_pose=get_offset_pose(left_takeover_pose.copy(),-.05,direction="z",mode="intrinsic")
        left_move_pose=get_rotation_replaced_pose(left_takeover_pose.copy(),-30,rot_axis="z",mode="extrinsic",degrees=True)
        left_move_pose=get_rotated_pose(left_move_pose,-90,rot_axis="y",mode="intrinsic",degrees=True)
        left_move_pose=get_rotated_pose(left_move_pose,180,rot_axis="x",mode="intrinsic",degrees=True)
        left_move_pose=get_rotated_pose(left_move_pose,-10,rot_axis="y",mode="intrinsic",degrees=True)
        # release is the holder/object target, not the left gripper origin.
        # Reproduce the expert affordance semantics: holder XY positions the
        # placement template, while the gripper keeps its fixed orientation and
        # receives the 2 cm local-z grasp offset.  Both objects move together
        # with table height, already carried by grasp_z; only apply the residual
        # release-z displacement so that height is not counted twice.
        left_move_pose=get_replaced_pose(left_move_pose,release[:2],axis_str_replace="xy")
        expected_release_z=(
            float(self.canonical_holder_pose[2,3])
            + float(grasp[2]-float(self.canonical_pen_grasp_pose[2,3]))
        )
        release_z_residual=float(release[2])-expected_release_z
        left_move_pose[2,3]+=release_z_residual
        left_place_pose=get_offset_pose(
            left_move_pose.copy(),.02,direction="z",mode="intrinsic")
        # Keep the final release uniformly above the previous insertion pose to
        # avoid driving the pen or gripper into the holder collision geometry.
        left_place_pose[2,3]+=.05
        # Never descend after takeover: raise the release endpoint to at least
        # the elevated handover height.  The transfer and final approach are
        # therefore horizontal or upward in world Z.
        left_place_pose[2,3]=max(float(left_place_pose[2,3]),float(left_takeover_pose[2,3]))
        # Tail motion is L-shaped without an arbitrary clearance: after
        # takeover, preserve the current world Z while translating to the
        # release EEF XY; only then change Z at fixed XY to place/release.
        left_move_pose=left_place_pose.copy()
        left_move_pose[2,3]=left_takeover_pose[2,3]
        print(f"ParaPred release semantics: holder={release.tolist()} expected_holder_z={expected_release_z:.6f} residual_z={release_z_residual:.6f} eef_xy_transfer={left_move_pose[:3,3].tolist()} eef_place={left_place_pose[:3,3].tolist()}",flush=True)

        rq_pre,ok_rpre=solve("right_arm",right_pre_pose,right)
        rq_grasp,ok_rgrasp=solve("right_arm",right_grasp_pose,rq_pre)
        rq_unblock,ok_runblock=solve("right_arm",right_unblock_pose,rq_grasp)
        rq_regrasp,ok_rregrasp=solve("right_arm",right_regrasp_pose,rq_unblock)
        rq_regrasp2,ok_rregrasp2=solve("right_arm",right_regrasp2_pose,rq_regrasp)
        rq_regrasp3,ok_rregrasp3=solve("right_arm",right_regrasp3_pose,rq_regrasp2)
        rq_regrasp4,ok_rregrasp4=solve("right_arm",right_regrasp4_pose,rq_regrasp3)
        rq_up,ok_rup=solve("right_arm",right_up_pose,rq_regrasp4)
        rq_handover,ok_rhandover=solve("right_arm",right_handover_pose,rq_up)
        rq_back,ok_rback=solve("right_arm",right_back_pose,rq_handover)
        if not ok_rback:
            for retreat_scale in (.75,.50,.25,0.):
                candidate=right_handover_pose.copy()
                candidate=get_offset_pose(candidate,.10*retreat_scale,direction="z",mode="extrinsic")
                candidate=get_offset_pose(candidate,-.05*retreat_scale,direction="y",mode="extrinsic")
                rq_back,ok_rback=solve("right_arm",candidate,rq_handover)
                if ok_rback:
                    right_back_pose=candidate
                    print(f"ParaPred retreat fallback: using {retreat_scale:.2f} scale",flush=True)
                    break
        lq_takeover,ok_ltakeover=solve("left_arm",left_takeover_pose,left)
        lq_pre,ok_lpre=solve("left_arm",left_pre_pose,lq_takeover)
        lq_move,ok_lmove=solve("left_arm",left_move_pose,lq_takeover)
        lq_place,ok_lplace=solve("left_arm",left_place_pose,lq_move)
        feasibility=(ok_rpre,ok_rgrasp,ok_runblock,ok_rregrasp,ok_rregrasp2,ok_rregrasp3,ok_rregrasp4,ok_rup,ok_rhandover,ok_rback,ok_ltakeover,ok_lpre,ok_lmove,ok_lplace)
        print(f"ParaPred template IK feasibility={feasibility}",flush=True)
        # Exactly 300 actions.  After takeover, the right arm releases and
        # retreats completely before any left-arm transport begins.
        move("right_arm",rq_pre,15)
        move("right_arm",rq_grasp,10); grip("right_arm",0,15,0,smooth=False)
        move("right_arm",rq_unblock,5); grip("right_arm",0,15,0,smooth=False)
        move("right_arm",rq_regrasp,5); grip("right_arm",0,15,0,smooth=False)
        move("right_arm",rq_regrasp2,5); grip("right_arm",0,15,0,smooth=False)
        move("right_arm",rq_regrasp3,5); grip("right_arm",0,15,0,smooth=False)
        move("right_arm",rq_regrasp4,5); grip("right_arm",0,15,0,smooth=False)
        move("right_arm",rq_up,15)
        move("right_arm",rq_handover,15)
        move("left_arm",lq_pre,10)
        move("left_arm",lq_takeover,10)
        grip("left_arm",0,15,5)
        grip("right_arm",0.05,10,5)
        move("right_arm",rq_back,10)
        # Do not solve IK independently at every Cartesian sample: that can
        # switch IK branches and fling a held object.  Use the expert TOPPRA
        # generator between the already validated high-Z endpoint solutions.
        move("left_arm",lq_move,20)
        move("left_arm",lq_place,25)
        grip("left_arm",0.05,15,5)
        hold(0)
        self._event_lengths=(15,10,15,5,15,5,15,5,15,5,15,5,15,15,15,10,10,20,15,10,20,25,20,0)
        assert len(self._action_queue)==300, len(self._action_queue)
        self._last_action=action.copy()
        print(f"ParaPred fixed expert-pose program: {len(self._action_queue)} actions; segments={self._event_lengths}",flush=True)
        return np.asarray(self._action_queue,dtype=np.float32)

    
    def plan_expert_trajectory(self,env,pred):
        """Backward-compatible alias; no expert trajectory is read or reused."""
        return self.plan_trajectory(env,pred)

    @staticmethod
    def split_actions_into_events(actions):
        """Split fixed task into right-grasp, handover, and left-release segments."""
        actions=np.asarray(actions)
        left_eef,right_eef=actions[:,6],actions[:,13]
        right_close=np.flatnonzero((right_eef[:-1]>=0.045) & (right_eef[1:]<=0.005))+1
        if not len(right_close): return [actions]
        first=int(right_close[0])+1
        left_close=np.flatnonzero((np.arange(1,len(actions))>=first) & (left_eef[:-1]>=0.045) & (left_eef[1:]<=0.005))+1
        handover_start=int(left_close[0]) if len(left_close) else first
        right_open=np.flatnonzero((np.arange(1,len(actions))>=handover_start) & (right_eef[:-1]<=0.005) & (right_eef[1:]>=0.045))+1
        second=int(right_open[0])+1 if len(right_open) else first
        cuts=sorted(set((first,second)))
        return [part for part in np.split(actions,cuts) if len(part)]

    def has_pending_actions(self):
        return bool(self._action_queue or self._event_queues)

    def get_action(self,count=None):
        out=[]
        count=self.action_chunk if count is None else count
        if not self._action_queue and self._event_queues:
            self._action_queue.extend(self._event_queues.popleft())
        while self._action_queue and len(out)<count:
            self._last_action=np.asarray(self._action_queue.popleft(),dtype=np.float32).copy()
            out.append(self._last_action)
        return np.asarray(out,dtype=np.float32)

    def load_checkpoint(self,path):
        x=torch.load(path,map_location=self.device)
        self.load_state_dict(x.get("model",x))
