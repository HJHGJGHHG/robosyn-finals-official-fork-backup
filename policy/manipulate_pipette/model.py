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


def pipette_grasp_from_targets(targets):
    """Analytically derive expert grasp geometry from a random pipette pose."""
    base=targets["pipette_base_pose"]
    grasp=base.clone()
    rx=base.new_tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    rotation=base[:,:3,:3]@rx
    grasp[:,:3,:3]=rotation
    grasp[:,:3,3]=base[:,:3,3]+rotation[:,:,2]*0.02
    return grasp


def independent_parameters_from_targets(targets):
    grasp_cm=pipette_grasp_from_targets(targets)[:,:3,3]*WORLD_TO_CM
    beaker_cm=targets["beaker1_pose"][:,:3,3]*WORLD_TO_CM
    return torch.stack((grasp_cm,beaker_cm),dim=1)

def orientation_sincos_from_targets(targets,canonical_rotation):
    """World-Z yaw of the analytically derived grasp frame."""
    rotation=pipette_grasp_from_targets(targets)[:,:3,:3]
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
        pipette=torch.tensor([[.86769676,-.48192343,-.12186938,.60382640],[.48554257,.87421300,0.,-.24019773],[.10653979,-.05917277,.99254614,.85627630],[0.,0.,0.,1.]])
        pipette_grasp=torch.tensor([[.86769676,.48192343,.12186938,.60626382],[.48554257,-.87421298,0.,-.24019773],[.10653979,.05917277,-.99254614,.83642536],[0.,0.,0.,1.]])
        beaker=torch.tensor([[.9987032,-.04930404,.01269084,.5254067],[.04937076,.99876803,-.00499874,.09027869],[-.01242875,.00561882,.99990696,.824901],[0.,0.,0.,1.]])
        self.register_buffer("left_base_pose",left)
        self.register_buffer("right_base_pose",right)
        self.register_buffer("canonical_pipette_pose",pipette)
        self.register_buffer("canonical_pipette_grasp_pose",pipette_grasp)
        self.register_buffer("canonical_beaker_pose",beaker)
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
        pipette=self.canonical_pipette_pose.expand(b,-1,-1).clone()
        pipette_grasp=self.canonical_pipette_grasp_pose.expand(b,-1,-1).clone()
        beaker=self.canonical_beaker_pose.expand(b,-1,-1).clone()
        zero=torch.zeros_like(theta); one=torch.ones_like(theta)
        rotz=torch.stack((
            torch.cos(theta),-torch.sin(theta),zero,
            torch.sin(theta), torch.cos(theta),zero,
            zero,zero,one),dim=-1).view(b,3,3)
        pipette[:,:3,:3]=rotz@self.canonical_pipette_pose[:3,:3]
        pipette_grasp[:,:3,:3]=rotz@self.canonical_pipette_grasp_pose[:3,:3]
        pipette[:,:3,3]=points[:,0]+(self.canonical_pipette_pose[:3,3]-self.canonical_pipette_grasp_pose[:3,3])
        pipette_grasp[:,:3,3]=points[:,0]
        beaker[:,:3,3]=points[:,1]
        return {
            "left_arm_init_qpos":state[:,:6],
            "right_arm_init_qpos":state[:,7:13],
            "pipette_base_pose":pipette,
            "right_arm_pipette_grasp_pose":pipette_grasp,
            "beaker1_pose":beaker,
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
        action=np.concatenate((left,[0.0],right,[0.05])).astype(np.float32)
        right_grasp_pose=np.asarray(pred["right_arm_pipette_grasp_pose"][0].detach().cpu(),dtype=np.float32)
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
        # Fixed 300-frame task template. Only model-predicted interaction
        # points and pipette yaw affect execution; no expert trajectory is read.
        grasp=points[0].astype(np.float32).copy()
        beaker=points[1].astype(np.float32).copy()
        print(f"ParaPred pipette points (m): grasp={grasp.tolist()} beaker={beaker.tolist()}",flush=True)

        right_grasp_pose[:3,3]=grasp
        # Expert geometry: approach along local -Z, lift in world Z, then
        # approach the beaker vertically while keeping the pipette upright.
        right_pre_pose=get_offset_pose(right_grasp_pose.copy(),-.10,direction="z",mode="intrinsic")
        right_pre_pose=get_rotated_pose(right_pre_pose,20,rot_axis="y",mode="intrinsic",degrees=True)
        right_up_pose=get_offset_pose(right_grasp_pose.copy(),.15,direction="z",mode="extrinsic")
        right_up_pose=get_rotated_pose(right_up_pose,30,rot_axis="y",mode="intrinsic",degrees=True)
        aim=float(np.arctan2(beaker[1]-float(self.right_base_pose[1,3]),beaker[0]-float(self.right_base_pose[0,3])))
        def beaker_pose(height):
            pose=get_replaced_pose(right_up_pose.copy(),beaker,axis_str_replace="xyz")
            pose=get_rotation_replaced_pose(pose,aim,rot_axis="z",mode="intrinsic")
            pose=get_rotated_pose(pose,-90,rot_axis="y",mode="intrinsic",degrees=True)
            pose=get_rotated_pose(pose,180,rot_axis="x",mode="intrinsic",degrees=True)
            pose=get_offset_pose(pose,height,direction="z",mode="extrinsic")
            pose=get_offset_pose(pose,.02,direction="z",mode="intrinsic")
            return pose
        right_beaker_pre_pose=beaker_pose(.35)
        right_beaker_pose=beaker_pose(.20)
        rq_pre,ok_rpre=solve("right_arm",right_pre_pose,right)
        rq_grasp,ok_rgrasp=solve("right_arm",right_grasp_pose,rq_pre)
        if not ok_rgrasp:
            # Preserve predicted Z. Search nearby XY and wrist yaw only, ordered
            # by deviation from the model prediction, and execute the first
            # reachable candidate instead of silently holding the old qpos.
            original_grasp_pose=right_grasp_pose.copy()
            inward=np.asarray(self.right_base_pose[:2,3].detach().cpu(),dtype=np.float32)-original_grasp_pose[:2,3]
            inward=inward/max(float(np.linalg.norm(inward)),1e-6)
            tangent=np.asarray([-inward[1],inward[0]],dtype=np.float32)
            candidate_specs=[]
            for radial in (0.0,.005,.010,.015):
                for lateral in (0.0,.005,-.005,.010,-.010):
                    if radial==0.0 and lateral==0.0:
                        continue
                    for yaw_deg in (0.0,5.0,-5.0,10.0,-10.0):
                        cost=float(np.hypot(radial,lateral))+abs(yaw_deg)*1e-4
                        candidate_specs.append((cost,radial,lateral,yaw_deg))
            for _,radial,lateral,yaw_deg in sorted(candidate_specs):
                candidate=original_grasp_pose.copy()
                candidate[:2,3]+=radial*inward+lateral*tangent
                if yaw_deg:
                    candidate=get_rotated_pose(candidate,yaw_deg,rot_axis="z",mode="intrinsic",degrees=True)
                rq_candidate,ok_candidate=solve("right_arm",candidate,rq_pre)
                if ok_candidate:
                    rq_grasp,ok_rgrasp=rq_candidate,True
                    right_grasp_pose=candidate
                    print(f"ParaPred grasp IK XY/yaw fallback: inward_m={radial:.3f} lateral_m={lateral:.3f} yaw_deg={yaw_deg:.1f}",flush=True)
                    break
            if not ok_rgrasp:
                # The reachable pre-grasp pose differs mainly in pitch/roll.
                # Relax only orientation toward that verified pose; XYZ stays at
                # the model prediction. Project each blend back onto SO(3).
                source_rotation=original_grasp_pose[:3,:3]
                safe_rotation=right_pre_pose[:3,:3]
                for relax in (.25,.50,.75,1.0):
                    candidate=original_grasp_pose.copy()
                    blended=(1.0-relax)*source_rotation+relax*safe_rotation
                    u,_,vt=np.linalg.svd(blended)
                    rotation=u@vt
                    if np.linalg.det(rotation)<0:
                        u[:,-1]*=-1
                        rotation=u@vt
                    candidate[:3,:3]=rotation
                    rq_candidate,ok_candidate=solve("right_arm",candidate,rq_pre)
                    if ok_candidate:
                        rq_grasp,ok_rgrasp=rq_candidate,True
                        right_grasp_pose=candidate
                        print(f"ParaPred grasp IK orientation fallback: relax={relax:.2f}",flush=True)
                        break
            if not ok_rgrasp:
                print("ParaPred grasp IK XY/yaw/orientation fallback exhausted without solution",flush=True)
        rq_up,ok_rup=solve("right_arm",right_up_pose,rq_grasp)
        rq_beaker_pre,ok_rbpre=solve("right_arm",right_beaker_pre_pose,rq_up)
        rq_beaker,ok_rb=solve("right_arm",right_beaker_pose,rq_beaker_pre)
        # Anchor every press target to the FK pose at the right arm final hold.
        right_final_pose=get_fk_xpos(rq_beaker,"right_arm",base.robot.compute_fk).astype(np.float32)
        # Left EEF reaches the pipette button only after the right arm has
        # settled above the beaker. Press and retreat are vertical/safe.
        left_ready_pose=get_offset_pose(right_final_pose.copy(),-.014,direction="z",mode="intrinsic")
        left_ready_pose=get_rotated_pose(left_ready_pose,-90,rot_axis="x",mode="intrinsic",degrees=True)
        left_ready_pose=get_offset_pose(left_ready_pose,.12,direction="z",mode="extrinsic")
        left_ready_pose=get_rotated_pose(left_ready_pose,-20,rot_axis="y",mode="intrinsic",degrees=True)
        # Use 10 mm of additional overtravel so contact variation still drives
        # the pipette plunger past the official slide-joint success threshold.
        left_press_pose=get_offset_pose(left_ready_pose.copy(),-.042,direction="z",mode="extrinsic")
        # One-dimensional search along the final held-pipette pose projected
        # onto world XY: center, then +/-7/14/21/28 mm.
        press_search_poses=[]
        search_direction=np.asarray(right_final_pose[:2,2],dtype=np.float32)
        search_direction/=max(float(np.linalg.norm(search_direction)),1e-6)
        for offset in (.007,-.007,.014,-.014,.021,-.021,.028,-.028):
            dx=float(search_direction[0])*offset
            dy=float(search_direction[1])*offset
            ready_pose=get_offset_pose(left_ready_pose.copy(),dx,direction="x",mode="extrinsic")
            ready_pose=get_offset_pose(ready_pose,dy,direction="y",mode="extrinsic")
            press_pose=get_offset_pose(ready_pose.copy(),-.042,direction="z",mode="extrinsic")
            press_search_poses.append((ready_pose,press_pose))
        left_retreat_pose=get_offset_pose(left_ready_pose.copy(),.05,direction="z",mode="extrinsic")
        left_retreat_pose=get_offset_pose(left_retreat_pose,.05,direction="x",mode="extrinsic")
        left_retreat_pose=get_offset_pose(left_retreat_pose,.08,direction="y",mode="extrinsic")

        lq_ready,ok_lready=solve("left_arm",left_ready_pose,left)
        lq_press,ok_lpress=solve("left_arm",left_press_pose,lq_ready)
        press_search_q=[]
        press_search_ok=[]
        for ready_pose,press_pose in press_search_poses:
            ready_q,ok_ready=solve("left_arm",ready_pose,lq_ready)
            press_q,ok_press=solve("left_arm",press_pose,ready_q)
            press_search_q.append((ready_q,press_q))
            press_search_ok.extend((ok_ready,ok_press))
        lq_retreat,ok_lretreat=solve("left_arm",left_retreat_pose,press_search_q[-1][0])
        feasibility=(ok_rpre,ok_rgrasp,ok_rup,ok_rbpre,ok_rb,ok_lready,ok_lpress,*press_search_ok,ok_lretreat)
        print(f"ParaPred manipulate-pipette IK feasibility={feasibility}",flush=True)

        # Exactly 500 actions: reach, grasp-and-settle, lift, move above beaker,
        # left-button approach, press-and-hold, then collision-free retreat.
        move("right_arm",rq_pre,30)
        move("right_arm",rq_grasp,25)
        grip("right_arm",0.0,20,5,smooth=False)
        move("right_arm",rq_up,20)
        move("right_arm",rq_beaker_pre,20)
        move("right_arm",rq_beaker,20)
        move("left_arm",lq_ready,30)
        # Center first, then +/-7/14/21/28 mm along the final held-pipette direction.
        move("left_arm",lq_press,15)
        hold(5)
        move("left_arm",lq_ready,10)
        for ready_q,press_q in press_search_q:
            move("left_arm",ready_q,5)
            move("left_arm",press_q,15)
            hold(5)
            move("left_arm",ready_q,10)
        move("left_arm",lq_retreat,20)
        self._event_lengths=(30,25,25,20,20,20,30,30,*([35]*8),20)
        assert len(self._action_queue)==500,len(self._action_queue)
        self._last_action=action.copy()
        print(f"ParaPred fixed manipulate-pipette program: {len(self._action_queue)} actions; segments={self._event_lengths}",flush=True)
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
        x=torch.load(path,map_location=self.device,weights_only=False)
        state=x.get("model",x)
        incompatible=self.load_state_dict(state,strict=False)
        allowed={"canonical_pipette_pose","canonical_pipette_grasp_pose","canonical_beaker_pose"}
        missing=set(incompatible.missing_keys)-allowed
        unexpected=set(incompatible.unexpected_keys)-{"canonical_pen_pose","canonical_pen_grasp_pose","canonical_holder_pose"}
        if missing or unexpected: raise RuntimeError(f"incompatible checkpoint: missing={sorted(missing)} unexpected={sorted(unexpected)}")
