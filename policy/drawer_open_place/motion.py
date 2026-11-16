"""Reused sample-loading IK branch selection, FK checks and quintic timing."""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation

class PlanningError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

@dataclass
class Kinematics:
    solver: object
    base: np.ndarray
    limits: np.ndarray
    velocity: np.ndarray

    def ik(self, poses):
        local=np.linalg.inv(self.base) @ np.asarray(poses)
        mask,q=self.solver.get_ik(torch.tensor(local,dtype=torch.float32),return_all_solutions=True)
        return mask.cpu().numpy()>0,q.cpu().numpy()

    def fk(self,q):
        # CPU eager FK avoids repeated compilation and validates against the URDF chain.
        t=self.solver.get_fk(torch.tensor(np.asarray(q),dtype=torch.float32)).detach().cpu().numpy()
        return self.base @ t


def branch_path(kin, poses, seed, margin_deg=2, max_jump=None):
    valid,q=kin.ik(poses)
    raw_counts=valid.sum(1)
    margins=np.minimum(q-kin.limits[:,0],kin.limits[:,1]-q).min(-1)
    valid &= margins>=np.deg2rad(margin_deg)
    if not valid.any(1).all():
        i=int(np.flatnonzero(~valid.any(1))[0])
        raise PlanningError('ik_or_margin',dict(waypoint=i,xyz=poses[i,:3,3].tolist(),raw_ik_count=int(raw_counts[i]),margin_valid_count=int(valid[i].sum())))
    costs=np.where(valid[0],((q[0]-seed)**2).sum(-1),np.inf)
    parents=[]
    for i in range(1,len(q)):
        diff=q[i,None,:,:]-q[i-1,:,None,:]
        edge=(diff**2).sum(-1)
        if max_jump is not None: edge[np.max(abs(diff),axis=-1)>max_jump]=np.inf
        score=costs[:,None]+edge; parent=score.argmin(0)
        new_costs=np.where(valid[i],score[parent,np.arange(q.shape[1])],np.inf)
        if not np.isfinite(new_costs).any():
            gaps=np.max(abs(diff),axis=-1)
            bridges=np.where(np.isfinite(costs)[:,None]&valid[i][None,:],gaps,np.inf)
            pair=np.unravel_index(np.argmin(bridges),bridges.shape)
            raise PlanningError('ik_branch_discontinuity',dict(waypoint=i,xyz=poses[i,:3,3].tolist(),raw_ik_count=int(raw_counts[i]),margin_valid_count=int(valid[i].sum()),minimum_bridge_deg=float(np.rad2deg(bridges[pair])),bridge_joint_deltas_deg=np.rad2deg(diff[pair]).tolist()))
        costs=new_costs;parents.append(parent)
    if not np.isfinite(costs).any(): raise PlanningError('ik_branch_discontinuity')
    idx=int(costs.argmin());selected=[q[-1,idx]]
    for i in range(len(q)-2,-1,-1):
        idx=int(parents[i][idx]);selected.append(q[i,idx])
    chosen=np.array(selected[::-1])
    actual=kin.fk(chosen)
    pos=np.linalg.norm(actual[:,:3,3]-poses[:,:3,3],axis=1)
    ang=Rotation.from_matrix(actual[:,:3,:3]@poses[:,:3,:3].transpose(0,2,1)).magnitude()
    if pos.max()>.0005 or ang.max()>np.deg2rad(.25): raise PlanningError('fk_residual')
    chosen_margin=np.minimum(chosen-kin.limits[:,0],kin.limits[:,1]-chosen).min()
    return chosen,float(chosen_margin)


def smooth_joint(start,end,n):
    u=np.arange(1,n+1)/n;s=10*u**3-15*u**4+6*u**5
    return start+s[:,None]*(end-start)


