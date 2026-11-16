"""Append a continuous reverse-slide close sequence to the unchanged P3 prefix."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import DRAWER_ROTATION
from .motion import PlanningError,branch_path,smooth_joint


def append_close_tail(result,parameters,kin,max_steps=1000,epsilon=.01,*,skip_park=False,timing='baseline'):
    if timing not in ('baseline','compact'):raise ValueError('Unknown closing timing')
    if not .005<=epsilon<=.02:raise ValueError('Close clearance must be5--20mm')
    current=result['actions'][-1].copy();chunks=[result['actions']];stages=list(result['stages'])
    g=result['targets'];direction=DRAWER_ROTATION[:,0]
    handle=np.asarray(parameters)[3:6];drawer=np.asarray(parameters)[6:9]
    if result.get('p6_initial_grasp'):
        handle=g['right_contact'][:3,3]
    park=g['left_withdraw'].copy()
    if not skip_park:
        park[:3,3]+=[0,.05,.02]
        park[2,3]=max(park[2,3],drawer[2]+.18)
    coarse=g['right_pull'].copy();coarse[:3,3]=handle+.04*direction
    close=coarse.copy();close[:3,3]=handle+epsilon*direction
    retreat=close.copy();retreat[:3,3]+=.08*direction
    margin=result['min_margin_rad'];jump=result['max_contact_jump_rad']

    def append(name,n,arm=None,target=None,grip=None):
        nonlocal current,margin,jump
        if timing=='compact':n={'right_push_coarse':65,'right_retreat':30}.get(name,n)
        sl=slice(0,6) if arm=='left_arm' else slice(7,13)
        if arm:
            start=kin[arm].fk(current[sl][None])[0]
            count=max(2,int(np.ceil(np.linalg.norm(target[:3,3]-start[:3,3])/.002)))
            poses=np.repeat(target[None],count+1,axis=0)
            poses[:,:3,3]=np.linspace(start[:3,3],target[:3,3],count+1)
            path,m=branch_path(kin[arm],poses,current[sl],max_jump=np.deg2rad(10))
            if np.max(abs(path[0]-current[sl]))>.01:raise PlanningError(name+':branch_join')
            margin=min(margin,m);jump=max(jump,float(abs(np.diff(path,axis=0)).max()))
        for _ in range(6):
            a=np.tile(current,(n,1))
            if arm:
                u=np.arange(1,n+1)/n;index=(10*u**3-15*u**4+6*u**5)*(len(path)-1)
                a[:,sl]=np.column_stack([np.interp(index,np.arange(len(path)),path[:,j]) for j in range(6)])
                expected=np.column_stack([np.interp(index,np.arange(len(path)),poses[:,j,3]) for j in range(3)])
                actual=kin[arm].fk(a[:,sl])
                if np.linalg.norm(actual[:,:3,3]-expected,axis=1).max()>.0005:raise PlanningError(name+':cartesian_residual')
                if Rotation.from_matrix(actual[:,:3,:3]@target[:3,:3].T).magnitude().max()>np.deg2rad(.25):raise PlanningError(name+':angular_residual')
                v=np.diff(np.vstack([current[sl],a[:,sl],a[-1,sl]]),axis=0)/.04
                acc=np.diff(np.vstack([np.zeros((1,6)),v]),axis=0)/.04
                ratio=max(float((abs(v)/kin[arm].velocity).max()),float(np.sqrt(abs(acc).max()/20)))
                if ratio>1.0001:n=int(np.ceil(n*ratio*1.02));continue
            if grip:
                for col,value in grip.items():a[:,col]=smooth_joint(current[col:col+1],np.array([value]),n)[:,0]
            break
        else:raise PlanningError(name+':retiming_failed')
        offset=sum(len(x) for x in chunks)
        if offset+n>max_steps:raise PlanningError(name+':step_budget')
        chunks.append(a);stages.append(dict(name=name,start=offset,steps=n));current=a[-1].copy()

    if not skip_park:append('left_clearance_park',30,'left_arm',park)
    append('right_push_coarse',75,'right_arm',coarse)
    append('right_push_fine',35,'right_arm',close)
    append('right_close_settle',10)
    append('right_release_handle',15,grip={13:.05})
    # Allow the jaws to release the handle before retreat can pull it open again.
    append('right_release_settle',20)
    append('right_retreat',35,'right_arm',retreat)
    append('closed_final_hold',30)
    return dict(result,actions=np.concatenate(chunks),stages=stages,min_margin_rad=margin,max_contact_jump_rad=jump,close_start_step=len(result['actions'])+1,close_targets=dict(park=park.tolist(),coarse=coarse.tolist(),close=close.tolist(),retreat=retreat.tolist()),close_epsilon_m=epsilon)
