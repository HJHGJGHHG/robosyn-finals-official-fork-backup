"""Bounded 14D planner with explicit kinematics adapter and time/limit checks.

Does not access object poses from a simulator. Collision qualification is supplied
separately; a plan without it remains diagnostic-only.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .geometry import candidate_geometry, interpolate_pose

class PlanningError(RuntimeError): pass

@dataclass
class Kinematics:
    solver: object
    base: np.ndarray
    limits: np.ndarray
    velocity: np.ndarray

    def ik(self, poses):
        poses=np.asarray(poses)
        cache=getattr(self,'_ik_cache',None)
        if cache is None:
            local=np.linalg.inv(self.base) @ poses
            mask,q=self.solver.get_ik(torch.tensor(local,dtype=torch.float32),return_all_solutions=True)
            return mask.cpu().numpy()>0,q.cpu().numpy()
        keys=[pose.tobytes() for pose in poses]
        missing={key:pose for key,pose in zip(keys,poses) if key not in cache}
        if missing:
            local=np.linalg.inv(self.base) @ np.asarray(list(missing.values()))
            mask,q=self.solver.get_ik(torch.tensor(local,dtype=torch.float32),return_all_solutions=True)
            for key,m,v in zip(missing,mask.cpu().numpy()>0,q.cpu().numpy()):cache[key]=(m.copy(),v.copy())
        # branch_path filters validity in place; cached solver masks must remain untouched.
        return np.array([cache[k][0] for k in keys]),np.array([cache[k][1] for k in keys])

    def fk(self,q):
        # CPU eager FK avoids repeated compilation and validates against the URDF chain.
        t=self.solver.get_fk(torch.tensor(np.asarray(q),dtype=torch.float32)).detach().cpu().numpy()
        return self.base @ t


def branch_path(kin, poses, seed, margin_deg=2, max_jump=None):
    valid,q=kin.ik(poses)
    margins=np.minimum(q-kin.limits[:,0],kin.limits[:,1]-q).min(-1)
    valid &= margins>=np.deg2rad(margin_deg)
    if not valid.any(1).all(): raise PlanningError('ik_or_margin')
    costs=np.where(valid[0],((q[0]-seed)**2).sum(-1),np.inf)
    parents=[]
    for i in range(1,len(q)):
        diff=q[i,None,:,:]-q[i-1,:,None,:]
        edge=(diff**2).sum(-1)
        if max_jump is not None: edge[np.max(abs(diff),axis=-1)>max_jump]=np.inf
        score=costs[:,None]+edge; parent=score.argmin(0)
        costs=np.where(valid[i],score[parent,np.arange(8)],np.inf);parents.append(parent)
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


def plan_candidate(tube,rack,kin,qinit,geom,shift,yaw,delta,hyaw,retreat_mode='local',geometry_override=None,start_stage=None,max_steps=500,end_stage=None,settle_steps=24,transport_steps=40):
    g=candidate_geometry(tube,rack,kin['right_arm'].base,geom,shift,yaw,delta,hyaw,retreat_mode) if geometry_override is None else geometry_override
    # Handover template first presses 5 mm below the predicted contact in world Z.
    g['right']['press']=g['right']['grasp'].copy();g['right']['press'][2,3]-=.005
    if start_stage not in (None,'left_to_pregrasp','left_transport','align_above_hole','insert'):raise ValueError('Unsupported suffix start')
    if end_stage not in (None,'left_proof') or (end_stage=='left_proof' and start_stage!='left_to_pregrasp'):raise ValueError('Unsupported bounded handover')
    right_keys=['pre','grasp','press','proof','up','hand','retreat'] if start_stage is None else (['retreat'] if start_stage=='left_to_pregrasp' else [])
    left_keys={None:['left_pre','left','left_proof','above','near','place','withdraw'],
               'left_to_pregrasp':['left_pre','left','left_proof','above','near','place','withdraw'],
               'left_transport':['above','near','place','withdraw'],'align_above_hole':['near','place','withdraw'],
               'insert':['place','withdraw']}[start_stage]
    if end_stage=='left_proof':left_keys=left_keys[:3]
    qr={}
    if right_keys:
        right=np.array([g['right_retreat'] if k=='retreat' else g['right'][k] for k in right_keys])
        try: values,_=branch_path(kin['right_arm'],right,qinit[7:13]);qr=dict(zip(right_keys,values))
        except PlanningError as exc:
            valid,q=kin['right_arm'].ik(right)
            margin=np.minimum(q-kin['right_arm'].limits[:,0],kin['right_arm'].limits[:,1]-q).min(-1)
            valid &= margin>=np.deg2rad(2)
            failed=[key for key,ok in zip(right_keys,valid.any(1)) if not ok]
            raise PlanningError('right_keypoints:'+str(exc)+':'+','.join(failed)) from exc
    left=np.array([g[k] for k in left_keys])
    try: values,_=branch_path(kin['left_arm'],left,qinit[:6]);ql=dict(zip(left_keys,values))
    except PlanningError as exc:
        valid,q=kin['left_arm'].ik(left)
        valid &= np.minimum(q-kin['left_arm'].limits[:,0],kin['left_arm'].limits[:,1]-q).min(-1)>=np.deg2rad(2)
        failed=[k for k,v in zip(left_keys,valid.any(1)) if not v]
        raise PlanningError('left_keypoints:'+str(exc)+':'+','.join(failed)) from exc
    if start_stage=='align_above_hole':ql['above']=qinit[:6].copy()
    if start_stage=='insert':ql['near']=qinit[:6].copy()
    # Dense Cartesian segments: sequential branch graph and <=2 mm knots.
    cart={}
    segments=[] if end_stage=='left_proof' else [('insert','left_arm',g['near'],g['place'],ql['near']),
              ('withdraw','left_arm',g['place'],g['withdraw'],ql['place'])]
    if end_stage is None and start_stage!='insert':
        segments.insert(0,('align_above_hole','left_arm',g['above'],g['near'],ql['above']))
        if transport_steps==120 and start_stage!='align_above_hole':
            # Cartesian transport avoids the wrist swing of joint interpolation.
            transport_start=kin['left_arm'].fk(qinit[None,:6])[0] if start_stage=='left_transport' else g['left_proof']
            transport_seed=qinit[:6] if start_stage=='left_transport' else ql['left_proof']
            segments.insert(0,('left_transport','left_arm',transport_start,g['above'],transport_seed))
    if start_stage in (None,'left_to_pregrasp'):segments.insert(0,('left_approach','left_arm',g['left_pre'],g['left'],ql['left_pre']))
    if start_stage is None:segments[:0]=[
        ('right_descend','right_arm',g['right']['pre'],g['right']['grasp'],qr['pre']),
        ('right_press_down','right_arm',g['right']['grasp'],g['right']['press'],qr['grasp']),
        ('right_proof_lift','right_arm',g['right']['press'],g['right']['proof'],qr['press'])]
    for name,arm,a,b,seed in segments:
        n=max(3,int(np.ceil(np.linalg.norm(b[:3,3]-a[:3,3])/.002))+1)
        poses=interpolate_pose(a,b,n)
        try: path,_=branch_path(kin[arm],poses,seed,max_jump=np.deg2rad(10))
        except PlanningError as exc: raise PlanningError(name+':'+str(exc)) from exc
        # Initial branch must join the already selected waypoint.
        if np.max(abs(path[0]-seed))>.01: raise PlanningError(name+':branch_join')
        cart[name]=(path,poses,arm)
    # Budgeted insertion search around the predicted rack hole.
    search_sequence=[]
    if end_stage is None and start_stage not in ('insert',):
        axis=np.asarray(rack)[:3,2];plane=np.asarray(rack)[:3,:2]
        center=g['near'].copy();current_pose=center.copy();current_q=ql['near'].copy()
        depth_limit=float((g['near'][:3,3]-g['place'][:3,3])@axis)
        search_radius=.005;nominal_clearance=.0009251335802494057
        golden=np.pi*(3-np.sqrt(5))
        points=[(0.,0.)]+[(search_radius*np.sqrt(i/9)*np.cos(i*golden),
                           search_radius*np.sqrt(i/9)*np.sin(i*golden)) for i in range(1,10)]
        depth=0.
        def search_cart(name,target,minimum_steps,spacing=.001):
            nonlocal current_pose,current_q
            distance=np.linalg.norm(target[:3,3]-current_pose[:3,3])
            poses=interpolate_pose(current_pose,target,max(3,int(np.ceil(distance/spacing))+1))
            path,_=branch_path(kin['left_arm'],poses,current_q,max_jump=np.deg2rad(10))
            if np.max(abs(path[0]-current_q))>.01:raise PlanningError(name+':branch_join')
            cart[name]=(path,poses,'left_arm');current_pose=target.copy();current_q=path[-1].copy()
            search_sequence.append((name,minimum_steps,current_q.copy()))
        for i,(x,y) in enumerate(points):
            shift=center.copy();shift[:3,3]=center[:3,3]+plane@np.array([x,y])-axis*depth
            search_cart(f'search_shift_{i:02d}',shift,4)
            down=min(depth+.005,depth_limit)
            probe=shift.copy();probe[:3,3]=center[:3,3]+plane@np.array([x,y])-axis*down
            search_cart(f'search_insert_{i:02d}',probe,6)
            depth=max(0.,down-.001)
            lift=probe.copy();lift[:3,3]+=axis*.001
            search_cart(f'search_lift_{i:02d}',lift,3)
        # Secondary reverse pass. Once the tip is through both rack apertures,
        # alternate between two absolute depths so the lower aperture keeps
        # constraining the tube while points are revisited in reverse.
        secondary_depth=.035-(-.026133400627454624)
        secondary_table_height=.01321
        reverse_insert=.011
        reverse_down=min(secondary_depth+reverse_insert,depth_limit)
        effective_tip_length=.127
        secondary_margin=.002
        minimum_recovery_lift=.001
        secondary_penetration=reverse_down-secondary_depth
        adaptive_reverse=[]

        x,y=points[-1]
        secondary=center.copy()
        secondary[:3,3]=center[:3,3]+plane@np.array([x,y])-axis*reverse_down
        search_cart('secondary_enter_09',secondary,8,spacing=.002)
        for i in range(len(points)-2,-1,-1):
            x,y=points[i]
            previous=np.asarray(points[i+1])
            current_xy=np.asarray(points[i])
            lateral=float(np.linalg.norm(current_xy-previous))
            predicted_tilt=float(np.arctan2(lateral,effective_tip_length))
            recovery=float(effective_tip_length*(1-np.cos(predicted_tilt))+.001)
            safe_max=float(secondary_penetration*np.cos(predicted_tilt)-secondary_margin)
            lift_amount=float(np.clip(recovery,minimum_recovery_lift,safe_max))
            reverse_up=reverse_down-lift_amount
            adaptive_reverse.append(dict(index=i,lateral_m=lateral,
                predicted_tilt_deg=float(np.rad2deg(predicted_tilt)),
                recovery_lift_m=recovery,safe_max_lift_m=safe_max,
                applied_lift_m=lift_amount,up_depth_m=reverse_up))
            up=current_pose.copy();up[:3,3]+=axis*lift_amount
            search_cart(f'secondary_lift_{i:02d}',up,3,spacing=.002)
            shift=center.copy()
            shift[:3,3]=center[:3,3]+plane@np.array([x,y])-axis*reverse_up
            search_cart(f'secondary_shift_{i:02d}',shift,3,spacing=.002)
            down=shift.copy();down[:3,3]-=axis*lift_amount
            search_cart(f'secondary_insert_{i:02d}',down,3,spacing=.002)
        g['search_tail']=dict(radius_m=search_radius,points_xy_m=points,
            nominal_clearance_m=nominal_clearance,attempts=len(points),
            insert_each_m=.005,lift_each_m=.001,coverage='golden_angle_adaptive_secondary_reverse',
            secondary_depth_m=secondary_depth,secondary_table_height_m=secondary_table_height,
            reverse_insert_m=reverse_insert,reverse_down_depth_m=reverse_down,
            effective_tip_length_m=effective_tip_length,secondary_margin_m=secondary_margin,
            adaptive_reverse=adaptive_reverse,
            reverse_points=list(range(len(points)-2,-1,-1)))
    stages=[];current=qinit.copy();dt=.04
    def append(name,n,arm=None,end=None,grip=None,_retime=0):
        nonlocal current
        if arm:
            sl=slice(0,6) if arm=='left_arm' else slice(7,13)
            # Quintic blend peak derivative is 1.875. Allocate real time from
            # effective velocity bounds; never silently compress the path.
            if name not in cart:
                required=int(np.ceil(np.max(1.875*abs(end-current[sl])/kin[arm].velocity)/dt))
                n=max(n,required)
        actions=np.tile(current,(n,1))
        if arm:
            sl=slice(0,6) if arm=='left_arm' else slice(7,13)
            if name in cart:
                path,poses,_=cart[name]
                if name=='insert':
                    # Nonuniform timing, slower within the last 8 mm.
                    distance=np.linalg.norm(poses[1:,:3,3]-poses[:-1,:3,3],axis=1)
                    remaining=np.linalg.norm(poses[:-1,:3,3]-poses[-1,:3,3],axis=1)
                    speeds=np.where(remaining<=.008,.010,.035)
                    times=np.r_[0,np.cumsum(distance/speeds)]
                    if times[-1]>n*dt: raise PlanningError('insertion_time_budget')
                    times=times/times[-1]*(n*dt)
                    sample=np.arange(1,n+1)*dt
                    segment=np.interp(sample,times,np.arange(len(path)))
                else:
                    u=np.arange(1,n+1)/n;s=10*u**3-15*u**4+6*u**5
                    segment=s*(len(path)-1)
                actions[:,sl]=np.column_stack([np.interp(segment,np.arange(len(path)),path[:,j]) for j in range(6)])
                # Verify interpolated joints against Cartesian curve, not just knots.
                fk=kin[arm].fk(actions[:,sl]);target=np.column_stack([np.interp(segment,np.arange(len(poses)),poses[:,j,3]) for j in range(3)])
                if np.linalg.norm(fk[:,:3,3]-target,axis=1).max()>.0005: raise PlanningError(name+':cartesian_path_error')
                if np.max(abs(actions[-1,sl]-end))>.01: raise PlanningError(name+':end_branch_join')
            else: actions[:,sl]=smooth_joint(current[sl],end,n)
        if grip:
            for col,value in grip.items():
                hold=6 if value==0 else 0
                ramp=smooth_joint(np.array([current[col]]),np.array([value]),n-hold)[:,0]
                actions[:,col]=np.r_[ramp,np.full(hold,value)]
        if arm:
            v=np.diff(np.vstack([current[sl],actions[:,sl],actions[-1,sl]]),axis=0)/dt
            acc=np.diff(np.vstack([np.zeros((1,6)),v]),axis=0)/dt
            ratio=max(float(np.max(abs(v)/kin[arm].velocity)),float(np.sqrt(np.max(abs(acc))/20.0)))
            if ratio>1.0001:
                if _retime>=4:raise PlanningError(name+':retiming_failed')
                return append(name,int(np.ceil(n*ratio*1.01)),arm,end,grip,_retime+1)
        if name=='left_transport' and transport_steps in (80,120) and n>transport_steps:
            raise PlanningError('left_transport:shared_time_budget')
        stages.append(dict(name=name,start=sum(len(v['actions']) for v in stages),actions=actions))
        current=actions[-1].copy()
    if start_stage is None:
        append('open_and_observe',12,grip={6:.05,13:.05})
        append('right_pregrasp',40,'right_arm',qr['pre']);append('right_descend',20,'right_arm',qr['grasp'])
        append('right_press_down',10,'right_arm',qr['press'])
        # Handover-proven staged grasp: close while stationary, then relieve
        # table constraint in two 3 mm increments and re-issue closure after
        # every increment. Intermediate endpoints come from the validated
        # Cartesian grasp-to-proof IK branch; no GT state is used.
        proof_path=cart.pop('right_proof_lift')[0]
        append('right_close',15,grip={13:0})
        proof_distance=float(np.linalg.norm(g['right']['proof'][:3,3]-g['right']['press'][:3,3]))
        for lift_m in (.003,.006):
            index=int(round((lift_m/proof_distance)*(len(proof_path)-1)))
            append('right_proof_lift',5,'right_arm',proof_path[index])
            append('right_close',15,grip={13:0})
        append('right_to_up',30,'right_arm',qr['up']);append('right_to_handover',30,'right_arm',qr['hand'])
    if start_stage in (None,'left_to_pregrasp'):
        append('left_to_pregrasp',32,'left_arm',ql['left_pre']);append('left_approach',18,'left_arm',ql['left'])
        append('left_close',20,grip={6:0});append('right_release',12,grip={13:.05})
        append('right_retreat',18,'right_arm',qr['retreat']);append('left_proof',8,'left_arm',ql['left_proof'])
    if end_stage is None:
        if start_stage not in ('align_above_hole','insert'):append('left_transport',transport_steps,'left_arm',ql['above'])
        if start_stage!='insert':append('align_above_hole',20,'left_arm',ql['near'])
        if search_sequence:
            for name,minimum_steps,target_q in search_sequence:
                append(name,minimum_steps,'left_arm',target_q)
        else:
            append('insert',40,'left_arm',ql['place'])
        append('release',12,grip={6:.05})
        append('withdraw',20,'left_arm',ql['withdraw']);append('settle',min(settle_steps,12))
    actions=np.concatenate([s['actions'] for s in stages])
    if len(actions)>min(max_steps,700) or not np.isfinite(actions).all(): raise PlanningError('invalid_action_budget_700')
    audit={}
    for arm,sl in [('left_arm',slice(0,6)),('right_arm',slice(7,13))]:
        qs=np.vstack([qinit[sl],actions[:,sl]])
        if ((actions[:,sl]<kin[arm].limits[:,0]-1e-5)|(actions[:,sl]>kin[arm].limits[:,1]+1e-5)).any(): raise PlanningError('joint_limits')
        vel=np.diff(qs,axis=0)/dt;acc=np.diff(vel,axis=0)/dt
        vmax=abs(vel).max(0)
        if (vmax>kin[arm].velocity+1e-4).any(): raise PlanningError(arm+':velocity_budget')
        if abs(acc).max()>20.0001:raise PlanningError(arm+':acceleration_budget')
        audit[arm]={'peak_velocity_rad_s':vmax.tolist(),'peak_acceleration_rad_s2':abs(acc).max(0).tolist(),
                    'acceleration_limit_rad_s2':20.0,
                    'acceleration_qualification':'explicit engineering planning cap; not a manufacturer rating'}
    return dict(actions=actions,stages=[{k:v for k,v in s.items() if k!='actions'}|{'steps':len(s['actions'])} for s in stages],
                geometry=g,audit=audit,collision_qualified=False)


def _plan_grid(tube,rack,kin,qinit,geom,max_candidates=864,shift_only=None,max_steps=500,settle_steps=24,transport_steps=40,control_variant="C0"):
    if control_variant != "C0": raise ValueError("V4 control variant not implemented")
    failures=[];attempts=0
    shifts=[shift_only] if shift_only is not None else [.11,.12,.10]
    for delta in [(0,0,0),(-.03,0,0),(0,.03,0),(0,0,.03)]:
        for hyaw in [0,-10,10]:
            for shift in shifts:
                for yaw in [0,-10,10,-20,20,-90,90,180]:
                    for retreat in ['local','toward_base','world_negative_y']:
                        attempts+=1
                        try:
                            result=plan_candidate(tube,rack,kin,qinit,geom,shift,yaw,delta,hyaw,retreat,max_steps=max_steps,settle_steps=settle_steps,transport_steps=transport_steps)
                            result['candidate_attempts']=attempts;result['rejections']=failures
                            return result
                        except PlanningError as exc: failures.append({'shift':shift,'yaw':yaw,'delta':delta,'hand_yaw':hyaw,'retreat_mode':retreat,'reason':str(exc)})
                        if attempts>=max_candidates:break
                    if attempts>=max_candidates: break
                if attempts>=max_candidates: break
            if attempts>=max_candidates: break
        if attempts>=max_candidates: break
    error=PlanningError('no_complete_plan');error.rejections=failures;raise error


def plan(tube,rack,kin,qinit,geom,max_candidates=864,shift_only=None,max_steps=500,settle_steps=24,transport_steps=40,control_variant="C0"):
    """Preserve the original complete plan; try two wrist profiles only on failure."""
    import copy
    cached={name:copy.copy(k) for name,k in kin.items()}
    for k in cached.values():k._ik_cache={}
    options=dict(max_candidates=max_candidates,shift_only=shift_only,max_steps=max_steps,settle_steps=settle_steps,transport_steps=transport_steps,control_variant=control_variant)
    profiles=[{},dict(grasp_tilt_deg=0,proof_tilt_deg=5,up_tilt_deg=40),dict(grasp_tilt_deg=5,proof_tilt_deg=5,up_tilt_deg=40)]
    if not geom.get('right_repair_enabled',True):profiles=profiles[:1]
    failures=[]
    for index,profile in enumerate(profiles):
        try:
            result=_plan_grid(tube,rack,cached,qinit,dict(geom,right_profile=profile),**options)
            result['right_profile_index']=index
            result['prior_profile_rejections']=failures
            return result
        except PlanningError as exc:
            failures.append(dict(profile=index,error=str(exc),rejections=getattr(exc,'rejections',[])))
    error=PlanningError('no_complete_plan');error.rejections=failures;raise error
