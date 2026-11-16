"""Support-aware placement, measured retreat and explicit failure escape."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import DRAWER_ROTATION,pose
from .motion import PlanningError,branch_path,smooth_joint
from .refinement import Suffix,end
from .close_tail import append_close_tail
from .clearance import check as clear
from .corridor import tomato,check_pose,check_path,collision_pairs,parts,intersects
from .clearance import link_vertices


def right_pairs(q,kin,x,cfg,opening):
    right=link_vertices(q[7:],kin['right_arm'].base,cfg['asset_source'])
    pairs=set()
    for name in ('OuterBox','InnerBox'):
        for i,v in enumerate(parts(cfg['drawer_asset'],name)):
            if name=='InnerBox':
                v=np.concatenate([v+[opening-.003,0,0],v+[opening+.003,0,0]])
            obstacle=v@DRAWER_ROTATION.T+np.asarray(x[6:9])
            for link,vertices in right.items():
                if intersects(vertices,obstacle):pairs.add((link,name,i))
    return pairs


def check_initial_route(result,x,kin,cfg,*,start_step=0):
    """Qualify new initial grips without inheriting any non-handle collisions."""
    handle={('link7','InnerBox',10),('link8','InnerBox',10)}
    right_tcp=kin['right_arm'].fk(result['actions'][:,7:13])
    checked=set()
    for s in result['stages']:
        for i in range(max(start_step,s['start']),s['start']+s['steps']):
            q=result['actions'][i].copy()
            if i<end(result,'right_grip_settle'):opening=0.
            elif i<end(result,'right_pull') or s['name'] in ('right_push_coarse','right_push_fine'):
                opening=float((right_tcp[i,:3,3]-result['targets']['right_contact'][:3,3])@DRAWER_ROTATION[:,0])
            elif i<end(result,'right_push_coarse'):opening=cfg['pull_distance_m']
            else:opening=cfg['close_epsilon_m']
            jaws=(q[13],.004,.008,.012) if end(result,'right_close')<=i<end(result,'right_push_fine') else (q[13],)
            if end(result,'left_close')<=i<end(result,'left_release'):q[6]=max(q[6],result.get('p6_held_jaw_m',.029))
            for jaw in jaws:
                q[13]=jaw
                key=(q[7:].tobytes(),opening)
                if key not in checked:
                    bad=right_pairs(q,kin,x,cfg,opening)-handle
                    if bad:raise PlanningError('p6_initial_right_collision',dict(stage=s['name'],step=i,jaw=jaw,pairs=sorted(bad)))
                    checked.add(key)
                pairs=collision_pairs(q,kin,x,cfg,opening=max(1e-9,opening))
                if pairs:raise PlanningError('p6_initial_route_collision',dict(stage=s['name'],step=i,jaw=jaw,pairs=sorted(pairs)))
    contact=result['actions'][end(result,'right_handle_approach')-1].copy()
    contact[13]=.002
    bar=parts(cfg['drawer_asset'],'InnerBox')[10]@DRAWER_ROTATION.T+np.asarray(x[6:9])
    fingers=link_vertices(contact[7:],kin['right_arm'].base,cfg['asset_source'])
    if not all(intersects(fingers[name],bar,margin=0.) for name in ('link7','link8')):
        raise PlanningError('p6_initial_handle_not_straddled')
    contact[13]=.020
    if right_pairs(contact,kin,x,cfg,0.)&handle:
        raise PlanningError('p6_initial_approach_contact')
    return True


def initial_grasp(base,x,kin,cfg,qinit,max_steps):
    from .planner import plan_fixed
    errors=[];cache={}
    for tilt,twist,offset in itertools.product((-20,-30,-40),(-15,-30),(.005,.010,.015)):
        geometry=dict(base['selected_geometry'],right_tilt=tilt,right_twist=twist,right_contact_offset_m=offset)
        try:
            candidate=plan_fixed(x,kin,qinit,max_steps,cfg['pull_distance_m'],cfg.get('timing','baseline'),geometry=geometry,route_cache=cache)
            candidate['p6_initial_grasp']=geometry
            candidate['selected_geometry']=geometry
            state=candidate['actions'][end(candidate,'left_lift')-1].copy()
            handle={('link7','InnerBox',10),('link8','InnerBox',10)}
            for jaw in (0.,.004,.008,.012,.020):
                state[13]=jaw
                pairs=right_pairs(state,kin,x,cfg,cfg['pull_distance_m'])
                if pairs-handle:raise PlanningError('p6_initial_right_endpoint',dict(jaw=jaw,pairs=sorted(pairs-handle)))
                if jaw==0 and not handle<=pairs:raise PlanningError('p6_initial_handle_not_straddled')
            # Wide finger opening near the closed front wall can invalidate an otherwise usable grip.
            close=candidate['targets']['right_contact'].copy()
            close[:3,3]+=cfg['close_epsilon_m']*DRAWER_ROTATION[:,0]
            path,_=branch_path(kin['right_arm'],close[None],state[7:13],max_jump=np.deg2rad(10))
            released=state.copy();released[7:13]=path[-1]
            for jaw in np.linspace(0,.05,26):
                released[13]=jaw
                bad=right_pairs(released,kin,x,cfg,cfg['close_epsilon_m'])-handle
                if bad:raise PlanningError('p6_initial_release_collision',dict(jaw=jaw,pairs=sorted(bad)))
            state[6]=.029;state[13]=.008
            candidate=refine(candidate,x,kin,cfg,measured=state)
            if len(candidate['actions'])>max_steps:
                raise PlanningError('p6_initial_grasp:step_budget')
            check_initial_route(candidate,x,kin,cfg)
            candidate.pop('p6_held_jaw_m',None)
            candidate['p6_initial_grasp_rejections']=errors
            return candidate
        except PlanningError as exc:
            errors.append(dict(geometry=geometry,error=str(exc),diagnostics=exc.diagnostics))
    raise PlanningError('p6_initial_grasp_unavailable',dict(attempts=errors))


def release_pose(base,x,kin,cfg,tilt,yaw,side):
    grasp=base['targets']['left_grasp']
    object_r=Rotation.from_euler('x',90,degrees=True).as_matrix()
    object_pose=pose(object_r,np.asarray(x[:3])-object_r@np.array([0,.01,.035]))
    held=np.linalg.inv(grasp)@object_pose
    drawer_center=np.asarray(x[6:9])+DRAWER_ROTATION@np.array([cfg['pull_distance_m'],0.,0.])
    center=drawer_center+DRAWER_ROTATION@np.array([cfg.get('p6_forward_m',.03),side,0.])
    angle=np.arctan2(*(center-kin['left_arm'].base[:3,3])[1::-1])+np.deg2rad(yaw)
    r=Rotation.from_euler('z',angle).as_matrix()@Rotation.from_euler('YX',[tilt,180],degrees=True).as_matrix()
    relative=tomato(cfg['asset_source'])@held[:3,:3].T+held[:3,3]
    world=relative@r.T
    p=center.copy();p[:2]-=(world[:,:2].min(0)+world[:,:2].max(0))/2
    p[2]=x[8]+.007+cfg.get('p6_release_gap_m',.010)-world[:,2].min()
    local=(world+p-drawer_center)@DRAWER_ROTATION
    if (abs(local[:,:2]).max(0)>.067).any():raise PlanningError('p6_object_footprint')
    return pose(r,p)


def retreat(result,x,kin,cfg,cut,measured=None,*,failure=False):
    q=result['actions'][cut-1].copy() if measured is None else measured.copy()
    start=kin['left_arm'].fk(q[None,:6])[0]
    errors=[]
    # Lower routes earn clearance through the same full-arm checks, without the fixed high wrist pose.
    routes=itertools.chain(((lift,side,.18) for lift,side in itertools.product((.12,.10),(.10,.08))),
                           ((.04,side,height) for height,side in itertools.product((.16,.14,.12),(.10,.14))))
    for lift,side,height in routes:
        try:
            b=Suffix(result,cut,kin,1000 if failure else 980)
            b.append('failure_unload' if failure else 'left_unload_to_measured',10,joint=q[:6])
            high=start.copy();high[2,3]=max(start[2,3]+lift,x[8]+height);high[1,3]+=side/2
            park=high.copy();park[1,3]+=side/2
            compact=cfg.get('timing')=='compact' and not failure
            b.append('failure_lift' if failure else 'left_withdraw',35 if compact else 40,target=high)
            b.append('failure_exit' if failure else 'left_clearance_park',25 if compact else 30,target=park)
            if not failure:b.append('left_open_clear',10,grip=.05)
            b.append('failure_settle' if failure else 'left_park_settle',10)
            out=b.finish()
            # Endpoint and path are independently checked; an intersecting initial state must be allowed to unload.
            begin=cut+10
            if failure:
                allowed=collision_pairs(q,kin,x,cfg)
                for action in out['actions'][begin:]:
                    pairs=collision_pairs(action,kin,x,cfg)
                    if not pairs<=allowed:raise PlanningError('p6_escape_new_collision')
                    allowed=pairs
                check_pose(out['actions'][-1],kin,x,cfg)
            else:check_path(out['actions'][begin:],kin,x,cfg)
            clear(out['actions'][-1,:7],kin['left_arm'].base,np.asarray(x[6:9]),cfg['asset_source'])
            out['targets']=dict(out['targets'],left_withdraw=park)
            if failure:
                out['close_start_step']=1001;out['p6_escape']=True
                return out
            return append_close_tail(out,x,kin,980,epsilon=cfg['close_epsilon_m'],skip_park=True,timing=cfg.get('timing','baseline'))
        except PlanningError as exc:errors.append(dict(lift=lift,side=side,height=height,error=str(exc),diagnostics=exc.diagnostics))
    raise PlanningError('p6_retreat_unavailable',dict(attempts=errors))


def check_landing(q,held_jaw,kin,x,cfg):
    # shortcut: sample single-joint tracking errors, expand if physical traces show coupled misses.
    for jaw in (held_jaw-.001,held_jaw+.001,cfg.get('p6_release_jaw_m',.03)):
        state=q.copy();state[6]=jaw
        check_pose(state,kin,x,cfg)
        for joint,offset in itertools.product(range(6),(-np.deg2rad(.6),np.deg2rad(.6))):
            perturbed=state.copy();perturbed[joint]+=offset
            check_pose(perturbed,kin,x,cfg)


def _refine(base,x,kin,cfg,measured=None,*,checked_retreat=False):
    cut=end(base,'left_lift');errors=[]
    gaps=[cfg.get('p6_release_gap_m',.010)]
    if measured is not None:gaps+= [gap for gap in (.030,.035) if gap>gaps[0]]
    candidates=itertools.product(gaps,cfg.get('p6_tilts',[-50,-70,-80]),cfg.get('p6_yaws',[0,15,-15]),cfg.get('p6_sides',[0,.01,-.01]))
    for gap,tilt,yaw,side in candidates:
        try:
            cfg=dict(cfg,p6_release_gap_m=gap)
            target=release_pose(base,x,kin,cfg,tilt,yaw,side)
            hover=target.copy();hover[2,3]=max(base['targets']['left_lift'][2,3],target[2,3]+.12)
            hover[1,3]+=.06
            high=target.copy();high[2,3]=max(target[2,3]+.12,x[8]+.18)
            high[1,3]+=.05
            park=high.copy();park[1,3]+=.05
            poses=[hover]
            for dest in ((target,) if checked_retreat else (target,high,park)):
                start=poses[-1]
                for u in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(dest[:3,3]-start[:3,3])/.002)))+1)[1:]:
                    t=dest.copy();t[:3,3]=(1-u)*start[:3,3]+u*dest[:3,3];poses.append(t)
            path,margin=branch_path(kin['left_arm'],np.array(poses),base['actions'][cut-1,:6],max_jump=np.deg2rad(10))
            b=Suffix(base,cut,kin,980);b.margin=min(b.margin,margin)
            compact=cfg.get('timing')=='compact'
            b.append('left_transport',50 if compact else 65,joint=path[0])
            b.append('left_lower',45 if compact else 50,target=target)
            b.append('left_release',20,grip=cfg.get('p6_release_jaw_m',.03))
            b.append('release_settle',15)
            out=b.finish();out['targets']=dict(base['targets'],left_place=target,left_hover=hover)
            # Reject the final closed/open poses first, before checking the complete route.
            check_pose(out['actions'][end(out,'left_lower')-1],kin,x,cfg)
            check_pose(out['actions'][-1],kin,x,cfg)
            if measured is not None:
                jaws=(.004,.008,.012,measured[13]) if base.get('p6_initial_grasp') else (None,)
                for jaw in jaws:
                    landing=out['actions'][end(out,'left_lower')-1].copy()
                    held=out['actions'][cut:].copy()
                    if jaw is not None:landing[13]=jaw;held[:,13]=jaw
                    check_landing(landing,measured[6],kin,x,cfg)
                    held[:,6]=np.maximum(held[:,6],measured[6])
                    check_path(held,kin,x,cfg)
            check_path(out['actions'][cut:],kin,x,cfg)
            out=retreat(out,x,kin,cfg,len(out['actions']))
            out['p6_runtime_replan_at']=end(out,'release_settle')
            out['p6_candidate']=dict(tilt=tilt,yaw=yaw,side=side,release_gap_m=cfg.get('p6_release_gap_m',.010))
            out['p6_rejections']=errors;out['p5_mode']='p6'
            if measured is not None:out['p6_held_jaw_m']=float(measured[6])
            return out
        except PlanningError as exc:errors.append(dict(tilt=tilt,yaw=yaw,side=side,release_gap_m=gap,error=str(exc),diagnostics=exc.diagnostics))
    raise PlanningError('p6_all_corridors_failed',dict(candidates=errors))


def refine(base,x,kin,cfg,measured=None):
    try:
        return _refine(base,x,kin,cfg,measured)
    except PlanningError as original:
        # Preserve every working route; qualify alternative retreat only after failure.
        try:
            out=_refine(base,x,kin,cfg,measured,checked_retreat=True)
        except PlanningError as fallback:
            raise PlanningError('p6_all_corridors_failed',dict(
                candidates=original.diagnostics.get('candidates',[]),
                retreat_fallback=fallback.diagnostics)) from fallback
        out['p6_retreat_fallback']=True
        out['p6_original_rejections']=original.diagnostics
        return out


def retry_left_grasp(model,q):
    r,i=model.result,model.cursor
    if r.get('p6_left_contact_retry'):
        raise PlanningError('p6_left_contact_empty')
    errors=[]
    for tilt in (0,-30):
        try:
            anchor=np.asarray(model.parameters[:3])
            angle=np.arctan2(*(anchor-model.kin['left_arm'].base[:3,3])[1::-1])
            rotation=Rotation.from_euler('z',angle).as_matrix()@Rotation.from_euler('YX',[tilt,180],degrees=True).as_matrix()
            grasp=pose(rotation,anchor+.02*rotation[:,2]-[0,0,.01])
            high=grasp.copy();high[2,3]=r['targets']['left_lift'][2,3]
            b=Suffix(r,i,model.kin,980)
            b.stages=[dict(s,name='initial_attempt_'+s['name']) if s['name'] in ('left_close','left_grip_settle','left_lift') else s for s in b.stages]
            b.append('left_retry_unload',10,joint=q[:6])
            b.append('left_retry_open',10,grip=.05)
            path,margin=branch_path(model.kin['left_arm'],high[None],q[:6],max_jump=np.deg2rad(10))
            b.margin=min(b.margin,margin)
            b.append('left_retry_overhead',30,joint=path[-1])
            b.append('left_retry_approach',35,target=grasp)
            b.append('left_close',25,grip=0.)
            b.append('left_grip_settle',10)
            b.append('left_lift',35,target=high)
            out=b.finish()
            out['targets']=dict(out['targets'],left_grasp=grasp,left_lift=high)
            object_r=Rotation.from_euler('x',90,degrees=True).as_matrix()
            object_vertices=tomato(model.config['asset_source'])@object_r.T+anchor-object_r@np.array([0,.01,.035])
            contact=out['actions'][end(out,'left_close')-1,:7].copy()
            for jaw,expected in ((0.,{'link7','link8'}),(.05,set())):
                contact[6]=jaw
                links=link_vertices(contact,model.kin['left_arm'].base,model.config['asset_source'])
                touching={name for name,v in links.items() if intersects(v,object_vertices,margin=0.)}
                if touching!=expected:raise PlanningError('p6_left_retry:contact_geometry')
            held=out['actions'][-1].copy();held[6]=.029;held[13]=q[13]
            out=refine(out,model.parameters,model.kin,model.config,measured=held)
            check_path(out['actions'][i:],model.kin,model.parameters,model.config)
            # The measured right hold and the occupied left jaw must also clear the new approach.
            occupied=out['actions'][i:end(out,'left_lift')].copy();occupied[:,13]=q[13]
            occupied[end(out,'left_close')-i:,6]=.029
            check_path(occupied,model.kin,model.parameters,model.config)
            if r.get('p6_initial_grasp'):
                check_initial_route(out,model.parameters,model.kin,model.config,start_step=i)
            if len(out['actions'])>model.max_steps:raise PlanningError('p6_left_retry:step_budget')
            out.pop('p6_held_jaw_m',None)
            out['p6_left_contact_retry']=dict(step=i,jaw_m=float(q[6]),tilt=tilt,rejections=errors)
            return out
        except PlanningError as exc:
            errors.append(dict(tilt=tilt,error=str(exc),diagnostics=exc.diagnostics))
    raise PlanningError('p6_left_retry_unavailable',dict(attempts=errors))


def update(model,q):
    r,i=model.result,model.cursor
    if model.execution_failure or not r['stages']:return
    if i==end(r,'right_grip_settle') and q[13]<.002:
        if r.get('p6_initial_contact_retry'):
            raise PlanningError('p6_initial_contact_empty')
        # An empty jaw before the first pull permits a new initial approach at closed geometry.
        initial=q.copy();initial[[6,13]]=np.clip(initial[[6,13]],0,.05)
        previous=r['actions'][i-1];n=10
        for _ in range(6):
            unload=smooth_joint(previous,initial,n);ratio=1.
            for arm,sl in (('left_arm',slice(0,6)),('right_arm',slice(7,13))):
                v=np.diff(np.vstack([previous[sl],unload[:,sl],initial[sl]]),axis=0)/.04
                acc=np.diff(np.vstack([np.zeros((1,6)),v]),axis=0)/.04
                ratio=max(ratio,float((abs(v)/model.kin[arm].velocity).max()),float(np.sqrt(abs(acc).max()/20)))
            if ratio<=1.0001:break
            n=int(np.ceil(n*ratio*1.02))
        else:raise PlanningError('p6_initial_unload:retiming_failed')
        limits=model.kin['right_arm'].limits
        margin=float(np.minimum(unload[:,7:13]-limits[:,0],limits[:,1]-unload[:,7:13]).min())
        if margin<np.deg2rad(2):raise PlanningError('p6_initial_unload:joint_margin')
        for state in unload:
            bad=right_pairs(state,model.kin,model.parameters,model.config,0.)-{('link7','InnerBox',10),('link8','InnerBox',10)}
            other=collision_pairs(state,model.kin,model.parameters,model.config,opening=1e-9)
            if bad or other:raise PlanningError('p6_initial_unload_collision',dict(right=sorted(bad),other=sorted(other)))
        offset=i+n
        candidate=initial_grasp(r,model.parameters,model.kin,model.config,initial,model.max_steps-offset)
        candidate['min_margin_rad']=min(candidate['min_margin_rad'],margin)
        candidate['actions']=np.concatenate([r['actions'][:i],unload,candidate['actions']])
        candidate['stages']=[dict(s,name='initial_attempt_'+s['name']) for s in r['stages'] if s['start']+s['steps']<=i]+[
            dict(name='initial_contact_unload',start=i,steps=n)]+[
            dict(s,start=s['start']+offset) for s in candidate['stages']]
        for key in ('close_start_step','p6_runtime_replan_at'):
            candidate[key]+=offset
        candidate['p6_initial_contact_retry']=dict(step=i,unload_steps=n,jaw_m=float(q[13]))
        model.result=candidate
        r=candidate
    if r.get('p6_initial_grasp') and (end(r,'left_lift')<=i or r.get('p6_left_contact_retry')) and i<r['close_start_step']:
        actual=model.kin['right_arm'].fk(q[None,7:13])[0]
        if not .002<=q[13]<=.012 or np.linalg.norm(actual[:3,3]-r['targets']['right_pull'][:3,3])>.003:
            raise PlanningError('p6_right_grasp_feedback',dict(jaw_m=float(q[13])))
        bad=right_pairs(q,model.kin,model.parameters,model.config,model.config['pull_distance_m'])-{('link7','InnerBox',10),('link8','InnerBox',10)}
        if bad:raise PlanningError('p6_right_feedback_collision',dict(pairs=sorted(bad)))
    if i==end(r,'left_lift') and q[6]<.002:
        model.result=retry_left_grasp(model,q)
        return
    if i==end(r,'left_lift') and 'p6_held_jaw_m' not in r:
        check_pose(q,model.kin,model.parameters,model.config)
        model.result=refine(r,model.parameters,model.kin,model.config,measured=q)
        if r.get('p6_initial_grasp'):check_initial_route(model.result,model.parameters,model.kin,model.config,start_step=i)
        r=model.result
    stage=next((s for s in r['stages'] if s['start']<=i<s['start']+s['steps']),None)
    if stage is None:return
    name=stage['name'];actual=model.kin['left_arm'].fk(q[None,:6])[0]
    if name=='left_lower':
        history=model.p6_history;history.append((i,float(actual[2,3])))
        if len(history)>8:history.pop(0)
        error=float(np.rad2deg(abs(q[:6]-r['actions'][max(0,i-1),:6]).max()))
        if len(history)==8 and i-stage['start']>=10:
            old_command=model.kin['left_arm'].fk(r['actions'][history[0][0]:history[0][0]+1,:6])[0]
            new_command=model.kin['left_arm'].fk(r['actions'][i:i+1,:6])[0]
            if error>2 and old_command[2,3]-new_command[2,3]>.004 and history[0][1]-actual[2,3]<.001:
                raise PlanningError('p6_descent_stall',dict(joint_error_deg=error))
    if name=='left_release' and i==stage['start']:
        target=r['targets']['left_place']
        if np.linalg.norm(actual[:3,3]-target[:3,3])>.015:
            raise PlanningError('p6_release_pose_error')
        check_pose(q,model.kin,model.parameters,model.config)
    if r.get('p6_runtime_replan_at')==i and not model.replanned:
        model.result=retreat(r,model.parameters,model.kin,model.config,i,q);model.replanned=True
        if r.get('p6_initial_grasp'):check_initial_route(model.result,model.parameters,model.kin,model.config,start_step=i)
    if name=='right_push_coarse' and i==stage['start']:
        clear(q[:7],model.kin['left_arm'].base,model.parameters[6:9],model.config['asset_source'])
        if q[6]<.04:raise PlanningError('p6_close_before_jaw_open')


def escape(model,q,exc):
    model.execution_failure=dict(step=model.cursor+1,reason=str(exc),diagnostics=exc.diagnostics)
    try:
        model.result=retreat(model.result,model.parameters,model.kin,model.config,model.cursor,q,failure=True)
    except PlanningError as failure:
        model.execution_failure['escape_error']=str(failure)
        try:
            b=Suffix(model.result,model.cursor,model.kin,1000)
            b.append('failure_unload',10,joint=q[:6]);model.result=b.finish()
        except PlanningError as unload:
            model.execution_failure['unload_error']=str(unload)
            model.result['actions']=model.result['actions'][:model.cursor].copy()
        model.result['close_start_step']=1001
