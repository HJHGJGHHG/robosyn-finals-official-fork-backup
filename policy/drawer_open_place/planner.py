"""Sequential baseline: right opens/holds, left places/withdraws, drawer stays open."""
import itertools
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import targets,DRAWER_ROTATION
from .motion import PlanningError, branch_path, smooth_joint


def plan_fixed(parameters, kin, qinit, max_steps=900, pull_distance=.19, timing='baseline', *, geometry=None, route_cache=None):
    """Return Nx14 physical joint commands and stage diagnostics at 25 Hz.

    kin is the existing CobotMagic IK/FK adapter. qinit uses meters for jaws.
    Contact motions follow straight Cartesian lines; free approaches use joint
    interpolation. Raises a named PlanningError, never substitutes an idle plan.
    Collision/contact qualification requires subsequent simulator rollouts.
    timing='compact' only changes stage duration; geometry and contact timing
    remain paired with the baseline and all retiming limits still apply.
    """
    if timing not in ('baseline','compact'): raise ValueError('Unknown timing')
    compact = {'open_grippers':10,'right_prehandle':35,'right_hold':10,
               'left_pregrasp':40,'left_lift':35,'left_transport':50,
               'left_withdraw':30,'final_hold':25}
    current = np.asarray(qinit, float).copy()
    if current.shape != (14,) or not np.isfinite(current).all():
        raise ValueError('Expected finite physical qpos (14,)')
    if ((current[[6, 13]] < 0) | (current[[6, 13]] > .05)).any():
        raise ValueError('Physical jaw range is [0,.05] meters')
    for arm,sl in [('left_arm',slice(0,6)),('right_arm',slice(7,13))]:
        if ((current[sl]<kin[arm].limits[:,0]-1e-4)|(current[sl]>kin[arm].limits[:,1]+1e-4)).any():
            raise PlanningError(arm+':initial_joint_limits')
    geometry = dict(geometry or {})
    twist = geometry.pop('right_twist', 0.)
    contact_offset = geometry.pop('right_contact_offset_m', 0.)
    if not np.isfinite(twist) or abs(twist)>45:
        raise ValueError('Initial right twist must be within 45 degrees')
    if not np.isfinite(contact_offset) or not 0<=contact_offset<=.02:
        raise ValueError('Initial right contact offset must be within 20 mm')
    g = targets(parameters, kin['left_arm'].base, pull_distance, **geometry)
    if twist:
        # Rotate about the closing axis so the fingers still straddle the handle.
        for name in ('right_pre','right_contact','right_pull'):
            g[name][:3,:3] = g[name][:3,:3] @ Rotation.from_euler('y',twist,degrees=True).as_matrix()
    if contact_offset:
        for name in ('right_pre','right_contact','right_pull'):
            g[name][:3,3] += contact_offset*DRAWER_ROTATION[:,0]
    if route_cache is None: route_cache = {}
    margins, jumps = [], []
    chunks, stages = [], []
    dt = .04
    routes = {}
    # Select each arm's contact-path branch jointly, so a locally nearest IK
    # solution cannot strand a later pull/lower segment on another branch.
    for arm, sl, sequence in [
        ('right_arm',slice(7,13),[('right_prehandle','right_pre'),('right_handle_approach','right_contact'),('right_pull','right_pull')]),
        ('left_arm',slice(0,6),[('left_pregrasp','left_pre'),('left_approach','left_grasp'),('left_lift','left_lift')]),
        ('left_arm',slice(0,6),[('left_transport','left_hover'),('left_lower','left_place'),('left_withdraw','left_withdraw')]),
    ]:
        poses = [g[sequence[0][1]]]; spans = [(0,0)]
        for _, key in sequence[1:]:
            start, end = poses[-1], g[key]
            count = max(2,int(np.ceil(np.linalg.norm(end[:3,3]-start[:3,3])/.002)))
            begin = len(poses)-1
            for u in np.linspace(0,1,count+1)[1:]:
                t=end.copy(); t[:3,3]=start[:3,3]*(1-u)+end[:3,3]*u; poses.append(t)
            spans.append((begin,len(poses)-1))
        poses=np.asarray(poses)
        try:
            seed=routes['left_lift'][0][-1] if sequence[0][0]=='left_transport' else current[sl]
            key = (arm, poses.tobytes(), np.asarray(seed).tobytes())
            if key not in route_cache:
                try: route_cache[key] = branch_path(kin[arm],poses,seed,max_jump=np.deg2rad(10))
                except PlanningError as exc: route_cache[key] = exc
            cached = route_cache[key]
            if isinstance(cached, PlanningError): raise cached
            path,margin = cached
            margins.append(margin)
            jumps.append(float(np.max(abs(np.diff(path,axis=0)))) if len(path)>1 else 0.)
        except PlanningError as exc:
            raise PlanningError(sequence[0][0]+':contact_route:'+str(exc), diagnostics=dict(exc.diagnostics, arm=arm, chain=[name for name,_ in sequence], spans=spans)) from exc
        for (name,_),(start,end) in zip(sequence,spans):
            routes[name]=(path[start:end+1],poses[start:end+1])

    def append(name, n, arm=None, target=None, grip=None, straight=False):
        nonlocal current
        if timing=='compact':
            n=compact.get(name,n)
            if name=='right_pull':n=int(np.ceil(n*pull_distance/.19))
        sl = slice(0, 6) if arm == 'left_arm' else slice(7, 13)
        if arm:
            path,poses=routes[name]
            if straight and np.max(abs(path[0]-current[sl])) > .01:
                raise PlanningError(name+':branch_join')
        for _ in range(6):
            a = np.tile(current, (n, 1))
            if arm:
                if straight:
                    u = np.arange(1, n+1)/n
                    index = (10*u**3-15*u**4+6*u**5)*(len(path)-1)
                    a[:, sl] = np.column_stack([np.interp(index, np.arange(len(path)), path[:, j]) for j in range(6)])
                    expected = np.column_stack([np.interp(index, np.arange(len(path)), poses[:, j, 3]) for j in range(3)])
                    if np.linalg.norm(kin[arm].fk(a[:, sl])[:, :3, 3]-expected, axis=1).max() > .0005:
                        raise PlanningError(name+':cartesian_residual')
                    if twist and arm=='right_arm' and Rotation.from_matrix(kin[arm].fk(a[:,sl])[:,:3,:3]@poses[-1,:3,:3].T).magnitude().max()>np.deg2rad(.25):
                        raise PlanningError(name+':angular_residual')
                else:
                    a[:, sl] = smooth_joint(current[sl], path[-1], n)
                v = np.diff(np.vstack([current[sl], a[:, sl], a[-1, sl]]), axis=0)/dt
                acc = np.diff(np.vstack([np.zeros((1, 6)), v]), axis=0)/dt
                ratio = max(float(np.max(abs(v)/kin[arm].velocity)), float(np.sqrt(np.max(abs(acc))/20)))
                if ratio > 1.0001:
                    n = int(np.ceil(n*ratio*1.02)); continue
            if grip:
                for col, value in grip.items():
                    a[:, col] = smooth_joint(current[col:col+1], np.array([value]), n)[:, 0]
            break
        else:
            raise PlanningError(name+':retiming_failed')
        start = sum(len(x) for x in chunks)
        if start+n > max_steps:
            raise PlanningError(name+':step_budget')
        chunks.append(a); stages.append({'name':name, 'start':start, 'steps':n})
        current = a[-1].copy()

    append('open_grippers',15,grip={6:.05,13:.05})
    append('right_prehandle',45,'right_arm',g['right_pre'])
    append('right_partial_close',10,grip={13:.02})
    append('right_handle_approach',40,'right_arm',g['right_contact'],straight=True)
    append('right_close',20,grip={13:0})
    append('right_grip_settle',10)
    append('right_pull',110,'right_arm',g['right_pull'],straight=True)
    append('right_hold',20)
    append('left_pregrasp',55,'left_arm',g['left_pre'])
    append('left_approach',30,'left_arm',g['left_grasp'],straight=True)
    append('left_close',25,grip={6:0})
    append('left_grip_settle',10)
    append('left_lift',45,'left_arm',g['left_lift'],straight=True)
    append('left_transport',65,'left_arm',g['left_hover'])
    append('left_lower',50,'left_arm',g['left_place'],straight=True)
    append('left_release',20,grip={6:.05})
    append('release_settle',20)
    append('left_withdraw',40,'left_arm',g['left_withdraw'],straight=True)
    append('final_hold',50)
    actions = np.concatenate(chunks)
    return {'actions':actions, 'stages':stages, 'targets':g,
            'collision_qualified':False, 'opening_verified':False,
            'pull_distance_m':pull_distance, 'hz':25, 'timing':timing,
            'min_margin_rad':float(min(margins)), 'max_contact_jump_rad':max(jumps),
            'joint_travel_rad':float(np.abs(np.diff(np.vstack([qinit,actions]),axis=0))[:,[0,1,2,3,4,5,7,8,9,10,11,12]].sum())}


def plan(parameters, kin, qinit, max_steps=1000, pull_distance=.19, timing='baseline'):
    """Search at most 120 routes; prefer stable geometry within quality tolerances."""
    started=time.perf_counter();cache={};attempts=[];feasible=[]
    candidates=itertools.chain(
        itertools.product((.04,.06,.10),(-30,-20,-40),(-70,-50,-80)),
        itertools.product((.04,.06,.10),(-60,-65,-70,-72.5,-75,-77.5,-80),(-70,-50,-80)),
        itertools.product((.02,),(-30,-20,-40,-60,-65,-70,-72.5,-75,-77.5,-80),(-70,-50,-80)))
    for index,(distance,right,place) in enumerate(candidates):
        # Try steeper tilts, then a shorter pregrasp, only after prior routes fail.
        if index in (27,90) and feasible: break
        geometry=dict(pregrasp_distance=distance,right_tilt=right,place_tilt=place)
        try:
            result=plan_fixed(parameters,kin,qinit,max_steps,pull_distance,timing,geometry=geometry,route_cache=cache)
        except PlanningError as exc:
            attempts.append(dict(index=index,geometry=geometry,ok=False,error=str(exc),diagnostics=exc.diagnostics));continue
        score=(-result['min_margin_rad'],result['max_contact_jump_rad'],result['joint_travel_rad'],index)
        attempt=dict(index=index,geometry=geometry,ok=True,steps=len(result['actions']),raw_score=list(score))
        attempts.append(attempt);feasible.append((attempt,result))
    diagnostics=dict(candidates=attempts,feasible_candidates=sum(a['ok'] for a in attempts),candidate_count=len(attempts),unique_route_solves=len(cache),seconds=time.perf_counter()-started)
    if not feasible:raise PlanningError('all_candidates_failed',diagnostics=diagnostics)
    # Anchor each equivalence band to the best surviving value, avoiding chained ties.
    survivors=[a for a,_ in feasible]
    for a in survivors:
        a['score']=[0,(.04,.06,.10,.02).index(a['geometry']['pregrasp_distance']),a['index']]
    for column,tolerance in enumerate((np.deg2rad(.1),np.deg2rad(.05),.01)):
        minimum=min(a['raw_score'][column] for a in survivors)
        survivors=[a for a in survivors if a['raw_score'][column]<=minimum+tolerance]
        for a in survivors:a['score'][0]-=1
    selected,best=min(feasible,key=lambda item:tuple(item[0]['score']))
    best['selected_geometry']=selected['geometry'];best['selected_candidate']=selected['index']
    best['search']=diagnostics
    return best
