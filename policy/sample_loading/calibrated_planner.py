"""Same bounded template with offline held-transform calibration; no env access."""
import numpy as np
from .geometry import candidate_geometry, centered_takeover, offset
from .planner import plan_candidate, PlanningError

def geometry(tube,rack,base,geom,calibration,shift,yaw,delta,hyaw,retreat):
    g=candidate_geometry(tube,rack,base,geom,shift,yaw,delta,hyaw,retreat)
    g['gr']=np.asarray(calibration['constants']['right_grasp'],dtype=float)
    g['tube_h']=g['right']['hand']@g['gr']
    g['left'],g['left_pre'],g['left_proof']=centered_takeover(g['tube_h'],g['left'],
        calibration['contact_axial_m'],depth=calibration['takeover_depth_m'])
    # Measured mean after prior left transport, fitted once offline. Never
    # substitute the current environment's actual grasp transform here.
    g['gl']=np.asarray(calibration['constants']['left_grasp'],dtype=float)
    g['place']=g['tube_goal']@np.linalg.inv(g['gl'])
    depth=geom['reference_plate_z_m']+.006-(geom['support_local_z_m']+.0015)
    g['near']=offset(g['place'],rack[:3,2]*depth)
    g['above']=offset(g['near'],rack[:3,2]*.014)
    g['withdraw']=offset(g['place'],rack[:3,2]*.08)
    g['params']['template']='fixed_grasp_v2'
    return g

def plan(tube,rack,kin,qinit,geom,calibration,max_candidates=864):
    failures=[];attempts=0
    # Centered contact fixes the axial position, so the old shift sweep would
    # duplicate identical candidates. Keep the old nominal shift in metadata.
    for delta in [(0,0,0),(-.03,0,0),(0,.03,0),(0,0,.03)]:
        for hyaw in [0,-10,10]:
            for yaw in [0,-10,10,-20,20,-90,90,180]:
                for retreat in ['local','toward_base','world_negative_y']:
                    attempts+=1
                    g=geometry(tube,rack,kin['right_arm'].base,geom,calibration,.11,yaw,delta,hyaw,retreat)
                    try:
                        result=plan_candidate(tube,rack,kin,qinit,geom,.11,yaw,delta,hyaw,retreat,geometry_override=g)
                        result.update(candidate_attempts=attempts,rejections=failures)
                        return result
                    except PlanningError as exc:
                        failures.append({'yaw':yaw,'delta':delta,'hand_yaw':hyaw,'retreat_mode':retreat,'reason':str(exc)})
                    if attempts>=max_candidates:
                        error=PlanningError('no_calibrated_plan');error.rejections=failures;raise error
    error=PlanningError('no_calibrated_plan');error.rejections=failures;raise error
