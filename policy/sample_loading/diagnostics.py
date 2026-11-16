"""Offline geometry diagnostics and explicit 10D oracle substitutions; no simulator access."""
import numpy as np

MODES = {'O0': 'predicted', 'O1': 'grasp', 'O2': 'rack', 'O3': 'both', 'C0': 'predicted'}


def mix_parameters(predicted, targets, mode):
    """Return a finite (10,) vector; replace only the named position/angle components."""
    x = np.asarray(predicted, dtype=float).copy()
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError('invalid predicted parameters')
    if mode not in MODES.values():
        raise ValueError(mode)
    if mode == 'predicted':
        if targets is not None:
            raise ValueError('predicted policy must receive no ground truth')
        return x
    y = np.asarray(targets, dtype=float)
    if y.shape != (10,) or not np.isfinite(y).all():
        raise ValueError('invalid diagnostic targets')
    if mode in ('grasp', 'both'):
        x[[0, 1, 2, 6, 7]] = y[[0, 1, 2, 6, 7]]
    if mode in ('rack', 'both'):
        x[[3, 4, 5, 8, 9]] = y[[3, 4, 5, 8, 9]]
    return x


def pose_difference(a, b):
    """Compare two 4x4 rigid poses: position in mm, SO(3) angle and directed tube-axis angle."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4, 4) or b.shape != (4, 4) or not np.isfinite([a,b]).all():
        raise ValueError('invalid rigid poses')
    for t in (a, b):
        if not np.allclose(t[3], [0,0,0,1], atol=1e-5) or not np.allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=1e-4) or np.linalg.det(t[:3,:3]) < .999:
            raise ValueError('nonrigid pose')
    def rotation(t):
        u, _, vt = np.linalg.svd(t[:3, :3])
        return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    relative = rotation(a) @ rotation(b).T
    sine = np.linalg.norm([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0], relative[1,0]-relative[0,1]]) / 2
    cosine = np.clip((np.trace(relative)-1)/2, -1, 1)
    axis_a, axis_b = a[:3,2]/np.linalg.norm(a[:3,2]), b[:3,2]/np.linalg.norm(b[:3,2])
    return dict(position_mm=float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),
                rotation_deg=float(np.degrees(np.arctan2(sine, cosine))),
                tube_axis_deg=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(axis_a,axis_b)), np.dot(axis_a,axis_b)))))


def initial_difference(actual, expert):
    """Report pose and robot differences separately; use strict declared bounds for pairing."""
    out = {k: pose_difference(actual[k], expert[k]) for k in ('cube','rack')}
    q, e = np.asarray(actual['obs_qpos']).reshape(-1), np.asarray(expert['obs_qpos']).reshape(-1)
    if q.shape != (14,) or e.shape != (14,) or not np.isfinite([q,e]).all():
        raise ValueError('invalid initial qpos')
    out['joint_max_rad'] = float(np.max(abs(q[[0,1,2,3,4,5,7,8,9,10,11,12]]-e[[0,1,2,3,4,5,7,8,9,10,11,12]])))
    out['gripper_max_mm'] = float(np.max(abs(q[[6,13]]-e[[6,13]]))*50)
    out['passed'] = (out['cube']['position_mm'] <= 1 and out['cube']['rotation_deg'] <= .1
                     and out['rack']['position_mm'] <= .01 and out['rack']['rotation_deg'] <= .01
                     and out['joint_max_rad'] <= 1e-5 and out['gripper_max_mm'] <= .001)
    return out


class StaticController:
    """Adapter for the unmodified historical C0 action sequence; adds only result recording."""
    def __init__(self, model, output):
        self.model, self.output = model, output
        self.phase = 'static_c0'
        model.result['search'] = dict(applied=False, reason='historical_C0_no_search')
    def before_step(self, obs):
        pass
    def after_step(self, obs):
        pass
    def summary(self):
        from harness.run_episode import write
        write(self.output/'search_summary.json', dict(variant='C0', phase=self.phase,
              action_steps=len(self.model.result['actions']), actual_lateral_moves=19,
              secondary_localization=False, finish_reason='adaptive_secondary_reverse_open_loop'))


def stage_diagnostics(directory, geometry):
    """Derive stage-end tube-axis/bottom-center metrics from saved GT traces, never for control."""
    import json
    from pathlib import Path
    ep=Path(directory)
    with (ep/'trace.jsonl').open() as stream:
        trace={x['step']:x for x in map(json.loads,stream)}
    stages=json.loads((ep/'plan.json').read_text()).get('stages',[])
    samples=[]
    for stage in stages:
        step=min(stage['start']+stage['steps'],max(trace))
        row=trace[step];tube=np.asarray(row['tube_pose']);rack=np.asarray(row['rack_pose']);eef=np.asarray(row['left_eef_pose'])
        relative=np.linalg.inv(rack)@tube
        bottom=(relative@np.array([0,0,geometry['tube_bounds_m'][0][2],1]))[:3]
        samples.append(dict(stage=stage['name'],step=step,
                       tube_axis_tilt_deg=float(np.degrees(np.arccos(np.clip(relative[2,2],-1,1)))),
                       bottom_center_rack_m=bottom.tolist(),
                       bottom_xy_error_to_target_mm=float(np.linalg.norm(bottom[:2]-geometry['hole_center_xy_m'])*1000),
                       bottom_depth_below_plate_mm=float((geometry['reference_plate_z_m']-bottom[2])*1000),
                       hand_to_tube=(np.linalg.inv(eef)@tube).tolist()))
    return samples
