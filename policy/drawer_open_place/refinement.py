"""P5 suffixes reuse the selected P4 grasp, branch solver and right closing tail."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import targets
from .motion import PlanningError, branch_path, smooth_joint
from .close_tail import append_close_tail


class Suffix:
    def __init__(self, result, cut, kin, budget):
        self.result = dict(result)
        self.chunks = [result['actions'][:cut].copy()]
        self.stages = [dict(s) for s in result['stages'] if s['start']+s['steps'] <= cut]
        self.current = self.chunks[0][-1].copy()
        self.kin, self.budget = kin, budget
        self.margin, self.jump = result['min_margin_rad'], result['max_contact_jump_rad']

    def append(self, name, n, target=None, joint=None, grip=None):
        k = self.kin['left_arm']
        if target is not None:
            start = k.fk(self.current[None, :6])[0]
            if Rotation.from_matrix(start[:3, :3]@target[:3, :3].T).magnitude() > np.deg2rad(.25):
                raise PlanningError(name+':rotation_must_finish_above_drawer')
            count = max(2, int(np.ceil(np.linalg.norm(target[:3, 3]-start[:3, 3])/.002)))
            poses = np.repeat(target[None], count+1, axis=0)
            poses[:, :3, 3] = np.linspace(start[:3, 3], target[:3, 3], count+1)
            path, margin = branch_path(k, poses, self.current[:6], max_jump=np.deg2rad(10))
            if abs(path[0]-self.current[:6]).max() > .01:
                raise PlanningError(name+':branch_join')
            self.margin = min(self.margin, margin)
            self.jump = max(self.jump, float(abs(np.diff(path, axis=0)).max()))
        for _ in range(6):
            a = np.tile(self.current, (n, 1))
            if target is not None:
                u = np.arange(1, n+1)/n
                ix = (10*u**3-15*u**4+6*u**5)*(len(path)-1)
                a[:, :6] = np.column_stack([np.interp(ix, np.arange(len(path)), path[:, j]) for j in range(6)])
                expected = np.column_stack([np.interp(ix, np.arange(len(path)), poses[:, j, 3]) for j in range(3)])
                actual = k.fk(a[:, :6])
                if np.linalg.norm(actual[:, :3, 3]-expected, axis=1).max() > .0005:
                    raise PlanningError(name+':cartesian_residual')
                if Rotation.from_matrix(actual[:, :3, :3]@target[:3, :3].T).magnitude().max() > np.deg2rad(.25):
                    raise PlanningError(name+':angular_residual')
            elif joint is not None:
                a[:, :6] = smooth_joint(self.current[:6], joint, n)
            v = np.diff(np.vstack([self.current[:6], a[:, :6], a[-1, :6]]), axis=0)/.04
            acc = np.diff(np.vstack([np.zeros((1, 6)), v]), axis=0)/.04
            ratio = max(float((abs(v)/k.velocity).max()), float(np.sqrt(abs(acc).max()/20)))
            if ratio > 1.0001:
                n = int(np.ceil(n*ratio*1.02))
                continue
            break
        else:
            raise PlanningError(name+':retiming_failed')
        margin = np.minimum(a[:, :6]-k.limits[:, 0], k.limits[:, 1]-a[:, :6]).min()
        if margin < np.deg2rad(2):
            raise PlanningError(name+':joint_margin')
        self.margin = min(self.margin, float(margin))
        if grip is not None:
            a[:, 6] = smooth_joint(self.current[6:7], np.array([grip]), n)[:, 0]
        offset = sum(len(x) for x in self.chunks)
        if offset+n > self.budget:
            raise PlanningError(name+':step_budget')
        self.chunks.append(a)
        self.stages.append(dict(name=name, start=offset, steps=n))
        self.current = a[-1].copy()

    def finish(self):
        return dict(self.result, actions=np.concatenate(self.chunks), stages=self.stages,
                    min_margin_rad=self.margin, max_contact_jump_rad=self.jump)


def end(result, name):
    s = next(s for s in result['stages'] if s['name'] == name)
    return s['start']+s['steps']


def retreat(result, parameters, kin, cfg, cut, measured=None):
    b = Suffix(result, cut, kin, 1000)
    q = b.current.copy() if measured is None else measured.copy()
    # Release the old command's elastic deflection gradually before measured-start lifting.
    b.append('left_unload_to_measured', 15, joint=q[:6])
    start = kin['left_arm'].fk(q[None, :6])[0]
    detach = start.copy(); detach[2, 3] += .02
    high = detach.copy(); high[2, 3] = max(high[2, 3], parameters[8]+cfg.get('p5_clearance_height_m', .20))
    side = high.copy(); side[1, 3] += cfg.get('p5_side_exit_m', .10)
    b.append('left_detach', 20, target=detach)
    b.append('left_clearance_lift', 30, target=high)
    b.append('left_clearance_park', 30, target=side)
    b.append('left_park_settle', 10)
    out = b.finish()
    out['targets'] = dict(out['targets'], left_withdraw=side)
    out['p5_withdraw_start'] = start.tolist()
    out = append_close_tail(out, parameters, kin, 980, epsilon=cfg['close_epsilon_m'], skip_park=True)
    out['p5_runtime_replan_at'] = cut
    return out


def refine(base, parameters, kin, cfg):
    mode = cfg.get('p5_mode', 'baseline')
    if mode == 'p6':
        from .p6 import refine as refine_p6
        return refine_p6(base,parameters,kin,cfg)
    if mode not in ('baseline', 'placement', 'retreat', 'combined'):
        raise ValueError('Unknown P5 mode')
    if mode == 'baseline':
        return append_close_tail(base, parameters, kin, 1000, epsilon=cfg['close_epsilon_m'])
    result = base
    if mode in ('placement', 'combined'):
        geometry = dict(base['selected_geometry'])
        if 'p5_place_tilt_deg' in cfg:geometry['place_tilt'] = cfg['p5_place_tilt_deg']
        g = targets(parameters, kin['left_arm'].base, cfg['pull_distance_m'], **geometry)
        b = Suffix(base, end(base, 'left_lift'), kin, 1000)
        coarse = g['left_place'].copy(); coarse[2, 3] += .02
        poses = [g['left_hover']]
        for dest in (coarse,g['left_place'],g['left_withdraw']):
            start = poses[-1]
            n = max(2,int(np.ceil(np.linalg.norm(dest[:3,3]-start[:3,3])/.002)))
            for u in np.linspace(0,1,n+1)[1:]:
                t = dest.copy();t[:3,3] = start[:3,3]*(1-u)+dest[:3,3]*u;poses.append(t)
        path, margin = branch_path(kin['left_arm'], np.array(poses), b.current[:6], max_jump=np.deg2rad(10))
        b.margin = min(b.margin, margin)
        b.append('left_transport', 65, joint=path[0])
        b.append('left_lower_coarse', 35, target=coarse)
        b.append('left_lower', 25, target=g['left_place'])
        b.append('left_release', 20, grip=.05)
        b.append('release_settle', 20)
        if mode == 'placement':
            b.append('left_withdraw', 40, target=g['left_withdraw'])
            b.append('final_hold', 40)
        result = b.finish(); result['targets'] = g
        result['p5_geometry'] = geometry
    if mode in ('retreat', 'combined'):
        result = retreat(result, parameters, kin, cfg, end(result, 'release_settle'))
    else:
        result = append_close_tail(result, parameters, kin, 1000, epsilon=cfg['close_epsilon_m'])
    result['p5_mode'] = mode
    return result
