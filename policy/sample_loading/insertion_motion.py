"""Retimed Cartesian primitives; qualifications apply to complete zero-speed segments."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import interpolate_pose
from .insertion_geometry import rigid
from .planner import branch_path, PlanningError


def audit_commands(actions, start, kin, previous=None):
    actions, start = np.asarray(actions), np.asarray(start)
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise PlanningError('invalid_actions')
    if ((actions[:, [6, 13]] < -1e-8) | (actions[:, [6, 13]] > .05 + 1e-8)).any():
        raise PlanningError('jaw_range')
    result = {}
    for arm, sl in [('left_arm', slice(0, 6)), ('right_arm', slice(7, 13))]:
        k = kin[arm]
        margin = np.minimum(actions[:, sl] - k.limits[:, 0], k.limits[:, 1] - actions[:, sl]).min()
        if margin < np.deg2rad(2) - 1e-6:
            raise PlanningError('joint_margin')
        q = np.vstack([start, start, actions, actions[-1], actions[-1]])
        if previous is not None:
            q[0] = previous
        velocity = np.diff(q[:, sl], axis=0) / .04
        acceleration = np.diff(velocity, axis=0) / .04
        if (abs(velocity) > k.velocity + 1e-4).any() or abs(acceleration).max() > 20.0001:
            raise PlanningError('command_dynamics')
        result[arm] = dict(margin_deg=float(np.degrees(margin)),
                          peak_velocity=float(abs(velocity).max()), peak_acceleration=float(abs(acceleration).max()))
    return result


def primitive(kin, start, target, *, minimum_steps=4, maximum_steps=1000):
    k = kin['left_arm']
    origin = k.fk(np.asarray(start)[None, :6])[0]
    target = rigid(target)
    angle = Rotation.from_matrix(target[:3, :3] @ origin[:3, :3].T).magnitude()
    n = max(3, int(np.ceil(np.linalg.norm(target[:3, 3] - origin[:3, 3]) / .001)) + 1,
            int(np.ceil(angle / np.deg2rad(.5))) + 1)
    poses = interpolate_pose(origin, target, n)
    path, _ = branch_path(k, poses, np.asarray(start)[:6], max_jump=np.deg2rad(10))
    if np.max(abs(path[0] - start[:6])) > .001:
        raise PlanningError('branch_join')
    path[0] = start[:6]
    steps = max(minimum_steps, n)
    while steps <= maximum_steps:
        u = np.arange(1, steps + 1) / steps
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        index = blend * (n - 1)
        actions = np.tile(start, (steps, 1))
        actions[:, :6] = np.column_stack([np.interp(index, np.arange(n), path[:, j]) for j in range(6)])
        try:
            audit = audit_commands(actions, start, kin)
        except PlanningError as exc:
            if str(exc) != 'command_dynamics':
                raise
            steps = int(np.ceil(steps * 1.2)) + 1
            continue
        actual = k.fk(actions[:, :6])
        expected = interpolate_pose(origin, target, 2)
        position = origin[:3, 3] + blend[:, None] * (target[:3, 3] - origin[:3, 3])
        rotvec = Rotation.from_matrix(expected[1, :3, :3] @ expected[0, :3, :3].T).as_rotvec()
        rotation = Rotation.from_rotvec(blend[:, None] * rotvec).as_matrix() @ origin[:3, :3]
        p_error = np.linalg.norm(actual[:, :3, 3] - position, axis=1).max()
        r_error = Rotation.from_matrix(actual[:, :3, :3] @ rotation.transpose(0, 2, 1)).magnitude().max()
        if p_error > .0005 or r_error > np.deg2rad(.25):
            raise PlanningError('dense_fk_residual')
        return actions, dict(arms=audit, samples=steps, position_error_m=float(p_error), rotation_error_rad=float(r_error))
    raise PlanningError('retimed_budget')


def enqueue_budget(step, actions, finish_steps, *, search=False):
    reserve = max(125, finish_steps)
    if step < 0 or len(actions) < 1 or step + len(actions) + reserve > 1000:
        raise PlanningError('finish_reserve')
    if search and step + len(actions) > 800:
        raise PlanningError('search_deadline')
    return reserve
