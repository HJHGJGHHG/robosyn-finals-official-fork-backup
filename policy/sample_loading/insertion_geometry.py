"""Pure estimated-geometry candidates for the V28 bounded insertion suffix."""
from copy import deepcopy
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

AXIAL_DEGREES = (0, -10, 10, -20, 20)
SEARCH_OFFSETS_M = .001 * np.array([(0, 0), (1, 0), (1, -1), (0, -1),
                                   (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)])


def rigid(value):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=1e-5)):
        raise ValueError('invalid_rigid_transform')
    return a


def clearance(tube_world, rack_world, bounds, plate_z):
    """Conservative tube AABB transformed into rack coordinates, in meters."""
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError('invalid_tube_bounds')
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
    transform = np.linalg.inv(rigid(rack_world)) @ rigid(tube_world)
    return float((corners @ transform[:3, :3].T + transform[:3, 3])[:, 2].min() - plate_z)


def axial_candidates(geometry, rack_world, current_eef, fixed):
    """Return geometry only; caller must qualify dense motion and contacts."""
    rack, eef = rigid(rack_world), rigid(current_eef)
    grip, place = rigid(geometry['gl']), rigid(geometry['place'])
    goal = place @ grip
    n = rack[:3, 2]
    current_gap = clearance(eef @ grip, rack, fixed['tube_bounds_m'], fixed['reference_plate_z_m'])
    lift = max(0., .012 - current_gap)
    if lift > .030 + 1e-9:
        raise ValueError('clearance_recovery_exceeds_30mm')
    for angle in AXIAL_DEGREES:
        turn = np.eye(4); turn[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
        target_tube = goal @ turn
        target = target_tube @ np.linalg.inv(grip)
        g = deepcopy(geometry)
        for key in ('near', 'place', 'withdraw'):
            t = target.copy(); t[:3, 3] += rigid(geometry[key])[:3, 3] - place[:3, 3]
            g[key] = t
        g['gl'] = grip.copy(); g['tube_goal'] = target_tube
        lifted = eef.copy(); lifted[:3, 3] += lift * n
        gap_at_goal = clearance(target_tube, rack, fixed['tube_bounds_m'], fixed['reference_plate_z_m'])
        high = target.copy(); high[:3, 3] += max(0., .012 - gap_at_goal) * n
        assert np.allclose(target_tube[:3, 2], goal[:3, 2], atol=1e-6)
        assert np.allclose(target_tube[:3, 3], goal[:3, 3], atol=1e-6)
        yield {'angle_deg': angle, 'geometry': g, 'lift_pose': lifted,
               'align_high_pose': high, 'extra_lift_m': lift,
               'current_conservative_clearance_m': current_gap,
               'requires_full_path_and_collision_validation': True}
