"""Explicit estimated geometry only; meters, world frame, no simulator access."""
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
LABELS = ('tomato_anchor_x', 'tomato_anchor_y', 'tomato_anchor_z',
          'handle_x', 'handle_y', 'handle_z', 'drawer_x', 'drawer_y', 'drawer_z')
DRAWER_ROTATION = Rotation.from_euler('xyz', [.3, 0, 180], degrees=True).as_matrix()


def pose(rotation, position):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, position
    return t


def encode(tomato_pose, handle_pose, drawer_pose):
    """Training-only 4x4 world poses -> nine labels in meters.

    Anchor is the expert's object-local (0, .01, .035) point before TCP offsets.
    handle_pose is handle_xpos, drawer_pose is outer_box, after the same reset.
    """
    t, h, d = [np.asarray(x, float) for x in (tomato_pose, handle_pose, drawer_pose)]
    if any(x.shape != (4, 4) or not np.isfinite(x).all() for x in (t, h, d)):
        raise ValueError('Expected finite 4x4 world poses')
    return np.r_[t[:3, 3] + t[:3, :3] @ [0, .01, .035], h[:3, 3], d[:3, 3]]


def targets(parameters, left_base, pull_distance=.19, *, pregrasp_distance=.10, right_tilt=-30, place_tilt=-70):
    """Nine visual estimates -> Cartesian targets for fixed-orientation simulation.

    pull_distance is commanded travel, not measured drawer opening. .19 m leaves
    10 mm to the URDF's .2 m limit; physical full-opening needs qualification.
    """
    x = np.asarray(parameters, float)
    if x.shape != (9,) or not np.isfinite(x).all():
        raise ValueError('Expected nine finite position estimates')
    if not 0 < pull_distance < .2:
        raise ValueError('Pull travel must be inside the slide joint limits')
    anchor, handle, drawer = x.reshape(3, 3)
    direction = DRAWER_ROTATION[:, 0]
    right_r = Rotation.from_euler('YZX', [90, -90, right_tilt], degrees=True).as_matrix()
    grasp_r = Rotation.from_euler('z', np.arctan2(*(anchor-left_base[:3, 3])[1::-1])).as_matrix()
    grasp_r = grasp_r @ Rotation.from_euler('YX', [-60, 180], degrees=True).as_matrix()
    grasp_p = anchor + .02 * grasp_r[:, 2] - [0, 0, .01]
    # shortcut: nominal tomato center-to-anchor offset; qualify yaw/slip sensitivity before real deployment.
    nominal_center = anchor + [0, .007, -.01]
    center_in_tool = grasp_r.T @ (nominal_center-grasp_p)
    # Mesh floor is z=.02 before the official .35 Z scale. Release 5 mm above support.
    center_goal = drawer + DRAWER_ROTATION @ [pull_distance, 0, .02*.35+.020744+.005]
    place_r = Rotation.from_euler('z',np.arctan2(*(center_goal-left_base[:3,3])[1::-1])).as_matrix()
    place_r = place_r @ Rotation.from_euler('YX',[place_tilt,180],degrees=True).as_matrix()
    place_p = center_goal-place_r @ center_in_tool
    hover_p = place_p.copy()
    hover_p[2] = max(grasp_p[2]+.16, drawer[2]+.16, place_p[2]+.10)
    lift_p = grasp_p.copy(); lift_p[2] = hover_p[2]
    result = {
        'right_pre': pose(right_r, handle+.10*direction),
        'right_contact': pose(right_r, handle),
        'right_pull': pose(right_r, handle+pull_distance*direction),
        'left_pre': pose(grasp_r, grasp_p-pregrasp_distance*grasp_r[:, 2]),
        'left_grasp': pose(grasp_r, grasp_p),
        'left_lift': pose(grasp_r, lift_p),
        'left_hover': pose(place_r, hover_p),
        'left_place': pose(place_r, place_p),
        'left_withdraw': pose(place_r, place_p+[0, 0, .12]),
    }
    return result
