"""Task geometry. Inputs are explicit estimates; no environment/GT access."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
LABELS = ('grasp_x','grasp_y','grasp_z','rack_x','rack_y','rack_z',
          'sin_tube_heading','cos_tube_heading','sin_rack_yaw','cos_rack_yaw')


def transform(rotation=None, position=None):
    t = np.eye(4)
    if rotation is not None: t[:3,:3] = rotation
    if position is not None: t[:3,3] = position
    return t


def rotate(t, axis, degrees):
    return t @ transform(Rotation.from_euler(axis, degrees, degrees=True).as_matrix())


def offset(t, vector, local=False):
    v = transform(position=vector)
    return t @ v if local else v @ t


def align(t, axis, direction):
    t = t.copy(); a = t[:3,axis]; b = np.asarray(direction,dtype=float)
    b /= np.linalg.norm(b); cross = np.cross(a,b); norm = np.linalg.norm(cross)
    dot = np.clip(a @ b, -1, 1)
    if norm > 1e-8:
        t[:3,:3] = Rotation.from_rotvec(cross/norm*np.arccos(dot)).as_matrix() @ t[:3,:3]
    elif dot < 0:
        unit = np.eye(3)[np.argmin(abs(a))]; c = np.cross(a, unit); c /= np.linalg.norm(c)
        t[:3,:3] = Rotation.from_rotvec(c*np.pi).as_matrix() @ t[:3,:3]
    return t


def encode(tube, rack):
    tube, rack = np.asarray(tube).reshape(4,4), np.asarray(rack).reshape(4,4)
    heading = np.arctan2(tube[1,2], tube[0,2])
    yaw = np.arctan2(rack[1,0], rack[0,0])
    return np.r_[tube[:3,3]+.025*tube[:3,2], rack[:3,3],
                 np.sin(heading),np.cos(heading),np.sin(yaw),np.cos(yaw)]


def decode(label):
    x = np.asarray(label, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all(): raise ValueError('Invalid 10D label')
    if min(np.linalg.norm(x[6:8]),np.linalg.norm(x[8:10])) < .1: raise ValueError('Invalid angle pairs')
    heading, yaw = np.arctan2(x[6],x[7]), np.arctan2(x[8],x[9])
    rot = Rotation.from_euler('z',heading).as_matrix() @ Rotation.from_euler('y',np.pi/2).as_matrix()
    return transform(rot,x[:3]-.025*rot[:,2]), transform(Rotation.from_euler('z',yaw).as_matrix(),x[3:6])


def right_targets(tube, rack, right_base, profile=None):
    profile = profile or {}
    grasp = offset(tube,[0,0,.025],local=True)
    grasp = align(grasp,0,[0,0,-1]); grasp = align(grasp,1,[0,0,-1])
    grasp = rotate(rotate(grasp,'y',90),'z',180)
    grasp = offset(align(grasp,2,[0,0,-1]),[0,0,-.0044])
    grasp = rotate(grasp,'y',profile.get('grasp_tilt_deg',0))
    pre = rotate(offset(grasp,[0,0,-.05],local=True),'y',20)
    proof = rotate(offset(grasp,[0,0,.03]),'y',profile.get('proof_tilt_deg',0))
    up = rotate(offset(grasp,[0,0,.20]),'y',profile.get('up_tilt_deg',30))
    direction = rack[:3,3]-right_base[:3,3]
    hand = transform(Rotation.from_euler('z',np.arctan2(direction[1],direction[0])).as_matrix(),up[:3,3])
    hand = rotate(rotate(rotate(hand,'z',20),'y',-90),'x',180)
    hand = offset(hand,[.05,.10,.15])
    return dict(grasp=grasp,pre=pre,proof=proof,up=up,hand=hand)


def candidate_geometry(tube, rack, right_base, geometry, shift=.11, yaw=0, delta=(0,0,0), hand_yaw=0, retreat_mode='local'):
    rt = right_targets(tube,rack,right_base,geometry.get("right_profile"))
    gr = np.linalg.inv(rt['grasp']) @ tube
    pivot = (rt['hand']@gr)[:3,3]
    h = offset(rotate(transform(),'z',hand_yaw),pivot) @ transform(position=-pivot)
    h = offset(h,delta); rt['hand'] = h @ rt['hand']
    tube_h = rt['hand'] @ gr
    left = offset(rt['hand'],[-.06,0,.01],local=True)
    left = offset(rotate(left,'x',-90),[0,0,.015],local=True)
    left = offset(left,[shift,0,0],local=True)
    # Center the approach on the predicted tube axis before closing the jaws.
    # The old construction leaves the tube 14.4 mm off the jaw midplane,
    # then incorrectly assumes that offset survives closure and handover.
    # Preserve the selected axial contact height and 15 mm approach depth.
    contact_axial = float((left[:3,3]-tube_h[:3,3]) @ tube_h[:3,2])
    left,pre,left_proof = centered_takeover(tube_h,left,contact_axial,depth=.015)
    gl = np.linalg.inv(left) @ tube_h
    # Offline median rack-local XY prediction residual over all ten frozen
    # epoch78 development episodes: (+1.82, +2.01) mm. Correct only the
    # insertion reference; retain the physical hole geometry and raw decode.
    # These fixed constants do not observe the current object or grasp state.
    insertion_rack = offset(rack,[*geometry.get('prediction_bias_xy_m',[-.00182,-.00201]),0],local=True)
    goal = insertion_rack @ transform(Rotation.from_euler('z',yaw,degrees=True).as_matrix(),
        [*geometry['hole_center_xy_m'],geometry['controlled_release_tube_origin_z_m']])
    place = goal @ np.linalg.inv(gl)
    depth = geometry['reference_plate_z_m']+.006 - (geometry['support_local_z_m']+.0015)
    near = offset(place, rack[:3,2]*depth)
    above = offset(near, rack[:3,2]*.014)
    # Move along the right tool's negative approach axis; full mesh check decides safety.
    retreat = offset(rt['hand'],[0,0,-.08],local=True)
    if retreat_mode=='toward_base':
        direction=right_base[:3,3]-rt['hand'][:3,3];direction[2]=0
        retreat=offset(rt['hand'],direction/np.linalg.norm(direction)*.08)
    elif retreat_mode=='world_negative_y':retreat=offset(rt['hand'],[0,-.08,0])
    return dict(right=rt,left=left,left_pre=pre,left_proof=left_proof,
                place=place,near=near,above=above,withdraw=offset(place,rack[:3,2]*.08),
                right_retreat=retreat,gr=gr,gl=gl,tube_goal=goal,tube_h=tube_h,
                params=dict(shift=shift,yaw=yaw,delta=list(delta),hand_yaw=hand_yaw,retreat_mode=retreat_mode,right_profile=geometry.get('right_profile',{})))


def interpolate_pose(a,b,n):
    from scipy.spatial.transform import Slerp
    u=np.linspace(0,1,n)
    rots=Slerp([0,1],Rotation.from_matrix(np.array([a[:3,:3],b[:3,:3]])))(u).as_matrix()
    result=np.tile(np.eye(4),(n,1,1));result[:,:3,:3]=rots
    result[:,:3,3]=a[:3,3]+u[:,None]*(b[:3,3]-a[:3,3])
    return result


def centered_takeover(tube, reference_left, contact_axial, roll_deg=0, depth=.015):
    """Center the *contact point* in the jaws, including tilted held tubes."""
    left=rotate(reference_left,'x',roll_deg)
    contact=tube[:3,3]+tube[:3,2]*contact_axial
    left[:3,3]=contact+left[:3,2]*depth
    pre=offset(rotate(left,'x',10),[0,0,-.08],local=True)
    return left,pre,offset(left,[0,0,.01])
