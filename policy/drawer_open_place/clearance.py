"""Conservative full-link AABBs against the inflated drawer sweep, after parking."""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import DRAWER_ROTATION
from .motion import PlanningError


@lru_cache(None)
def model(source):
    directory=Path(source)/'assets/CobotMagicArm';tree=ET.parse(directory/'CobotMagicWithGripperV100.urdf').getroot()
    meshes={}
    for link in tree.findall('link'):
        collision=link.find('collision')
        if collision is None:continue
        mesh=collision.find('geometry/mesh')
        if mesh is None:continue
        path=directory/'Collision'/Path(mesh.attrib['filename']).name
        vertices=np.array([list(map(float,line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')])
        origin=collision.find('origin');xyz=np.array(list(map(float,origin.attrib.get('xyz','0 0 0').split()))) if origin is not None else np.zeros(3)
        rpy=list(map(float,origin.attrib.get('rpy','0 0 0').split())) if origin is not None else [0,0,0]
        meshes[link.attrib['name']]=vertices@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz
    return tree.findall('joint'),meshes


def link_vertices(q,base,source):
    joints,meshes=model(source);frames={'arm_base':base};remaining=list(joints)
    for _ in range(len(joints)):
        changed=False
        for joint in remaining[:]:
            parent=joint.find('parent').attrib['link']
            if parent not in frames:continue
            origin=joint.find('origin');t=np.eye(4)
            if origin is not None:
                t[:3,3]=list(map(float,origin.attrib.get('xyz','0 0 0').split()))
                t[:3,:3]=Rotation.from_euler('xyz',list(map(float,origin.attrib.get('rpy','0 0 0').split()))).as_matrix()
            name=joint.attrib['name'];motion=np.eye(4)
            if joint.attrib['type']!='fixed':
                idx=int(name.replace('joint',''))-1;value=q[min(idx,6)];axis=np.array(list(map(float,joint.find('axis').attrib['xyz'].split())))
                if joint.attrib['type']=='prismatic':motion[:3,3]=value*axis
                else:motion[:3,:3]=Rotation.from_rotvec(value*axis).as_matrix()
            frames[joint.find('child').attrib['link']]=frames[parent]@t@motion;remaining.remove(joint);changed=True
        if not changed:break
    clouds={}
    for name,vertices in meshes.items():
        if name not in frames:continue
        t=frames[name];clouds[name]=vertices@t[:3,:3].T+t[:3,3]
    return clouds

def check(q,base,drawer,source):
    low=np.array([-.13,-.13,-.03]);high=np.array([.347,.13,.10])
    for name,world in link_vertices(q,base,source).items():
        local=(world-drawer)@DRAWER_ROTATION
        if ((local.max(0)>=low)&(local.min(0)<=high)).all():raise PlanningError('left_clearance_swept_aabb:'+name)
    return True
