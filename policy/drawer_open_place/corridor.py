"""Static public mesh checks; no simulator state or seed-dependent geometry."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from .clearance import link_vertices
from .geometry import DRAWER_ROTATION
from .motion import PlanningError


@lru_cache(None)
def parts(directory, name):
    result, vertices = [], []
    for line in (Path(directory)/(name+'ACD.obj')).read_text().splitlines():
        if line.startswith('o ') and vertices:
            result.append(np.asarray(vertices)*[1,1,.35]); vertices=[]
        if line.startswith('v '):vertices.append(list(map(float,line.split()[1:4])))
    if vertices:result.append(np.asarray(vertices)*[1,1,.35])
    assert result and all(len(v)>=4 for v in result)
    return result


@lru_cache(None)
def tomato(source):
    path=Path(source)/'assets/tomato/tomato_asset_simready/asset_simready.obj'
    v=np.array([list(map(float,s.split()[1:4])) for s in path.read_text().splitlines() if s.startswith('v ')])*.8
    return v[ConvexHull(v).vertices]


def intersects(a,b,margin=.002):
    if np.any(a.max(0)+margin<b.min(0)) or np.any(b.max(0)+margin<a.min(0)):return False
    # Convex hulls conservatively contain non-convex robot links; drawer ACD parts stay separate.
    equations=np.concatenate([ConvexHull(a).equations,ConvexHull(b).equations])
    result=linprog(np.zeros(3),A_ub=equations[:,:3],b_ub=-equations[:,3]+margin/2,
                   bounds=[(None,None)]*3,method='highs')
    if result.status==2:return False
    if not result.success:raise RuntimeError('Convex intersection solver: '+result.message)
    return True


def collision_pairs(q,kin,parameters,cfg,*,opening=None,opening_uncertainty=.003):
    if opening is None:opening=cfg['pull_distance_m']
    if not 0<opening<.2:raise ValueError('Expected drawer opening within the static rail limits')
    left=link_vertices(q[:7],kin['left_arm'].base,cfg['asset_source'])
    right=link_vertices(q[7:],kin['right_arm'].base,cfg['asset_source'])
    origin=np.asarray(parameters[6:9]);obstacles={}
    for name in ['OuterBox','InnerBox']:
        for i,v in enumerate(parts(cfg['drawer_asset'],name)):
            if name=='InnerBox':
                # Cover the estimated opening interval without pretending commanded travel is ground truth.
                v=np.concatenate([v+[opening-opening_uncertainty,0,0],v+[opening+opening_uncertainty,0,0]])
            obstacles[f'{name}:{i}']=v@DRAWER_ROTATION.T+origin
    pairs=set()
    for name,a in left.items():
        for other,b in obstacles.items():
            if intersects(a,b):pairs.add('p6_drawer_collision:'+name+':'+other)
        for other,b in right.items():
            if intersects(a,b):pairs.add('p6_arm_collision:'+name+':'+other)
    return pairs


def check_pose(q,kin,parameters,cfg):
    pairs=collision_pairs(q,kin,parameters,cfg)
    if pairs:raise PlanningError(sorted(pairs)[0],dict(pairs=sorted(pairs)))
    return True


def check_path(actions,kin,parameters,cfg):
    # shortcut: check each control command, not continuous-time swept meshes; qualify with physical traces.
    for q in actions:check_pose(q,kin,parameters,cfg)
