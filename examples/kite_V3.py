#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:27:55 2024

For proper depth sorting (z-order), all components of the rendered object need
to be contained in a single Poly3DCollection. Because depth sorting of polygons
is limited in accuracy, especially for long aspect ratio triangles, I apply
transparency to somewhat cover these depth sorting flaws.

An alternative approach would be to divide rendered surfaces into front and
back side, by using the orientation of face normals relative to the screen.
Then, manual sorting could be applied to the divided surface regions. But also
this approach is limited and works only for simple shapes.
https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html

Meshes and colors from my Three.js implementation of the V3 kite

@author: rschmehl
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import trimesh
from matplotlib.colors import LightSource
from transforms3d.axangles import axangle2mat

class LineSystem():
    '''Class for line system objects.
    '''

    def __init__(self, verts, segs):
        self.vertices = verts
        self.segments = segs

    @classmethod
    def load(cls, fn):
        """
        Read polylines from an obj-file. Adapted from
        https://github.com/zishun/MeshUtility/blob/main/meshutility/obj_lines_io.py

        Input
          fn: path to obj-file
        Output
          verts: vertex coordinates (n x 3)
          lines: polyline segments
        """
        verts = []
        segs = []
        with open(fn, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            s = lines[i].split()
            if (len(s) == 0):
                continue
            if (s[0] == 'v'):
                verts.append([float(s[1]), float(s[2]), float(s[3])])
            elif (s[0] == 'l'):
                L = list(map(lambda x: int(x),  s[1:]))
                segs.extend([[L[i], L[i+1]] for i in range(len(L)-1)])

        # segs are 1-indexed -> 0-indexed
        return cls(np.array(verts), np.array(segs)-1)

fig = plt.figure()
plt.rcParams['figure.figsize'] = [12, 12]
ax = fig.add_subplot(projection='3d', proj_type='ortho')

elev = 0   # default:  30
azim = 0   # default: -60
#ax.view_init(elev, azim)

mesh1 = trimesh.load('../data/kite_V3_canopy.obj')
mesh2 = trimesh.load('../data/kite_V3_LE.obj')
mesh3 = trimesh.load('../data/kite_V3_struts.obj')
mesh4 = trimesh.load('../data/kite_V3_KCU.obj')
mesh5 = LineSystem.load('../data/kite_V3_bridle.obj')

# Move KCU to correct position
mesh4.vertices[:,2] = mesh4.vertices[:,2] - 5.7
#mesh4.vertices[:,1] = mesh4.vertices[:,1] + 0.2

mesh  = trimesh.util.concatenate((mesh1, mesh2, mesh3, mesh4))
fc    = ['#dcdcdc4d']*len(mesh1.faces) + \
        ['#000000']*len(mesh2.faces) + \
        ['#000000']*len(mesh3.faces) + \
        ['#484848b3']*len(mesh4.faces)

nodes = mesh.vertices
faces = mesh.faces

ls = LightSource(azdeg=225.0, altdeg=40.0)

pc = Poly3DCollection([nodes[faces[i,:]] for i in range(len(faces))],
                       edgecolor=fc, facecolors=fc, \
                       linewidths=0, shade=True, lightsource=ls)

ax.add_collection3d(pc)

# Add bridle line system separately
nodes = mesh5.vertices
lines = mesh5.segments
m = axangle2mat([1, 0, 0], np.radians(90))
for i in range(len(nodes)):
    nodes[i] = m.dot(nodes[i])

lc = Line3DCollection([nodes[lines[i,:]] for i in range(len(lines))],
                       edgecolor='k', linewidths=1, linestyles='solid')

ax.add_collection3d(lc)

# Cartesian unit vectors
ax.scatter(0, 0, 0, color='k')
ax.plot([0,1], [0,0], [0,0], color='r')
ax.plot([0,0], [0,1], [0,0], color='g')
ax.plot([0,0], [0,0], [0,1], color='b')

ax.set_xlim3d([-3, 3])
ax.set_ylim3d([-3, 3])
ax.set_zlim3d([-4.5, 1.5])
ax.set_box_aspect([6,6,6], zoom=0.9)
ax.set_axis_off()

plt.savefig("kite_V3.svg", format="svg", dpi=300)
#plt.savefig("kite_V3.png", format="png")
plt.show()


