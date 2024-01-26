#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:27:55 2024

Meshes and colors from my Three.js implementation of the V3 kite

@author: rschmehl
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from matplotlib.colors import LightSource

fig = plt.figure()
plt.rcParams['figure.figsize'] = [10, 10]
ax = fig.add_subplot(projection='3d', proj_type='ortho')

elev = 0   # default:  30
azim = 270   # default: -60
#ax.view_init(elev, azim)

mesh1 = trimesh.load('../data/kite_V3_canopy.obj')
mesh2 = trimesh.load('../data/kite_V3_LE.obj')
mesh3 = trimesh.load('../data/kite_V3_struts.obj')
mesh4 = trimesh.load('../data/kite_V3_KCU.obj')
#
# >>> Lines should be read into a

# Move KCU to correct position
mesh4.vertices[:,2] = mesh4.vertices[:,2] - 5.7
mesh4.vertices[:,1] = mesh4.vertices[:,1] + 0.2

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

# Cartesian unit vectors
ax.scatter(0, 0, 0, color='k')
ax.plot([0,1], [0,0], [0,0], color='r')
ax.plot([0,0], [0,1], [0,0], color='g')
ax.plot([0,0], [0,0], [0,1], color='b')

ax.set_xlim3d([-3, 3])
ax.set_ylim3d([-3, 3])
ax.set_zlim3d([-4, 2])
ax.set_box_aspect([6,6,6], zoom=1)
ax.set_axis_off()

plt.savefig("kite_V3.svg", format="svg", dpi=300)
#plt.savefig("kite_V3.png", format="png")
plt.show()

