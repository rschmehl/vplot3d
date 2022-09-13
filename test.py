#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022


@author: rschmehl
"""

from mpl_toolkits.mplot3d import proj3d, art3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from itertools import product, combinations
from vplot3d import orthogonal_proj, Vector, Point, Arc, ArcMeasure, save_svg

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')


# Figure size
mpl.rcParams['figure.figsize'] = 10, 10

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')

#
# elevation and azimuth angle -> 0,0 gives yz-perspective of coordinate system
elev =  30
azim = -60
ax.view_init(elev, azim)
proj3d.persp_transformation = orthogonal_proj

# draw cube https://itecnote.com/tecnote/python-plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib/
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

v1 = Vector(np.array([0, 0, 0]),np.array([1, 1, 1]),shape='Arrow1Mend',zorder=11,color='k')
v2 = Vector(np.array([0, 0, 0]),np.array([1, 1, 0]),shape='Arrow1Lend',zorder=12,color='r', alpha=0.2)
v3 = Vector(np.array([0, 0, 0]),np.array([0, 0, 0.2]),scale=3,zorder=13,color='b')
v4 = Vector(np.array([0.5, 0.5, 0]),np.array([0, 0, 1.2]),scale=0.5,color='r')

p1 = Point(np.array([0, 0, 0]),shape='Point1M',zorder=30,color='b')
p2 = Point(np.array([0.5, 0.5, 0]),shape='Point1M',zorder=30,color='r')
p3 = Point(np.array([0.5, 0.5, 0.6]),shape='Point1M',color='r')
p4 = Point(np.array([0.25, 0.75, 0.9]),shape='Point1M',color='r')

a1 = Arc(np.array([0, 0, 0]),np.array([1, 0, 0]),np.array([0, 0, 1]),1,linewidth=2,color='b')
am = ArcMeasure(np.array([0, 0, 0]),np.array([1, 0, 0]),np.array([1, 1, 0]),1,linewidth=2,shape='Arrow1Mend',scale=0.707106781,zorder=31,color='k')

q1 = np.array([0.5, -0.5, 1.0])
q2 = np.array([0.8, -0.5, 0.9])
q3 = np.array([1.0, 1.0, 1.0])
q4 = np.array([0.4, 1.0, 1.0])

r1 = Point(q1,color='g',zorder=21)
r2 = Point(q2,color='g',zorder=21)
r3 = Point(q3,color='g',zorder=21)
r4 = Point(q4,color='g',zorder=21)

poly3d = [[q1, q2, q3, q4]]
ax.add_collection3d(art3d.Poly3DCollection(poly3d, facecolors='g', edgecolors='k', linewidths=1, alpha=0.95))

#pg = Polygon([[q1, q2, q3, q4]], )

ax.set_axis_off()
save_svg('test.svg')
