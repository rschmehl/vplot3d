#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022


@author: rschmehl
"""

from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from vplot3d import orthogonal_proj, Vector, Point, save_svg

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

v1 = Vector(np.array((0, 0, 0)),np.array((1, 1, 1)),shape='Arrow1Mend',zorder=11,color='k')
v2 = Vector(np.array((0, 0, 0)),np.array((1, 1, 0)),shape='Arrow1Lend',zorder=12,color='r', alpha=0.2)
v3 = Vector(np.array((0, 0, 0)),np.array((0, 0, 0.2)),scale=3,zorder=13,color='b')
v4 = Vector(np.array((0.5, 0.5, 0)),np.array((0, 0, 1.2)),scale=0.5,zorder=20,color='r')

p1 = Point(np.array((0, 0, 0)),shape='Point1M',zorder=30,color='b')
p2 = Point(np.array((0.5, 0.5, 0)),shape='Point1M',zorder=30,color='r')
p3 = Point(np.array((0.5, 0.5, 0.6)),shape='Point1M',zorder=30,color='r')
p4 = Point(np.array((0.5, 0.5, 1)),shape='Point1M',zorder=30,color='r')

ax.set_axis_off()
save_svg('test.svg')
