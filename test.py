#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022


@author: rschmehl
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import vplot3d as v3d
import numpy as np
from mpl_toolkits.mplot3d import proj3d, art3d
from itertools import product, combinations
from vplot3d import orthogonal_proj, Annotation3D, Line, Vector, Point, Arc, ArcMeasure, Polygon, save_svg
import subprocess
from IPython.display import display, Image

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')


mpl.rcParams['svg.fonttype']   = 'none'
mpl.rcParams['figure.figsize'] = 10, 10

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')

# Rendering options
#v3d.RAW_MATH = True
v3d.FONTSIZE = 20

#
# elevation and azimuth angle -> 0,0 gives yz-perspective of coordinate system
elev =  90   # default:  30
azim = -90   # default: -60
ax.view_init(elev, azim)
proj3d.persp_transformation = orthogonal_proj

# draw cube https://itecnote.com/tecnote/python-plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib/
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# draw a line
l1 = Line(np.array([0, 1, 1]),np.array([0, -1, -1]),linewidth=1)

# draw some vectors
v1 = Vector(np.array([0, 0, 0]),np.array([1, 1, 0.5]),shape='Arrow1Mend',zorder=11,color='k')
v2 = Vector(np.array([0, 0, 0]),np.array([1, 1, 0]),shape='Arrow1Lend',zorder=12,color='r', alpha=0.2)
v3 = Vector(np.array([0, 0, 0]),np.array([0, 0, 0.2]),scale=3,zorder=13,color='b',alpha=0.2)
v4 = Vector(np.array([0.5, 0.5, 0]),np.array([0, 0, 1.2]),scale=0.5,color='r')

# draw some points
p1 = Point(np.array([0, 0, 0]),shape='Point1M',zorder=30,color='b')
p2 = Point(np.array([0.5, 0.5, 0]),shape='Point1M',zorder=30,color='r')
p3 = Point(np.array([0.5, 0.5, 0.6]),shape='Point1M',color='r')
p4 = Point(np.array([0.25, 0.75, 0.9]),shape='Point1M',color='r')

p5 = Point(np.array([0.25, 0.65, 0.75]),shape='CoG',color='r')

# draw some arcs and arc measures
a1  = Arc(np.array([0, 0, 0]),np.array([1, 0, 0]),np.array([0, 0, 1]),1,linewidth=2,color='b')
am1 = ArcMeasure(np.array([0, 0, 0]),np.array([1, 0, 0]),np.array([1, 1, 0]),1,linewidth=2,shape='Arrow1Mend',scale=0.707106781,zorder=31,color='k')
am2 = ArcMeasure(np.array([0, 1, 1]),np.array([0, -1, 0]),np.array([0, -1, -1]),0.5,linewidth=2,shape='Arrow1Mend',scale=1,zorder=31,color='r')

# draw a polygon
p  = np.array([0, 0, 0])
q1 = np.array([0.5, -0.5, 1.0])
q2 = np.array([0.8, -0.5, 0.9])
q3 = np.array([1.0, 1.0, 1.0])
q4 = np.array([0.4, 1.0, 1.0])
u1 = np.array([1, 1, 0])
u2 = np.array([0, 1, 1])
u3 = np.array([1, 1, 1])
#print(poly3d)

r1 = Point(q1,color='g',zorder=21)
r2 = Point(q2,color='g',zorder=21)
r3 = Point(q3,color='g',zorder=21)
r4 = Point(q4,color='g',zorder=21)
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1])

# Polygon
pg1 = Polygon(p, [[q1, q2, q3, q4]], facecolor='g', edgecolor='k', scale=1, linewidth=1, alpha=0.5)
pg2 = Polygon.rotated(np.array([0, 0.3, 0.2]), v=[[u1, u2, u3]], e1=ex, e3=ez, facecolor='r', edgecolor='k', scale=0.5, linewidth=1, alpha=0.3)
voff = np.array([-0.25, 0, 0])
pg3 = Polygon.rotated(p, file='clarky.dat', e1=ex, e3=ez, voff=voff, facecolor='k', edgecolor='k', scale=1, linewidth=1, alpha=0.1, edgecoloralpha=0.8)

# add some text
t1 = Annotation3D('$P_1$',  xyz=q3, xytext=(10,10))
t2 = Annotation3D('$\phi$', xyz=q4)

ax.set_axis_off()
save_svg('test.svg')
plt.close()

# Use Inkscape to convert the SVG into a PNG file
p=subprocess.call(['/usr/bin/inkscape', 'test.svg', '--export-type=png', '--export-filename=test.png'])
display(Image(filename='test.png'))