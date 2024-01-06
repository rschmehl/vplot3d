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
from vplot3d import figsize, orthogonal_proj, Annotation3D, Line, Vector, Point, Arc, ArcMeasure, Polygon, save_svg
import subprocess
from IPython.display import display, Image

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')


mpl.rcParams['svg.fonttype']   = 'none'
mpl.rcParams['figure.figsize'] = figsize(400, 400)

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')

# Rendering options
#v3d.RAW_MATH = True
v3d.FONTSIZE = 20
v3d.ZOOM = 1.5

#
# elevation and azimuth angle -> 0,0 gives yz-perspective of coordinate system
elev = 20   # default:  30
azim = 30   # default: -60
ax.view_init(elev, azim)
#ax.dist = 6                # default=10, https://stackoverflow.com/a/42350761
proj3d.persp_transformation = orthogonal_proj

# Origin
PO = np.array([0, 0, 0])
O  = Point(PO,shape='Point1M',zorder=51,color='k')

# Cartesian base vectors
r  = 1
Px = np.array([r, 0, 0])
Py = np.array([0, r, 0])
Pz = np.array([0, 0, r])
e1 = Vector(PO, Px, shape='Arrow1Mend', zorder=50, color='k')
e2 = Vector(PO, Py, shape='Arrow1Mend', zorder=50, color='k')
e3 = Vector(PO, Pz, shape='Arrow1Mend', zorder=50, color='k')

# draw cube https://itecnote.com/tecnote/python-plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib/
rb = [0, r]
for s, e in combinations(np.array(list(product(rb, rb, rb))), 2):
    if np.sum(np.abs(s-e)) == rb[1]-rb[0]:
        ax.plot3D(*zip(s, e), color="b")

# Tether
beta = np.deg2rad(30)
phi  = 0
Pk = r*np.array([np.cos(beta), 0, np.sin(beta)])
l1 = Line(PO, Pk, linewidth=2, linestyle="solid")
K  = Point(Pk,shape='Point1M',scale=0.5,zorder=60,color='k',bgcolor='k')

# Arc measure
am1 = ArcMeasure(PO, Px, Pk, 1, linewidth=2, shape='Arrow1Mend', scale=0.4, zorder=31, color='k')

# Wing
voff  = np.array([0, 0, 0])
chi   = np.deg2rad(75)
cb    = np.cos(beta)
sb    = np.sin(beta)
cp    = np.cos(phi)
sp    = np.sin(phi)
cc    = np.cos(chi)
sc    = np.sin(chi)
er    = np.array([ cb*cp,  cb*sp, sb])
ephi  = np.array([   -sp,     cp,  0])
ebeta = np.array([-sb*cp, -sb*sp, cb])
vkt   = cc*ebeta + sc*ephi
pg1 = Polygon.rotated(Pk, file='planform.dat', e2=vkt, e3=er, voff=voff, facecolor='k', edgecolor='k', scale=0.00005, linewidth=1, alpha=0.1, edgecoloralpha=0.8)
pg2 = Polygon.rotated(Pk, file='tubeframe.dat', e2=vkt, e3=er, voff=voff, facecolor='k', edgecolor='k', scale=0.00005, linewidth=3, alpha=0, edgecoloralpha=1)

# Velocity vectors
Vw = np.array([0.5, 0, 0])
vw = Vector(Pk, Vw, shape='Arrow1Mend', zorder=55, linewidth=2, color='r')
Vk = 0.8*vkt
vk = Vector(Pk, Vk, shape='Arrow1Mend', zorder=55, linewidth=2, color='r')

ax.set_axis_off()
save_svg('planform.svg')
plt.close()

# Use Inkscape to convert the SVG into a PNG file
p=subprocess.call(['/usr/bin/inkscape', 'planform.svg', '--export-type=png', '--export-filename=planform.png'])
display(Image(filename='planform.png'))