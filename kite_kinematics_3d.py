#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022


@author: rschmehl
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib_inline
import vplot3d as v3d
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from vplot3d import figsize, orthogonal_proj, Line, Vector, Point, Arc, ArcMeasure, Polygon, save_svg
import subprocess
from IPython.display import display, Image

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

mpl.rcParams['svg.fonttype']   = 'none'
mpl.rcParams['figure.figsize'] = figsize(980, 700)

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')

# Set anticipated diagram range in data space
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1.05])
ax.set_zlim3d([-0.2, 1])
v3d.ZOOM = 2.4

# Diagram perspective
elev = 20   # default:  30
azim = 30   # default: -60
ax.view_init(elev, azim)
proj3d.persp_transformation = orthogonal_proj

# Origin
PO = np.array([0, 0, 0])
O  = Point(PO, shape='Point1M', zorder=100, color='k')

# Cartesian base vectors
r  = 1
Px = np.array([r, 0, 0])
Py = np.array([0, r, 0])
Pz = np.array([0, 0, r])
e1 = Vector(PO, Px, shape='Arrow1Mend', linewidth=5, zorder=50, color='k')
e2 = Vector(PO, Py, shape='Arrow1Mend', linewidth=5, zorder=50, color='k')
e3 = Vector(PO, Pz, shape='Arrow1Mend', linewidth=5, zorder=50, color='k')

# draw cube https://itecnote.com/tecnote/python-plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib/
#rb = [0, r]
#for s, e in combinations(np.array(list(product(rb, rb, rb))), 2):
#    if np.sum(np.abs(s-e)) == rb[1]-rb[0]:
#        ax.plot3D(*zip(s, e), color="b")

# Tether
beta = np.deg2rad(30)
phi  = 0
Pk   = r*np.array([np.cos(beta), 0, np.sin(beta)])
Pkx  = r*np.array([0, 0, np.sin(beta)])
l1   = Line(PO, Pk, linewidth=2, linestyle="solid")
K    = Point(Pk, shape='Point1M', zorder=100, color='k')

# Arc
a1 = Arc(PO, Px, -Px, -Py, r, linewidth=2, zorder=31, color='k', alpha=0.3, linestyle=(0,(6,6)))

# Arc measure
am1 = ArcMeasure(PO, Px, Pk, 1, linewidth=3, shape='Arrow1Mend', scale=0.3, zorder=31, color='r')

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
pg1 = Polygon.rotated(Pk, file='planform.dat', e2=vkt, e3=er, voff=voff, zorder=52, facecolor='k', edgecolor='k', scale=4e-5, linewidth=1, alpha=0.1, edgecoloralpha=0.8)
pg2 = Polygon.rotated(Pk, file='tubeframe.dat', e2=vkt, e3=er, voff=voff, zorder=52, facecolor='k', edgecolor='k', scale=4e-5, linewidth=5, alpha=0, edgecoloralpha=1)

# Velocity vectors
Vw = np.array([0.5, 0, 0])
vw = Vector(Pk, Vw, shape='Arrow1Mend', zorder=55, linewidth=5, color='r')
Vk = 0.8*vkt
vk = Vector(Pk, Vk, shape='Arrow1Mend', zorder=55, linewidth=5, color='r')

# Text labels
ax.annotate3D(r'$\vec{O}$', xyz=PO, xytext=(-0.5,-1.8))
ax.annotate3D(r'$\vec{K}$', xyz=PO+Pk, xytext=(0.7,0.5))
ax.annotate3D(r'$\xw$', xyz=Px, xytext=(0,-1.2))
ax.annotate3D(r'$\yw$', xyz=Py, xytext=(-0.8,-1.5))
ax.annotate3D(r'$\zw$', xyz=Pz, xytext=(0.6,-0.9))
ax.annotate3D(r'$\vvk$', xyz=PO+Pk+Vk, xytext=(-1,-1.8))
ax.annotate3D(r'$\vvw$', xyz=PO+Pk+Vw, xytext=(0,-1.6))
ax.annotate3D(r'$\beta$', xyz=PO+0.3*Pk, xytext=(0,-2))

ax.set_axis_off()

fname='kite_kinematics_3d'
save_svg(fname+'_tex.svg')
p=subprocess.call(['convert_tex.sh', fname+'_tex.svg'])
display(Image(filename=fname+'.png'))

###############################################################################
# Second plot

a2 = Arc(PO, Px, Px, Pz, r, linewidth=2, zorder=31, color='k', alpha=0.3, linestyle=(0,(6,6)))
a3 = Arc(Pkx, Px, Px, Pz, np.cos(beta)*r, linewidth=2, zorder=31, color='k', alpha=0.3, linestyle=(0,(6,6)))

fname='kite_kinematics_3d_a'
save_svg(fname+'_tex.svg')
p=subprocess.call(['convert_tex.sh', fname+'_tex.svg'])
display(Image(filename=fname+'.png'))

###############################################################################
plt.close()


