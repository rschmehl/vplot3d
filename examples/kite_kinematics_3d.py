#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022


@author: rschmehl
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Set this with environment variable PYTHONPATH
lib_path = Path('/home/rschmehl/projects/vplot3d')
sys.path.append(str(lib_path))

import vplot3d as v3d
from vplot3d import Line, Vector, Point, Arc, ArcMeasure, Polygon, save_svg_tex

def spherical_vector_base(beta, phi):
    '''Spherical vector base.
    '''
    cb    = np.cos(beta)
    sb    = np.sin(beta)
    cp    = np.cos(phi)
    sp    = np.sin(phi)
    return np.array([ cb*cp,  cb*sp, sb]), \
           np.array([   -sp,     cp,  0]), \
           np.array([-sb*cp, -sb*sp, cb])

# Setup figure and axes3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_axis_off()

# Initialize vector diagram
# See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
v3d.plot_zoom, v3d.plot_radius = v3d.init(width=980, height=700,
                                          xmin=-1,   xmax=1,
                                          ymin=-1,   ymax=1.05,
                                          zmin=-0.3, zmax=1,
                                          zoom=2.65,
                                          elev=20,   azim=30     )

# Origin
PO = np.array([0, 0, 0])
O  = Point(PO, shape='Point1M', zorder=100, color='k')

# Cartesian base vectors
Px = np.array([1, 0, 0])
Py = np.array([0, 1, 0])
Pz = np.array([0, 0, 1])
e1 = Vector(PO, Px, shape='Arrow1Mend', linewidth=5, zorder=50, color='k')
e2 = Vector(PO, Py, shape='Arrow1Mend', linewidth=5, zorder=50, color='k')
e3 = Vector(PO, Pz, shape='Arrow1Mend', linewidth=5, zorder=50, color='k')

# Tether & kite
r    = 0.8
beta = np.deg2rad(30)
phi  = np.deg2rad(15)
Pk   = r*np.array([np.cos(phi)*np.cos(beta), np.sin(phi)*np.cos(beta), np.sin(beta)])
l1   = Line(PO, Pk, linewidth=2, linestyle="solid")
K    = Point(Pk, shape='Point1M', zorder=100, color='k')

# Wing
er, ephi, ebeta = spherical_vector_base(beta=np.deg2rad(30), phi=np.deg2rad(15))
chi   = np.deg2rad(75)
vkt   = np.cos(chi)*ebeta + np.sin(chi)*ephi
pg1 = Polygon.rotated(Pk, file=lib_path / 'data' / 'V3-planform.dat', e2=vkt, e3=er, zorder=52, facecolor='k', edgecolor='k', scale=4e-5, linewidth=1, alpha=0.1, edgecoloralpha=0.8)
pg2 = Polygon.rotated(Pk, file=lib_path / 'data' / 'V3-tubeframe.dat', e2=vkt, e3=er, zorder=52, facecolor='k', edgecolor='k', scale=4e-5, linewidth=5, alpha=0, edgecoloralpha=1)

# Velocity vectors
Vw = np.array([0.5, 0, 0])
vw = Vector(Pk, Vw, shape='Arrow1Mend', zorder=55, linewidth=5, color='r')
Vk = 0.8*vkt
vk = Vector(Pk, Vk, shape='Arrow1Mend', zorder=55, linewidth=5, color='r')

# Text labels
ax.annotate3D(r'$\vec{O}$', xyz=PO, xytext=(-0.5,-1.8))
ax.annotate3D(r'$\vec{K}$', xyz=PO+Pk, xytext=(0.6,0.7))
ax.annotate3D(r'$\xw$', xyz=Px, xytext=(-0.4,-1.2))
ax.annotate3D(r'$\yw$', xyz=Py, xytext=(-0.8,-1.5))
ax.annotate3D(r'$\zw$', xyz=Pz, xytext=(0.6,-0.9))
ax.annotate3D(r'$\vvk$', xyz=PO+Pk+Vk, xytext=(-1,-1.8))
ax.annotate3D(r'$\vvw$', xyz=PO+Pk+Vw, xytext=(0,-1.6))

save_svg_tex('kite_kinematics_3d')

###############################################################################
# Second plot

pg1.remove()
pg2.remove()

# Spherical coordinates
Px   = np.array([r, 0, 0])
Py   = np.array([0, r, 0])
Pz   = np.array([0, 0, r])
Pkz  = np.array([0,     0,     Pk[2]])
Pkxy = np.array([Pk[0], Pk[1], 0    ])
Pxy  = r*np.array([np.cos(phi), np.sin(phi), 0])

a1 = Arc(PO,  Px, -Px, -Py, r, linewidth=2, zorder=31, color='k', alpha=0.3, linestyle=(0,(6,6)))
a2 = Arc(PO,  Px,  Px,  Pz, r, linewidth=2, zorder=31, color='k', alpha=0.3, linestyle=(0,(6,6)))
a3 = Arc(Pkz, Px,  Px,  Pz, np.cos(beta)*r, linewidth=2, zorder=31, color='k', alpha=0.3, linestyle=(0,(6,6)))

Z    = Point(Pz, shape='Point1M', zorder=100, color='k')
l2   = Line(PO, Pxy, linewidth=2, linestyle="solid")
am1  = ArcMeasure(PO, Pxy, Pk, radius=r, linewidth=3, zorder=31, color='k')
am2  = ArcMeasure(PO, Px,  Pxy, radius=r, linewidth=3, zorder=31, color='k')

ax.annotate3D(r'$\beta$', xyz=PO+Pk, xytext=(0.1,-3))
ax.annotate3D(r'$\phi$', xyz=PO+Pxy, xytext=(-1.4,1))

save_svg_tex('kite_kinematics_3d_a')

###############################################################################
plt.close()


