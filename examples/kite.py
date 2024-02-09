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
#lib_path = Path('/home/rschmehl/projects/vplot3d')
#sys.path.append(str(lib_path))

import vplot3d as v3d
from kiteV3 import KiteV3
from vplot3d import Point, Line, Vector, save_svg_tex

def spherical_vector_base(beta, phi):
    '''Spherical vector base.
    '''
    cb, sb = np.cos(beta), np.sin(beta)
    cp, sp = np.cos(phi),  np.sin(phi)
    return np.array([ cb*cp,  cb*sp, sb]), \
           np.array([   -sp,     cp,  0]), \
           np.array([-sb*cp, -sb*sp, cb])

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_axis_off()

# Initialize vector diagram
# See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
v3d.plot_zoom, v3d.plot_radius = v3d.init(width=600, height=600, \
                                          xmin=0,    xmax=1,     \
                                          ymin=0,    ymax=1,     \
                                          zmin=-0.3, zmax=1.5,   \
                                          zoom=1.5,              \
                                          elev=20,   azim=30    )

# Origin
PO = np.array([0, 0, 0])
O  = Point(PO, shape='Point1M', zorder=100, color='k')

# Cartesian base vectors
Px = np.array([1, 0, 0])
Py = np.array([0, 1, 0])
Pz = np.array([0, 0, 1])
e1 = Vector(PO, Px, shape='Arrow1Mend', linewidth=3, zorder=50, color='k')
e2 = Vector(PO, Py, shape='Arrow1Mend', linewidth=3, zorder=50, color='k')
e3 = Vector(PO, Pz, shape='Arrow1Mend', linewidth=3, zorder=50, color='k')

# Spherical vector base
er, ephi, ebeta = spherical_vector_base(beta=np.deg2rad(30), phi=np.deg2rad(15))

# Tether
lt  = Line(PO, er, linewidth=1, linestyle="solid")

# Kite
chi = np.deg2rad(0)
vkt = np.cos(chi)*ebeta + np.sin(chi)*ephi
kv3 = KiteV3.rotated(PO, e2=vkt, e3=er, voff=er, scale=0.05)
# Rasterized output of meshes
#v3d.rasterize_dpi = 72
#kv3 = KiteV3.rotated(PO, e2=vkt, e3=er, voff=er, scale=0.05, rasterized=True)

# Text labels
ax.annotate3D(r'$\vec{O}$', xyz=PO, xytext=(-0.5,-1.8))
ax.annotate3D(r'$\xw$', xyz=Px, xytext=(-0.4,-1.2))
ax.annotate3D(r'$\yw$', xyz=Py, xytext=(-0.8,-1.5))
ax.annotate3D(r'$\zw$', xyz=Pz, xytext=(0.6,-0.9))

save_svg_tex('kite')   # Save svg, post-process and display
plt.close()
