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
from itertools import product, combinations
import vplot3d.vplot3d as vplot3d_file
from vplot3d.vplot3d import init_view, Polygon, save_svg_tex

# Set this with environment variable PYTHONPATH
lib_path = Path(vplot3d_file.__file__).parent
sys.path.append(str(lib_path))

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_axis_off()

# Initialize vector diagram
# See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
init_view(width=600, height=600,
          xmin=0, xmax=1, ymin=-0.5, ymax=0.5, zmin=-0.5, zmax=0.5,
          zoom=1.5, elev=90, azim=270)

# draw cube https://itecnote.com/tecnote/python-plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib/
r = [0, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# Polygon
p  = np.array([0, 0, 0])
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1])
voff = np.array([0, 0, 0])
pg = Polygon.rotated(p, file=lib_path / 'data' / 'kite_V3_airfoil.dat', e1=ex, e2=ey, voff=voff, facecolor='k', edgecolor='k', scale=1, linewidth=2, alpha=0.5, edgecoloralpha=0.8)

save_svg_tex('kite_V3_airfoil')
plt.close()
