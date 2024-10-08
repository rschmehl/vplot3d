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
import vplot3d.vplot3d as vplot3d_file
from vplot3d.vplot3d import init_view, Polygon, save_svg_tex

# Set this with environment variable PYTHONPATH
lib_path = Path(vplot3d_file.__file__).parent
sys.path.append(str(lib_path))
data_path = Path.cwd().parent / 'data'

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_axis_off()
#mpl.rcParams['font.family'] = "Open Sans"

# Initialize vector diagram
init_view(width=600, height=600,
          xmin=0, xmax=1, ymin=0.0, ymax=1.0, zmin=-0.5, zmax=0.5,
          zoom=1.5, elev=90, azim=270)

# Polygon
PO = np.array([0, 0, 0])
Px = np.array([1, 0, 0])
Py = np.array([0, 1, 0])
Pz = np.array([0, 0, 1])

# Shapes
voff = np.array([0, 0, 0])
a1 = Polygon.rotated(PO, file=data_path / 'clark_y_airfoil.dat',
                     e1=Px, e2=Py, voff=voff, edgecolor='k', facecolor='k',
                     scale=1, linewidth=2, alpha=0.1, edgecoloralpha=0.8)
ax.annotate3D('Clark Y airfoil', xyz=0.37*Px-0.08*Py)

voff = np.array([0, 0.2, 0])
a2 = Polygon.rotated(PO, file=data_path / 'kite_V3_airfoil.dat',
                     e1=Px, e2=Py, voff=voff, edgecolor='k', facecolor='k',
                     scale=1, linewidth=2, alpha=0.1, edgecoloralpha=0.8)
ax.annotate3D('TU Delft V3 kite  center airfoil', xyz=0.25*Px+0.15*Py)

voff = np.array([0.5, 0.6, 0])
a3_1 = Polygon.rotated(PO, file=data_path / 'kite_V3_tubeframe.dat',
                       e2=Py, e3=Pz, voff=voff, edgecolor='k', facecolor='k',
                       scale=1e-4, zorder=10, linewidth=10, alpha=0, edgecoloralpha=1)
a3_2 = Polygon.rotated(PO, file=data_path / 'kite_V3_planform.dat',
                       e2=Py, e3=Pz, voff=voff, edgecolor='k', facecolor='w',
                       scale=1e-4, zorder=0, linewidth=2, alpha=0.5, edgecoloralpha=0.8)
ax.annotate3D('TU Delft V3 kite  planform', xyz=0.27*Px+0.34*Py)

save_svg_tex('plot_2D_shapes', fontsize=14)
plt.close()
