#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022


@author: rschmehl
"""

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
from pathlib import Path
from itertools import product, combinations
from IPython.display import display, Image

# Set this with environment variable PYTHONPATH
lib_path = Path('/home/rschmehl/projects/vplot3d')
sys.path.append(str(lib_path))

import vplot3d as v3d
from kiteV3 import KiteV3
from vplot3d import save_svg

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')

# Initialize vector diagram
# See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
v3d.plot_zoom, v3d.plot_radius = v3d.init(width=600, height=600, \
                                          xmin=0,    xmax=1,     \
                                          ymin=0,    ymax=1,     \
                                          zmin=-0.3, zmax=1.5,   \
                                          zoom=1.5,              \
                                          elev=30,   azim=-60    )

# draw cube https://itecnote.com/tecnote/python-plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib/
# r = [0, 1]
# for s, e in combinations(np.array(list(product(r, r, r))), 2):
#     if np.sum(np.abs(s-e)) == r[1]-r[0]:
#         ax.plot3D(*zip(s, e), color="b")

PO = np.array([0, 0, 0])
kv3 = KiteV3(PO, scale=0.1)

ax.set_axis_off()

fname='kite'
save_svg(fname+'_tex.svg')
p=subprocess.call([lib_path / 'tex' / 'convert_tex.sh', fname+'_tex.svg'])
display(Image(filename=fname+'.png'))
plt.close()
