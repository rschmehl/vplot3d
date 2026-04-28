#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 14:33:04 2026

Convention:
    
    - Points and vectors are arrays
    - Arrays with names ending in _rpb represent spherical coordinates (r, phi, beta) 
    - Names of vplot3d objects end with _obj 
    - Absolute points start with P (e.g. PO for the origin)
    - Vectors start with v (e.g. vex for the Cartesian unit vector in x-direction)

@author: rschmehl
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Folder with configuration and shared data
data_path   = Path.cwd().parent / 'data'
macros_path = Path.cwd().parent / 'tools'
#os.environ['CONF_PATH'] = str(dat_path)

from vplot3d.vplot3d import init_view, Line, Vector, Point, Arc, ArcMeasure, Polygon, Annotation3D, save_svg_tex

###############################################################################
# Theory
###############################################################################

def spherical_vector_base(phi, beta):
    '''Spherical vector base.
    '''
    cp    = np.cos(phi)
    sp    = np.sin(phi)
    cb    = np.cos(beta)
    sb    = np.sin(beta)
    return np.array([ cb*cp,  cb*sp, sb]), \
           np.array([   -sp,     cp,  0]), \
           np.array([-sb*cp, -sb*sp, cb])
           
def tangential_velocity_factor(phi, beta, chi, E, f):
    '''Tangential velocity factor for a massless kite.
    '''
    a = - np.sin(phi)*np.sin(chi) - np.cos(phi)*np.sin(beta)*np.cos(chi)
    b =   np.cos(phi)*np.cos(beta)
    return a + np.sqrt( a*a + b*b - 1 + E*E*(b - f)**2)
           
def pitch_rotation(ve1, ve2, ve3, alpha):
    '''Rotate around object x-axis
    '''
    ca    = np.cos(alpha)
    sa    = np.sin(alpha)
    return ca*ve1 - sa*ve3, ve2, sa*ve1 +ca*ve3

# Problem parameters
r       = 3              # Radial coordinate
beta    = np.deg2rad(37) # Elevation angle
phi     = np.deg2rad(20) # Azimuth angle
chi     = np.deg2rad(40) # Course angle
f       = 0.4            # Reeling factor
E       = 5              # Lift-to-drag ratio
vw      = 1.8            # Velocity scaling factor
Fa      = 3              # Force scaling factor

# Origin
vzero    = np.array([0, 0, 0])
PO       = vzero

# Cartesian unit vectors (wind reference frame)
vex      = np.array([1, 0, 0])
vey      = np.array([0, 1, 0])
vez      = np.array([0, 0, 1])

# Wind velocity
vvw      = vw*vex

###############################################################################
# Initialize plotting in 3D
###############################################################################

# Setup figure and axes3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho', computed_zorder=False)
ax.set_axis_off()

# Initialize vector diagram
# See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
# Dimetric projection
init_view(width=980, height=720, 
          xmin=-1.7, xmax=1.5, ymin=-1, ymax=0, zmin=-0.3, zmax=1.7,
          zoom=1, elev=26.565, azim=45)

# Wind reference frame
x_obj   = Line(PO, r*vex, linewidth=2, color='k', alpha=0.3)
y_obj   = Line(PO, r*vey, linewidth=2, color='k', alpha=0.3)
z_obj   = Line(PO, r*vez, linewidth=2, color='k', alpha=0.3)

# Origin
PO_obj  = Point(PO, shape='Point1M', zorder=100, color='k')

# Flight path loop
n        = 81
dphi     = np.deg2rad(0.5)
dchi     = np.deg2rad(0.25)
dr       = 0.005
p_phi    = phi
p_beta   = beta
p_r      = r
p_chi    = chi

for i in range(n):
    i_frame = n - i

    # Trigonometric coefficients
    sp      = np.sin(p_phi)
    cp      = np.cos(p_phi)
    sb      = np.sin(p_beta)
    cb      = np.cos(p_beta)
    sc      = np.sin(chi)
    cc      = np.cos(chi)
    tc      = np.tan(p_chi)
    
    # Kite state
    p_phi   = p_phi  - dphi
    p_beta  = p_beta - dphi*cb/tc
    p_r     = p_r    - dr
    p_chi   = p_chi  + dchi

    # Transformation spherical coordinates (r, phi, beta) to wind reference frame (x, y, z)
    rpb_to_xyz = np.array([
        [ cp*cb, -sp, -cp*sb ],
        [ sp*cb,  cp, -sp*sb ],
        [    sb,   0,     cb ]
        ])

    # Transformation course reference frame (r, chi, n) to wind reference frame (x, y, z)
    rcn_to_xyz = np.array([
        [ cp*cb, -sp*sc-cp*sb*cc,  sp*cc-cp*sb*sc ],
        [ sp*cb,  cp*sc-sp*sb*cc, -cp*cc-sp*sb*sc ],
        [    sb,           cb*cc,           cb*sc ]
        ])

    # Kite position
    Pk = np.dot( rpb_to_xyz, [ p_r, 0, 0 ] )

    # Spherical unit vectors 
    ver, vephi, vebeta = spherical_vector_base(p_phi, p_beta)

    # Kite velocity
    lam      = tangential_velocity_factor(p_phi, p_beta, p_chi, E, f)
    vvk_rcn  = vw*np.array([f, lam, 0])
    vvkr_rcn = vw*np.array([f, 0,   0])
    vvkt_rcn = vw*np.array([0, lam, 0])
    vvk      = np.dot( rcn_to_xyz, vvk_rcn )
    vvkr     = np.dot( rcn_to_xyz, vvkr_rcn )
    vvkt     = np.dot( rcn_to_xyz, vvkt_rcn )

    # Apparent wind velocity and components
    vva      = vvw - vvk
    va       = np.linalg.norm(vva)
    var      = np.dot( vva, ver )
    vvar     = var*ver
    vvat     = vva - vvar
    vat      = np.linalg.norm(vvat)

    # Aerodynamic forces
    vFa      = Fa*ver
    vD       = np.dot( vFa, vva/va )*vva/va
    vL       = vFa - vD
    L        = np.linalg.norm(vL)
    D        = np.linalg.norm(vD)

    # Aerodynamic reference frame
    veax     = vva/va                 # Pointing against apparent wind velocity
    vv       = np.cross(veax, vFa)
    veay     = -vv/np.linalg.norm(vv) # Pointing to right wing tip
    veaz     = np.cross(veax, veay)   # Pointing to origin

    # Tether
    t_obj   = Line(PO, Pk, linewidth=2, linestyle="solid")

    # Wing
    # Geometry from file: xy-data, with x-axis to right wing tip and y-axis to heading 
    pg1_obj = Polygon.rotated(Pk, file=data_path / 'kite_V3_planform.dat',
                              e1=veay, e2=-veax, facecolor='w', edgecolor='k', zorder=70,
                              scale=1.5e-4, linewidth=1, alpha=0.8, edgecoloralpha=0.8)
    pg2_obj = Polygon.rotated(Pk, file=data_path / 'kite_V3_tubeframe.dat',
                              e1=veay, e2=-veax, facecolor='k', edgecolor='k', zorder=70,
                              scale=1.5e-4, linewidth=4, alpha=0, edgecoloralpha=1)

    # Kite point
    K_obj   = Point(Pk, shape='Point1M', zorder=60, color='k')

    save_svg_tex('frame_' + str(i_frame), macro_file_path=dat_path / 'macros.tex')
    
    K_obj.remove()
    t_obj.remove()
    pg1_obj.remove()
    pg2_obj.remove()

plt.close()

# Next steps:
# 1. move generated files frame_*_tex.svg to folder input_frames/
# 2. svg2fbf -i input_frames/ -f flight_path.fbf.svg -s 30
# 3. When including the file flight_path.fbf.svg in other document, set width attribute explicitly
#
# More info: https://github.com/Emasoft/svg2fbf#quick-start