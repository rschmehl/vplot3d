#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programmatic creation of the frames for a frame-by-frame SVG animation. 

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
macros_path = Path.cwd().parent / 'data'
#os.environ['CONF_PATH'] = str(dat_path)

# Create a directory to store generated SVG files
path = 'flight_path'
Path(path).mkdir(parents=False, exist_ok=True)

# Animation file, width and height of generated SVF file
afile  = 'flight_path.fbf.svg'
width  = 980
height = 720

from vplot3d.vplot3d import init_view, Line, Vector, Point, Arc, ArcMeasure, Polygon, Annotation3D, save_svg_tex, save_svg2fbf

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
    
    See Eqs. (2.20), (2.21) and (2.22) in: 
    
    Schmehl, R., Noom, M., Vlugt, R. van der: Traction Power Generation with Tethered Wings. 
    In: Ahrens, U., Diehl, M., Schmehl, R. (eds.) Airborne Wind Energy, Green Energy and Technology, 
    Chap. 2, pp. 23–45. Springer, Berlin Heidelberg (2013). doi: 10.1007/978-3-642-39965-7_2
    
    Other than in the book chapter, the formulation used here employs the elevation angle beta
    and not the polar angle theta = 90° - beta.
    '''
    a = - np.sin(phi)*np.sin(chi) - np.cos(phi)*np.sin(beta)*np.cos(chi)
    b =   np.cos(phi)*np.cos(beta)
    return a + np.sqrt( a*a + b*b - 1 + E*E*(b - f)**2)
           
def pitch_rotation(ve1, ve2, ve3, alpha):
    '''Rotate around object y-axis (ve2 vector)
    
    Input
      ve1:   base vector x-axis (array)
      ve2:   base vector y-axis (array)
      ve3:   base vector y-axis (array)
      alpha: rotation angle
      
    Output
      pitch_rotation: rotated vector base
    '''
    ca    = np.cos(alpha)
    sa    = np.sin(alpha)
    return ca*ve1 + sa*ve3, ve2, -sa*ve1 + ca*ve3

# Problem parameters
r       = 3              # Radial coordinate
beta    = np.deg2rad(37) # Elevation angle
phi     = np.deg2rad(20) # Azimuth angle
chi     = np.deg2rad(40) # Course angle
alpha   = np.deg2rad(7)  # Wing angle of attack
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
init_view(width=width, height=height, 
          xmin=-1.7, xmax=1.5, ymin=-1, ymax=0, zmin=-0.3, zmax=1.7,
          zoom=1, elev=26.565, azim=45)

# Wind reference frame
x_obj   = Line(PO, r*vex, linewidth=2, color='k', alpha=0.3)
y_obj   = Line(PO, r*vey, linewidth=2, color='k', alpha=0.3)
z_obj   = Line(PO, r*vez, linewidth=2, color='k', alpha=0.3)

# Origin
PO_obj  = Point(PO, shape='Point1M', zorder=100, color='k')

# Flight path loop
n          = 81
p_betas    = np.zeros(n)
p_xyz      = np.zeros((3,n))
dphi       = np.deg2rad(0.5)
dchi       = np.deg2rad(0.25)
dr         = 0.005
p_phi      = phi
p_beta     = beta
p_r        = r
p_chi      = chi

# Iterate downwards
for i in range(n): # starts with i=0

    # Store beta value
    p_betas[n-1-i] = p_beta
    
    # Trigonometric coefficients
    cb      = np.cos(p_beta)
    tc      = np.tan(p_chi)
    dbeta   = dphi*cb/tc

    # Kite state
    p_phi         = p_phi  - dphi
    p_beta        = p_beta - dbeta
    p_r           = p_r    - dr
    p_chi         = p_chi  + dchi
    beta          = beta - dbeta

# Iterate back upwards
for i in range(n):
    i_frame = i+1
    
    # Kite state
    p_phi   = p_phi  + dphi
    p_beta  = p_betas[i]
    p_r     = p_r    + dr
    p_chi   = p_chi  - dchi       
    
    # Trigonometric coefficients
    sp      = np.sin(p_phi)
    cp      = np.cos(p_phi)
    sb      = np.sin(p_beta)
    cb      = np.cos(p_beta)
    sc      = np.sin(p_chi)
    cc      = np.cos(p_chi)
    tc      = np.tan(p_chi)

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
    veax     = - vva/va               # Pointing against apparent wind velocity
    vv       = np.cross(veax, vFa)
    veay     = -vv/np.linalg.norm(vv) # Pointing to right wing tip
    veaz     = np.cross(veax, veay)   # Orthogonal to both
    
    # Wing reference frame (rotation by alpha around veay)
    vekx, veky, vekz = pitch_rotation(veax, veay, veaz, alpha)
    
    # Kite trail
    p_xyz[:,i] = Pk
    p_obj,     = plt.plot(p_xyz[0, :i], p_xyz[1, :i], p_xyz[2, :i], linewidth=10, color="#3185cfff", alpha=0.25)

    # Tether
    t_obj   = Line(PO, Pk, linewidth=2, linestyle="solid")

    # Wing
    # Geometry from file: xy-data, with x-axis to right wing tip and y-axis to heading 
    pg1_obj = Polygon.rotated(Pk, file=data_path / 'kite_V3_planform.dat',
                              e1=veky, e2=vekx, facecolor='w', edgecolor='k', zorder=70,
                              scale=1.5e-4, linewidth=1, alpha=0.8, edgecoloralpha=0.8)
    pg2_obj = Polygon.rotated(Pk, file=data_path / 'kite_V3_tubeframe.dat',
                              e1=veky, e2=vekx, facecolor='k', edgecolor='k', zorder=70,
                              scale=1.5e-4, linewidth=4, alpha=0, edgecoloralpha=1)

    # Kite attachment point
    Kt_obj  = Point(Pk, scale=0.5, shape='Point1M', zorder=0, color='k', bgcolor='k')

    # Save SVG file
    file    = 'frame_' + str(i_frame)
    save_svg_tex(file, macro_file_path=macros_path / 'macros.tex')
    
    # Move generated file to folder
    fn = Path(file+'_tex.svg')
    fn.rename(Path(path) / fn)
  
    # Remove excess files
    Path(file+'.svg').unlink()
    Path(file+'.png').unlink()
    
    Kt_obj.remove()
    t_obj.remove()
    pg1_obj.remove()
    pg2_obj.remove()    
    p_obj.remove()

#print('End state:')
#print('r    = '+str(p_r))
#print('phi  = '+str(np.rad2deg(p_phi)))
#print('beta = '+str(np.rad2deg(p_beta)))

plt.close()

###############################################################################
# Create fbf.svg animation
###############################################################################
# atype: animation type, see https://github.com/Emasoft/svg2fbf#animation-types
save_svg2fbf(file=afile, path=path, width=width, height=height, atype='loop', fps=30) 


