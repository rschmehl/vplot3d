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
macros_path = Path.cwd().parent / 'data'
#os.environ['CONF_PATH'] = str(dat_path)

# Create a directory to store generated SVG files
path = 'view_rotation'
Path(path).mkdir(parents=False, exist_ok=True)

# Animation file, width and height of generated SVF file
afile  = 'view_rotation.fbf.svg'
width  = 980
height = 720

# Perspective start and end values
xmin   = [ -1.7,  1.0 ]
xmax   = [  1.5,  4.2 ] 
ymin   = [ -1.0, -1.0 ] 
ymax   = [  0.0,  0.0 ] 
zmin   = [ -0.3,  1.7 ] 
zmax   = [  1.7,  3.6 ]
zoom   = [  1.0,  1.0 ]

from vplot3d.vplot3d import init_view, Line, Vector, Point, Arc, Polygon, Annotation3D, save_svg_tex, save_svg2fbf, wipe_canvas

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
           
def perspective_angles(vn):
    '''Perspective angles azim and elev from viewing normal vector.
    '''
    nxy = np.sqrt( vn[0]*vn[0] + vn[1]*vn[1] )
    azim = np.rad2deg(np.acos( vn[0]/nxy ))
    elev = np.rad2deg(np.asin( vn[2] ))
    return azim, elev
           
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

def rotate_vector(v, k, gamma):
    """
    Rotates a vector v around an axis normal vector k by angle theta (radians).
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Statement
    """
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    return v*cg + np.cross(k,v)*sg + k*np.dot(k,v)*(1-cg)

def flight_path(phi, beta, r, chi):
    '''Generate a flight path
    '''
    n        = 41
    p_xyz    = np.array( [[0.]*n]*3 )
    dphi     = np.deg2rad(1)
    dchi     = np.deg2rad(0.5)
    dr       = 0.01
    p_phi    = phi
    p_beta   = beta
    p_r      = r
    p_chi    = chi
    for i in range(n):
        sp     = np.sin(p_phi)
        cp     = np.cos(p_phi)
        sb     = np.sin(p_beta)
        cb     = np.cos(p_beta)
        tc     = np.tan(p_chi)
        p_phi  = p_phi  - dphi
        p_beta = p_beta - dphi*cb/tc
        p_r    = p_r    - dr
        p_chi  = p_chi  + dchi
        M = np.array([
            [ cp*cb, -sp, -cp*sb ],
            [ sp*cb,  cp, -sp*sb ],
            [    sb,   0,     cb ]
            ])
        p_xyz[:,i] = np.dot( M, [ p_r, 0, 0 ] )
    return p_xyz

# Problem parameters
r       = 3              # Radial coordinate
beta    = np.deg2rad(37) # Elevation angle
phi     = np.deg2rad(20) # Azimuth angle
chi     = np.deg2rad(40) # Course angle
alpha   = np.deg2rad(7)  # Wing angle of attack
f       = 0.4            # Reeling factor
E       = 5              # Lift-to-drag ratio
vw      = 1.8            # Velocity scaling factor
Fa      = 3         # 6  # Force scaling factor
elev    = 26.565         # Elevation angle perspective
azim    = 45             # Azimuth angle perspective
ngamma  = 81             # Angle discretization

# Trigonometric coefficients
sp      = np.sin(phi)
cp      = np.cos(phi)
sb      = np.sin(beta)
cb      = np.cos(beta)
sc      = np.sin(chi)
cc      = np.cos(chi)

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

# Origin
vzero    = np.array([0, 0, 0])
PO       = vzero

# Cartesian unit vectors (wind reference frame)
vex      = np.array([1, 0, 0])
vey      = np.array([0, 1, 0])
vez      = np.array([0, 0, 1])

# Tether & kite
Pk       = r*np.array([cp*cb, sp*cb, sb])

# Kite trail
p_xyz = flight_path(phi, beta, r, chi)

# Wind velocity
vvw      = vw*vex

# Spherical unit vectors 
ver, vephi, vebeta = spherical_vector_base(phi, beta)

# Kite velocity
lam      = tangential_velocity_factor(phi, beta, chi, E, f)
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

# ver-vva-plane normal vector vn_n & viewing angles azim_n and elev_n
vv       = np.cross( ver, vva )
vn_n     = vv/np.sqrt( np.sum( vv*vv ) )
azim_n, elev_n = perspective_angles(vn_n)

# Initial view normal vector, from viewing angles
sa      = np.sin(np.deg2rad(azim))
ca      = np.cos(np.deg2rad(azim))
se      = np.sin(np.deg2rad(elev))
ce      = np.cos(np.deg2rad(elev))
vn_0    = np.array([ ca*ce, sa*ce, se ])

# Rotation axis
vp      = np.cross( vn_0, vn_n )
vpabs   = np.sqrt( np.sum( vp*vp ) )

# Rotation axis unit vector
vpn     = vp/vpabs

# Angle between the two normal vectors
gamma   = np.asin( vpabs )
dgamma  = gamma/(ngamma - 1)

###############################################################################
# Initialize plotting in 3D
###############################################################################

# Setup figure and axes3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho', computed_zorder=False)
ax.set_axis_off()
    
# Iterate from initial to final view
for i in range(ngamma): # starts with i=0
    i_frame = i+1
 
    # Rotate view
    gamma_i = dgamma * i
    vn_i = rotate_vector(vn_0, vpn, gamma_i)
    azim_i, elev_i = perspective_angles(vn_i)

    # Interpolated values
    xmin_i = xmin[0] + (xmin[1] - xmin[0]) * i/(ngamma - 1)
    xmax_i = xmax[0] + (xmax[1] - xmax[0]) * i/(ngamma - 1)
    ymin_i = ymin[0] + (ymin[1] - ymin[0]) * i/(ngamma - 1)
    ymax_i = ymax[0] + (ymax[1] - ymax[0]) * i/(ngamma - 1)
    zmin_i = zmin[0] + (zmin[1] - zmin[0]) * i/(ngamma - 1)
    zmax_i = zmax[0] + (zmax[1] - zmax[0]) * i/(ngamma - 1)
    zoom_i = zoom[0] + (zoom[1] - zoom[0]) * i/(ngamma - 1)

    # Initialize vector diagram
    # See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
    # Dimetric projection
    init_view(width=width, height=height, 
              xmin=xmin_i, xmax=xmax_i, ymin=ymin_i, ymax=ymax_i, zmin=zmin_i, zmax=zmax_i,
              zoom=zoom_i, elev=elev_i, azim=azim_i)
    
    ###############################################################################
    # Plot layer 01
    ###############################################################################
    
    # Wind reference frame
    x_obj   = Line(PO, r*vex, linewidth=2, color='k', alpha=0.3)
    y_obj   = Line(PO, r*vey, linewidth=2, color='k', alpha=0.3)
    z_obj   = Line(PO, r*vez, linewidth=2, color='k', alpha=0.3)
    
    # Origin
    PO_obj  = Point(PO, shape='Point1M', zorder=100, color='k')
    
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
    
    # Kite trail
    trail_obj, = plt.plot(p_xyz[0, :], p_xyz[1, :], p_xyz[2, :], linewidth=10, color="#3185cfff", alpha=0.25)
    
    # Kite point
    PK_obj  = Point(Pk, shape='Point1M', zorder=60, color='k')
    
    # Spherical frame
    PZ      = np.array([0, 0, r])
    PZ_obj  = Point(PZ, shape='Point1M', zorder=100, color='k')
    m1_obj  = Arc(PO,  vex, -vex, -vey, r, linewidth=2, color='k', alpha=0.3, linestyle=(0,(2,2)))
    m2_obj  = Arc(PO,  vey, -vey,  vex, r, linewidth=2, color='k', alpha=0.3, linestyle=(0,(2,2)))
    eq_obj  = Arc(PO,  vex,  vex,  vez, r, linewidth=2, color='k', alpha=0.3, linestyle=(0,(2,2)))
    
    # Spherical cross hair
    vepz    = cp*vex + sp*vey
    m3_obj  = Arc(PO,  vepz, vez, radius=r, linewidth=2, color='k', alpha=0.3, linestyle=(0,(2,2)))
    l1_obj  = Arc(PO+r*sb*vez,  r*cb*vex, r*cb*vex, vez, radius=r*cb,  linewidth=2, color='k', alpha=0.3, linestyle=(0,(2,2)))
    
    # Apparent wind velocity
    vva_obj    = Vector(Pk, vva, shape='Arrow1Mend', linewidth=4, zorder=50, color='r')
    
    # Aerodynamic forces
    vvar_obj   = Vector(Pk, vvar, shape='Arrow1Mend', linewidth=4, zorder=50, color='r')
    vvat_obj   = Vector(Pk, vvat, shape='Arrow1Mend', linewidth=4, zorder=50, color='r')
    vFa_obj    = Vector(Pk, vFa, shape='Arrow1Mend', linewidth=4, zorder=80, color='b')
    vD_obj     = Vector(Pk, vD, shape='Arrow1Mend', linewidth=4, zorder=50, color='b')
    vL_obj     = Vector(Pk, vL, shape='Arrow1Mend', linewidth=4, zorder=80, color='b')
    lvvatr_obj = Polygon(Pk, [[vzero, vvat, vva, vvar, vzero]], edgecolor='r', facecolor='w', linewidth=2, alpha=0.4,  edgecoloralpha=0.8, linestyle=(0,(2,2)), zorder=20)
    vt_obj     = Polygon(Pk, [[vzero, vvat, vva, vzero]], edgecolor='w', facecolor='r', linewidth=2, alpha=0.1,  edgecoloralpha=0.2, zorder=30)
    lvFa_obj   = Polygon(Pk, [[vD, vFa, vL, vzero]], edgecolor='b', facecolor='w', linewidth=2, alpha=0.4,  edgecoloralpha=0.4, linestyle=(0,(2,2)), zorder=20)
    ft_obj     = Polygon(Pk, [[vL, vFa, vzero]], edgecolor='w', facecolor='b', linewidth=2, alpha=0.1,  edgecoloralpha=0.2, zorder=20)

    # Save SVG file
    file       = 'frame_' + str(i_frame)
    save_svg_tex(file, macro_file_path=macros_path / 'macros.tex')

    # Move generated file to folder
    fn = Path(file+'_tex.svg')
    fn.rename(Path(path) / fn)
  
    # Remove excess files
    Path(file+'.svg').unlink()
    Path(file+'.png').unlink()
    
    trail_obj.remove()
    wipe_canvas()

plt.close()

###############################################################################
# Create fbf.svg animation
###############################################################################
# atype: animation type, see https://github.com/Emasoft/svg2fbf#animation-types
save_svg2fbf(file=afile, path=path, width=width, height=height, atype='pingpong_loop', fps=30) 


