#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022

3D vector diagrams with SVG output. 

- Vectors and points use Inkscape-compatible markers (seems to be unique)

@author: rschmehl
"""

from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import io
import xml.etree.ElementTree as ET
import textwrap

# Lists for vectors and markers
vectors = []
markers = []

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-0.0001,zback]])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
    https://stackoverflow.com/a/31364297

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    # get current Axes instance
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    #
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    #
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    #
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    #
    return plot_radius
    
def projected_length(beta_deg, phi_deg, vec):
    '''Project a vector from 3D data space into the viewing plane and 
    return its length. Implemented is here the transformation from 
    Cartesian coordinates (3D data space) into spherical coordinates, 
    that can be regarded as the observers reference frame after 
    reassigning the spherical vectorbase:
     e_r     -> e_x,observer (orthogonal to projection plane/display)
     e_phi   -> e_y,observer (horizontal display axis)
    -e_theta -> e_z,observer (vertical display axis)
    See https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Input
      beta_deg: elevation angle in degrees
      phi_deg : azimuth angle in degrees
      vec     : 3D vector
    Output
      projected_length: projected length
    '''
    beta =np.deg2rad(beta_deg)
    phi  =np.deg2rad(phi_deg)
    lx = np.cos(beta)*np.cos(phi)*vec[0]+np.cos(beta)*np.sin(phi)*vec[1]+np.sin(beta)*vec[2]
    ly =             -np.sin(phi)*vec[0]             +np.cos(phi)*vec[1]
    lz =-np.sin(beta)*np.cos(phi)*vec[0]-np.sin(beta)*np.sin(phi)*vec[1]+np.cos(beta)*vec[2]
    return np.sqrt(ly*ly + lz*lz)

class Vector:
    '''Class for vector objects.
    Opacity (alpha) is implemented on the level of the line path (not marker path) and is inherited to the marker.
    '''
    def __init__(self, p=[0, 0, 0], v=[1, 1, 1], id=None, shape='Arrow1Mend', zorder=0, color='k', alpha=1):
        #
        # get current Axes instance
        ax = plt.gca()
        # base point coordinates
        self.p = p
        # vector coordinates
        self.v = v
        # name identifier
        self.id = id
        # arrow head shape
        self.shape = shape
        # arrow color
        self.color = color
        # arrow transparency
        self.alpha = alpha
        # arrow head style
        self.style = shape + '_' + color
        # group id assigned when plotting the line
        self.gid = None
        # depth ordering
        self.zorder = zorder
        # marker id composed of path + color
        marker_id = shape + '-' + color
        # add it to the list of markers if it is not yet included
        if marker_id not in [m.id for m in markers]:
            markers.append(Marker(shape, color))
                # set unique gid
        self.gid = 'vector_' + str(len(vectors)+1)   
        # calculate vector end point
        q = p + v
        # plot the shortened line
        line, = ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], zorder=self.zorder, linewidth=3, solid_capstyle='butt', color=self.color, alpha=self.alpha)
        line.set_gid(self.gid)
        self.line = line     
        # add new vector to the list of vectors
        vectors.append(self)
        
    def adjust_length(self, plot_radius, elev, azim):
        '''
        Shorten the vector line such that the marker tip coincides with the actual vector end point
        '''
        v = self.v
        p = self.p
        # unit vector along original vector
        e = v/np.sqrt(v.dot(v))
        # offset for the used marker 
        delta = Marker.deltas[self.shape]
        # length of unit vector projected into display 
        l = projected_length(elev, azim, e)
        # corrected endpoint  
        q = p + v - e*2*plot_radius*delta/l
        # reset vector endpoint
        self.line.set_data_3d([p[0], q[0]], [p[1], q[1]], [p[2], q[2]])
        
        
class Marker:
    '''Class for marker objects. 
    Because color is not inherited from the object to which the marker is attached, we generate
    a marker for each color and embed the color in the marker's path.
    '''
    
    # Dictionary of marker paths, follows Inkscape default markers
    paths = {
        'Arrow1Mend' : ';stroke-width:1pt;opacity:1;" d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z" transform="scale(0.4) rotate(180) translate(10,0)',
        'Arrow1Lend' : ';stroke-width:1pt;opacity:1;" d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z" transform="scale(0.8) rotate(180) translate(10,0)'
    }
    # These offset values need to be determined empirically for each marker path 
    deltas = { 
        'Arrow1Mend' : 0.019,
        'Arrow1Lend' : 0.038
    }
       
    def __init__(self, shape=None, color='k', style='overflow:visible', refX=0, refY=0, orient='auto'):
        # shape
        self.shape = shape
        # color
        self.color = color
        # marker id composed of path + color
        self.id = shape + '-' + color
        # marker style
        self.style = style
        # refX and refY
        self.refX = str(refX)
        self.refY = str(refY)
        # orientation
        self.orient = orient
        # svg code
        hexcolor = mpl.colors.to_hex(color)
        marker_start = '<marker id="' + self.id + '" style="' + self.style + '" refX="' + self.refX + '" refY="' + self.refY + '" orient="' + self.orient + '">'
        marker_path  = '   <path style="stroke:' + hexcolor + ';fill:' + hexcolor + Marker.paths[shape] + '"/>'
        marker_end   = '</marker>'
        self.svg = marker_start + '\n' + marker_path + '\n' + marker_end

    # add a method to expand the paths dictionary on the fly
    
    
def save_svg(file='unnamed.svg'):
    '''Function for modifying the generated SVG code.
    
    Inspired by https://matplotlib.org/stable/gallery/misc/svg_filter_line.html and other sources.
    
    Before saving the drawing, vector lines are shortened. See also https://stackoverflow.com/a/50797203

    '''
    #
    # get current Axes instance and prepare axes for plotting
    ax = plt.gca()
    plot_radius = set_axes_equal(ax)
    print('plot_radius=' + str(plot_radius))    
    ax.set_box_aspect([1,1,1]) # requires matplotlib 3.3.0
    
    # Shorten all vectors lines such that the marker tip coincides with the actual vector end point
    for vec in vectors:
        vec.adjust_length(plot_radius, ax.elev, ax.azim) 
    
    # save the figure as a byte string in SVG format
    f = io.BytesIO()
    plt.savefig(f, format="svg")

    # read in the saved SVG and define the SVG namespace
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace("", ns)    
    tree, xmlid = ET.XMLID(f.getvalue())

    # remove the defs element with matplotlib's default css style
    for defs in tree.findall('{'+ns+'}defs'):
        if defs.findall('{'+ns+'}style'):
            tree.remove(defs)

    # create the defs section with previously generated marker elements
    defs = '<defs>' + '\n'
    for m in markers:
        defs =  defs + textwrap.indent(m.svg, 2*' ') + '\n'
    defs = defs + '</defs>'
    
    # insert the marker definition in the svg dom tree.
    tree.insert(0, ET.XML(defs))
    
    # process all vectors
    for v in vectors:
        velement  = xmlid[v.gid]
        marker_id = v.shape + '-' + v.color
        velement.set('marker-end', 'url(#' + marker_id + ')')
        # Convert "stroke-opacity" to just "opacity" to also get a transparent marker
        style = velement.find('.//{'+ns+'}path').attrib['style'].replace('stroke-opacity', 'opacity')
        velement.find('.//{'+ns+'}path').attrib['style'] = style

    # process all points
    
    print(f"Saving '{file}'")
    ET.ElementTree(tree).write(file, encoding="utf-8")
    
