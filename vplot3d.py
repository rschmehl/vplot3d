#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022

Programmatically creating 3D vector diagrams for SVG output. The following diagram objects can be used

- points
- lines and circular arcs
- vectors and arcmeasures

Points, vectors and arcmeasures use SVG markers.

@author: rschmehl
"""

from abc import ABC, abstractmethod
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import io
import xml.etree.ElementTree as ET
import textwrap

# Default values
ORIGIN    = np.array([0, 0, 0])
EX        = np.array([1, 0, 0])
EY        = np.array([0, 1, 0])
EZ        = np.array([0, 0, 1])
EXYZ      = np.array([1, 1, 1])
LINEWIDTH = 3
DEGREES   = np.arange(0, 361, 1)
COS       = np.cos(np.radians(DEGREES))
SIN       = np.sin(np.radians(DEGREES))

# Lists for geometrical objects
lines       = []
vectors     = []
arcs        = []
arcmeasures = []
points      = []
markers     = []

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

class Object3D(ABC):
    '''Abstract class for 3D objects.
    '''
    @abstractmethod
    def __init__(self, p=ORIGIN, id=None, linewidth=LINEWIDTH, scale=1, zorder=0, color='k', alpha=1):
        #
        # get current Axes instance
        self.ax = plt.gca()
        # reference point coordinates
        self.p = p
        # name identifier
        self.id = id
        # linewidth
        self.linewidth=linewidth
        # scaling factor wrt reference point p
        self.scale = scale
        # depth order
        self.zorder = zorder
        # object color
        self.color = color
        # object transparency
        self.alpha = alpha
        # object group id
        self.gid = None

    def add_marker(self, shape, edgecolor, facecolor):
        '''Add a marker to the object.
        Input
          shape     : type of marker to be added
          edgecolor : stroke color of marker
          facecolor : fill color of marker
        '''
        # arrow head shape
        self.shape = shape
        # arrow head style
        self.style = shape + '-' + edgecolor
        # add it to the list of markers if it is not yet included
        if self.style not in [m.id for m in markers]:
            # For arrowheads we use the same stroke and fill color (i.e. edge and face color)
            markers.append(Marker(shape, self.style, edgecolor=edgecolor, facecolor=facecolor))

class Point(Object3D):
    '''Class for point objects.
    '''
    def __init__(self, p=ORIGIN, id=None, linewidth=LINEWIDTH, shape='Point1M', zorder=0, color='k', alpha=1):
        #
        super().__init__(p, id, linewidth, 1, zorder, color, alpha)
        super().add_marker(shape, color, 'w')

        # set unique gid
        self.gid = 'point_' + str(len(points)+1)
        # plot a point where the marker is placed later
        line, = self.ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], p[2]], zorder=self.zorder, linewidth=self.linewidth, solid_capstyle='butt', color=self.color, alpha=self.alpha)
        line.set_gid(self.gid)
        self.line = line
        # add new point to the list of points
        points.append(self)

class Line(Object3D):
    '''Class for line objects.
    '''
    def __init__(self, p=ORIGIN, v=EXYZ, id=None, linewidth=LINEWIDTH, scale=1, zorder=0, color='k', alpha=1):

        super().__init__(p, id, linewidth, scale, zorder, color, alpha)

        # set unique gid
        self.gid = 'line_' + str(len(lines)+1)
        # line segment coordinates
        self.v = v*scale
        # calculate line end point
        q = p + v

        # plot the line
        line, = self.ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], zorder=self.zorder, linewidth=self.linewidth, solid_capstyle='butt', color=self.color, alpha=self.alpha)
        line.set_gid(self.gid)
        self.line = line
        # add new vector to the list of vectors
        lines.append(self)

class Vector(Line):
    '''Class for vector objects.
    '''
    def __init__(self, p=ORIGIN, v=EXYZ, id=None, linewidth=LINEWIDTH, shape='Arrow1Mend', scale=1, zorder=0, color='k', alpha=1):

        super().__init__(p, v, id, linewidth, scale, zorder, color, alpha)
        super().add_marker(shape, color, color)

        # set unique gid
        self.gid = 'vector_' + str(len(vectors)+1)
        self.line.set_gid(self.gid)

        # remove line from the list of lines
        lines.pop()

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
        delta = Marker.deltas[self.shape]*self.linewidth
        # length of unit vector projected into display
        l = projected_length(elev, azim, e)
        # corrected endpoint
        q = p + v - e*2*plot_radius*delta/l
        # reset vector endpoint
        self.line.set_data_3d([p[0], q[0]], [p[1], q[1]], [p[2], q[2]])

class Arc(Object3D):
    '''Class for circular arc objects.
    '''
    def __init__(self, p=ORIGIN, v1=EX, v2=EY, radius=1, id=None, linewidth=LINEWIDTH, scale=1, zorder=0, color='k', alpha=1):
        '''e1, e2 and e3 are spanning a local vector base
        '''
        super().__init__(p, id, linewidth, 1, zorder, color, alpha)

        # set unique gid
        self.gid = 'arc_' + str(len(arcs)+1)
        # normalized vectors spanning the arc
        self.e1 = e1 = v1 = v1/np.sqrt(np.dot(v1,v1))
        self.v2 = v2 =      v2/np.sqrt(np.dot(v2,v2))
        # angle between the two vectors
        self.angle = angle = np.degrees(np.arccos(np.dot(e1,v2)))
        # scaled radius
        self.radius = radius = radius*scale
        # normal vector
        n  = np.cross(e1,v2)
        # normalized normal vector
        e3 = n/np.sqrt(np.dot(n,n))
        # e1, e2 and e3 form a vectorbase
        e2 = np.cross(e3,e1)
        # find end index
        ip = np.argmax(DEGREES>angle)
        # array with coordinates of discretization points
        r = e1[:,np.newaxis]*COS[np.newaxis,:ip] + e2[:,np.newaxis]*SIN[np.newaxis,:ip]
        self.r = r

        # plot the line
        arc, = self.ax.plot(r[0,:], r[1,:], r[2,:], zorder=self.zorder, linewidth=self.linewidth, solid_capstyle='butt', color=self.color, alpha=self.alpha)
        arc.set_gid(self.gid)
        self.arc = arc
        # add new arc to the list of arcs
        arcs.append(self)

class ArcMeasure(Arc):
    '''Class for circular arc measure objects.
    '''
    def __init__(self, p=ORIGIN, v1=EX, v2=EY, radius=1, id=None, linewidth=LINEWIDTH, shape='Arrow1Mend', scale=1, zorder=0, color='k', alpha=1):
        '''e1, e2 and e3 are spanning a local vector base
        '''

        super().__init__(p, v1, v2, radius, id, linewidth, scale, zorder, color, alpha)
        super().add_marker(shape, color, color)

        # set unique gid
        self.gid = 'arcmeasure_' + str(len(arcmeasures)+1)
        self.arc.set_gid(self.gid)

        # remove line from the list of lines
        arcs.pop()

        # add new arc to the list of arcs
        arcmeasures.append(self)

    def adjust_length(self, plot_radius, elev, azim):
        '''
        Shorten the arc such that the marker tip coincides with the actual geometric target point.

        This is similar to the shortening of line segements that are used for vectors, with the additional
        step of removing line segemnts from the tip of the discretized arc and
        , such that the entire arrow head
        marker can be fitted


        '''

        # pick the last segment of the discretized arc
        s    = self.r[:,-1] - self.r[:,-2]
        sabs = np.sqrt(s.dot(s))
        # unit vector along this last segment
        e = s/sabs
        ax = plt.gca()

        # offset for the used marker
        delta = Marker.deltas[self.shape]*self.linewidth
        # length of unit vector projected into display
        l = projected_length(elev, azim, e)
        # endpoint correction in 3D space
        d = 2*plot_radius*delta/l

        # find the last node of the discretized arc to keep
        n    = int(np.ceil(d/sabs))            # n can be 1, 2, 3, ...
        s    = self.r[:,-1] - self.r[:,-1-n]
        sabs = np.sqrt(s.dot(s))
        e = s/sabs
#       ax.plot([self.r[0,-1-n], self.r[0,-1]], [self.r[1,-1-n], self.r[1,-1]], [self.r[2,-1-n], self.r[2,-1]])
        # length of unit vector projected into display
        l = projected_length(elev, azim, e)

        # endpoint correction in 3D space
        self.r[:,-n] = self.r[:,-1] - e*2*plot_radius*delta/l
        # crop nodal array
        if n > 1:
            np.delete(self.r, np.s_[-n+1:], 1)
            np.delete(self.r, np.s_[-1:], 1)
        self.arc.set_data_3d(self.r[0,:-n+1], self.r[1,:-n+1], self.r[2,:-n+1])

class Marker:
    '''Class for marker objects for use as
    - arrowheads of vectors,
    - arrowheads of arc measures,
    - points.

    To precisely position an arrowhead for arc measures, the marker path needs to be adjusted such that
    the base of the arrowhead (the local origin of the marker path) coincides with the end point of the
    line segment to which it is attached. This can best be achieved by adjusting the x-value of the
    translate() function of the path's transform attribute. This is easiest done in Inkscape, by first
    drawing a horizontal line and adding the desired arrowhead, then double clicking this line to show
    its two control points. Activating the XML Editor, one needs to look for the <defs> section and the
    correct <marker>. Clicking the <path> subelement of this marker, one can manually adjust the x-value
    of the translate() function until the base of the arrowhead is precisely positioned on the end
    control point of the line. The determined value is then changed in the corresponding entry of the
    paths dictionary below.

    In a second step, the corresponding value of the deltas dictionary below needs to be adjusted. The
    numerical value indicates by how far the line (vectors) or discretized arc (arcmeasures) needs to
    be shortened to position the tip of the arrowhead precisely on the target.

    The current SVG standard (1.1) does not allow for inheritance of the color attribute from
    the object to which the marker is attached, to the marker. The new SVG standard (2) does
    but it is not implemented in webbrowsers yet. For that reason, we generate separate
    markers for each combination of marker shape and color.
    '''

    # Dictionary of marker paths, follows Inkscape default markers
    paths = {
        'Arrow1Mend' : ';stroke-width:1pt;opacity:1;stroke-linejoin:miter" d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z" transform="scale(0.4) rotate(180) translate(-0.9,0)',
        'Arrow1Lend' : ';stroke-width:1pt;opacity:1;stroke-linejoin:miter" d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z" transform="scale(0.8) rotate(180) translate(-0.9,0)',
        'Point1M'    : ';stroke-width:3;opacity:1;f" d="M -2.5,-1.0 C -2.5,1.7600000 -4.7400000,4.0 -7.5,4.0 C -10.260000,4.0 -12.5,1.7600000 -12.5,-1.0 C -12.5,-3.7600000 -10.260000,-6.0 -7.5,-6.0 C -4.7400000,-6.0 -2.5,-3.7600000 -2.5,-1.0 z" transform="scale(0.25) translate(7.4, 1)'
    }

    # Dictionary of path offsets (to be determined empirically for each vector marker path)
    deltas = {
        'Arrow1Mend' : 0.0204,
        'Arrow1Lend' : 0.0408
    }

    def __init__(self, shape=None, style=None, facecolor='k', edgecolor='k', css_style='overflow:visible', refX=0, refY=0, orient='auto'):
        # shape
        self.shape = shape
        # colors
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        # marker id composed of path + color
        self.id = style
        # marker css style
        self.css_style = css_style
        # refX and refY
        self.refX = str(refX)
        self.refY = str(refY)
        # orientation
        self.orient = orient
        # svg code
        hexfacecolor = mpl.colors.to_hex(facecolor)
        hexedgecolor = mpl.colors.to_hex(edgecolor)
        marker_start = '<marker id="' + self.id + '" style="' + self.css_style + '" refX="' + self.refX + '" refY="' + self.refY + '" orient="' + self.orient + '">'
        marker_path  = '   <path style="stroke:' + hexedgecolor + ';fill:' + hexfacecolor + Marker.paths[shape] + '"/>'
        marker_end   = '</marker>'
        self.svg = marker_start + '\n' + marker_path + '\n' + marker_end

def save_svg(file='unnamed.svg'):
    '''Function for modifying the generated SVG code.

    Inspired by https://matplotlib.org/stable/gallery/misc/svg_filter_line.html and other sources.

    Before saving the drawing, vector lines are shortened. See also https://stackoverflow.com/a/50797203

    '''

    # get current Axes instance and prepare axes for plotting
    ax = plt.gca()
    plot_radius = set_axes_equal(ax)
    ax.set_box_aspect([1,1,1]) # requires matplotlib 3.3.0

    # Shorten all vectors lines such that the marker tip coincides with the actual vector end point
    for vec in vectors:
        vec.adjust_length(plot_radius, ax.elev, ax.azim)

    # Shorten all arcs such that the marker tip coincides with the actual vector end point
    for arc in arcmeasures:
        arc.adjust_length(plot_radius, ax.elev, ax.azim)

    # save the figure as a byte string in SVG format
    f = io.BytesIO()
    plt.savefig(f, format="svg")

    # read in the saved SVG and define the SVG namespace
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace("", ns)
    tree, xmlid = ET.XMLID(f.getvalue())

    # remove the defs element with matplotlib's default css style
#    for defs in tree.findall('{'+ns+'}defs'):
#        if defs.findall('{'+ns+'}style'):
#            tree.remove(defs)

    # create the defs section with previously generated marker elements
    defs = '<defs>' + '\n'
    for m in markers:
        defs =  defs + textwrap.indent(m.svg, 2*' ') + '\n'
    defs = defs + '</defs>'

    # insert the marker definition in the svg dom tree.
    tree.insert(0, ET.XML(defs))

#> should be possible to process this as just one list!

    # process all vectors
    for v in vectors:
        velement  = xmlid[v.gid]
        velement.set('marker-end', 'url(#' + v.style + ')')
        style = velement.find('.//{'+ns+'}path').attrib['style'].replace('stroke-opacity', 'opacity')
        velement.find('.//{'+ns+'}path').attrib['style'] = style

    # process all arcmeasures
    for a in arcmeasures:
        aelement  = xmlid[a.gid]
        aelement.set('marker-end', 'url(#' + a.style + ')')
        style = aelement.find('.//{'+ns+'}path').attrib['style'].replace('stroke-opacity', 'opacity')
        aelement.find('.//{'+ns+'}path').attrib['style'] = style

    # process all points
    for p in points:
        pelement  = xmlid[p.gid]
        pelement.set('marker-start', 'url(#' + p.style + ')')
        style = pelement.find('.//{'+ns+'}path').attrib['style'].replace('stroke-opacity', 'opacity')
        pelement.find('.//{'+ns+'}path').attrib['style'] = style

    print(f"Saving '{file}'")
    ET.ElementTree(tree).write(file, encoding="utf-8")
