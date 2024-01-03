#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:03:50 2022

Programmatically creating 3D vector diagrams for SVG output. The following
diagram objects can be used

- points,
- lines and circular arcs,
- vectors and arc measures.

Points and arrowheads (for vectors and arc measures) are generated as SVG
markers. At the time of writing this library, the MarkerKnockout feature did
not make it into the SVG2 standard. This would have been the perfect native
solution for the most painful problem addressed by this library, to make arrow
heads ending precisely at the target node:
https://svgwg.org/specs/markers/#MarkerKnockout

@author: Roland Schmehl
"""

from abc import ABC, abstractmethod
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d
from matplotlib.text import Annotation
from  matplotlib.colors import is_color_like
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy
import io
import xml.etree.ElementTree as ET
import textwrap

# Default values
ORIGIN    = np.array([0, 0, 0])
EX        = np.array([1, 0, 0])
EY        = np.array([0, 1, 0])
EZ        = np.array([0, 0, 1])
EXYZ      = np.array([1, 1, 1])
LINEWIDTH = 3                              # Linewidth of line objects
FONTSIZE  = 12                             # Fontsize for text objects
XYOFF     = (5,5)                          # xy-offset of text objects
DEGREES   = np.arange(0, 361, 1)           # Discretization of circular objects
COS       = np.cos(np.radians(DEGREES))
SIN       = np.sin(np.radians(DEGREES))

# Lists for geometrical objects
lines       = []
vectors     = []
arcs        = []
arcmeasures = []
polygons    = []
points      = []
markers     = []

# Open file with marker definitions
mtree = ET.parse('markers.svg')
mroot = mtree.getroot()

# Raw Latex math - see https://github.com/matplotlib/matplotlib/issues/4938#issuecomment-783252908
RAW_MATH  = False
def _m(s):
    '''Helper method escaping backslash for raw math output
    '''
    return s.replace("$", r"\$") if RAW_MATH else s

def figsize(figure_width_px, figure_height_px):
    ''' Sets figure size in inches, given a desired width and height in pixels.
    This approach is necessary because the SVG backend (print_svg) uses a
    hardcoded DPI value of 72.
    See also function save_svg, where the hardcoded points "pt" units are
    removed from the SVG root attributes width and height.

    Input
      figure_width_px:  figure width in pixels
      figure_height_px: figure height in pixels
    Output
      figsize:          figure width and height in inches
    '''
    fixed_dpi = 72    # SVG backend (print_svg) uses this hardcoded DPI value
    return figure_width_px/fixed_dpi, figure_height_px/fixed_dpi

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

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s

    Inspired by https://stackoverflow.com/a/42915422
    '''
    def __init__(self, s, xyz, xytext=None, fontsize=None, *args, **kwargs):
        if xytext is None:
            xytext = XYOFF
        if fontsize is None:
            fontsize = FONTSIZE
        super().__init__(_m(s), xy=(0,0), xytext=xytext, fontsize=fontsize, textcoords='offset points', *args, **kwargs)
        self._verts3d = xyz
        self.ax = plt.gca()
        self.ax.add_artist(self)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs,ys)
        Annotation.draw(self, renderer)

class Object3D(ABC):
    '''Abstract class for 3D objects.
    The three colors edgecolor, facecolor, bgcolor can be used to control the coloring of a specific Object3D.
    The specific meaning may vary per child class.
    '''
    @abstractmethod
    def __init__(self, p=ORIGIN, id=None, linewidth=LINEWIDTH, scale=1, zorder=0, edgecolor=None, facecolor=None, bgcolor=None, alpha=1):
        '''Constructor.

        The three colors edgecolor, facecolor and bgcolor can be used to set different color regions in the Object3D,
        depending on the specific implementation. This set could be extended in the future for more complex Object3D
        implementations.

        Input
          p         : reference point coordinates
          id        : name identifier
          linewidth : line width
          scale     : scale of object, relative to p
          zorder    : parameter used for depth sorting
          edgecolor : stroke color of object
          facecolor : fill color (foreground) of abject
          bgcolor   : fill color (background) of abject
          alpha     : transparency of object
        '''
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
        # object colors
        self.edgecolor = edgecolor
        self.facecolor = facecolor
        self.bgcolor = bgcolor
        # object transparency
        self.alpha = alpha
        # object group id
        self.gid = None

    def add_marker(self, shape, edgecolor, facecolor, bgcolor):
        '''Add a marker to the object.
        Input
          shape     : type of marker to be added
          edgecolor : stroke color of object
          facecolor : fill color (foreground) of abject
          bgcolor   : fill color (background) of abject
        '''
        # marker geometry
        self.shape = shape
        # marker style (= shape + colors)
        mec = mfc = mbc = ''
        if is_color_like(bgcolor):
            mbc = ':' + bgcolor
            mfc = ':N'
            mec = ':N'
        if is_color_like(facecolor):
            mfc = ':' + facecolor
            mec = ':N'
        if is_color_like(edgecolor):
            mec = '-' + edgecolor
        self.style = shape + mec + mfc + mbc
        # add it to the list of markers if it is not yet included
        if self.style not in [m.id for m in markers]:
            # For arrowheads we use the same stroke and fill color (i.e. edge and face color)
            markers.append(Marker(shape, self.style, edgecolor=edgecolor, facecolor=facecolor, bgcolor=bgcolor))

class Point(Object3D):
    '''Class for point objects.
    Display SVG marker objects from file markers.svg at a specific location.
    Markers scale with linewidth of the line they are attached to. We use this linewidth for scaling the Point obkects.
    Default values of colors are taken from SVG file.
    '''
    def __init__(self, p=ORIGIN, id=None, scale=1, shape='Point1M', zorder=0, color=None, edgecolor=None, facecolor=None, bgcolor=None, alpha=1, *args, **kwargs):
        '''Constructor.
        Input
          p         : point coordinates
          id        : name identifier
          scale     : scaling factor
          shape     : type of marker to be added
          zorder    : parameter used for depth sorting
          color     : color of point object
          edgecolor : stroke color of point
          facecolor : fill color (foreground) of point
          bgcolor   : fill color (background) of point
          alpha     : transparency of point
        '''
        if color is not None:
            if edgecolor is None: edgecolor = color
            if facecolor is None: facecolor = color
            if bgcolor is None: bgcolor = 'w'

        super().__init__(p, id, scale*LINEWIDTH, 1, zorder, edgecolor, facecolor, bgcolor, alpha)
        super().add_marker(shape, edgecolor, facecolor, bgcolor)

        # set unique gid
        self.gid = 'point_' + str(len(points)+1)
        # plot a point where the marker is placed later
        line, = self.ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], p[2]], zorder=self.zorder, linewidth=self.linewidth, solid_capstyle='butt', color=self.edgecolor, alpha=self.alpha)
        line.set_gid(self.gid)
        self.line = line
        # add new point to the list of points
        points.append(self)

class Line(Object3D):
    '''Class for line objects.
    '''
    def __init__(self, p=ORIGIN, v=EXYZ, id=None, linewidth=LINEWIDTH, scale=1, zorder=0, color='k', alpha=1, *args, **kwargs):
        '''Constructor.
        Input
          p         : line starting point coordinates (absolute)
          v         : line end point coordinates, relative to p
          id        : name identifier
          linewidth : line width
          scale     : scale of line, relative to p
          zorder    : parameter used for depth sorting
          color     : color of line
          alpha     : transparency of line
        '''
        super().__init__(p, id, linewidth, scale, zorder, color, None, None, alpha)

        # set unique gid
        self.gid = 'line_' + str(len(lines)+1)
        # line segment coordinates
        self.v = v*scale
        # calculate line end point
        q = p + v

        # plot the line
        line, = self.ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], zorder=self.zorder, linewidth=self.linewidth, solid_capstyle='butt', color=self.edgecolor, alpha=self.alpha)
        line.set_gid(self.gid)
        self.line = line
        # add new vector to the list of vectors
        lines.append(self)

class Vector(Line):
    '''Class for vector objects, consisting of a line and an arrowhead attached to its end point.
    The arrowhead is not drawn explicitly, but added as an SVG marker object.
    '''
    def __init__(self, p=ORIGIN, v=EXYZ, id=None, linewidth=LINEWIDTH, shape='Arrow1Mend', scale=1, zorder=0, color='k', alpha=1):
        '''Constructor.
        Input
          p         : vector origin coordinates (absolute)
          v         : vector target coordinates, relative to p
          id        : name identifier
          linewidth : line width
          shape     : type of marker to be added
          scale     : scale of line, relative to p
          zorder    : parameter used for depth sorting
          color     : color of line
          alpha     : transparency of line
        '''
        super().__init__(p, v, id, linewidth, scale, zorder, color, alpha)
        super().add_marker(shape, color, color, None)

        # set unique gid
        self.gid = 'vector_' + str(len(vectors)+1)
        self.line.set_gid(self.gid)

        # remove line from the list of lines
        lines.pop()

        # add new vector to the list of vectors
        vectors.append(self)

    def adjust_length(self, plot_radius, elev, azim):
        '''Shorten the line to which the arrowhead is attached such that the tip of the arrowhead
        coincides with the intended vector end point. Arrowhead markers are defined in such a way
        that the base of the arrowhead (the local origin of the marker path) coincides with the end
        point of the line segment to which it is attached. Because of this, the line has to be
        shortened by the length of the arrowhead to make the tip of the arrowhead coincide with the
        actual vector end point.

        Because the shortening of the line meeds to accommodate the 2D arrowhead in display
        coordinate space, we need to apply a scaling to the shortening in 3D model coordinate space.
        In fact, the shortening in 3D model coordinate space is always larger, depending on the
        respective projection.
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
    '''Class for circular arc objects, discretized by a number of equidistant line segments.
    '''
    def __init__(self, p=ORIGIN, v1=EX, v2=EY, radius=1, id=None, linewidth=LINEWIDTH, scale=1, zorder=0, color='k', alpha=1):
        '''Constructor.
        Input
          p         : line starting point coordinates (absolute)
          v1        : arc starting vector, relative to p
          v2        : arc target vector, relative to p
          radius    : radius of arc
          id        : name identifier
          linewidth : line width
          scale     : scale of line, relative to p
          zorder    : parameter used for depth sorting
          color     : color of line
          alpha     : transparency of line

        The computed unit vectors e1, e2 and e3 span a local vector base in which the
        discretized arc is computed.
        '''
        super().__init__(p, id, linewidth, 1, zorder, color, None, None, alpha)

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
        # nodal points of the discretized arc
        r = p[:,np.newaxis] + radius*(e1[:,np.newaxis]*COS[np.newaxis,:ip] + e2[:,np.newaxis]*SIN[np.newaxis,:ip])
        self.r = r

        # plot the line
        arc, = self.ax.plot(r[0,:], r[1,:], r[2,:], zorder=self.zorder, linewidth=self.linewidth, solid_capstyle='butt', color=self.edgecolor, alpha=self.alpha)
        arc.set_gid(self.gid)
        self.arc = arc
        # add new arc to the list of arcs
        arcs.append(self)

class ArcMeasure(Arc):
    '''Class for circular arc measure objects, discretized by a number of equidistant line
    segments and an arrowhead attached to its end point. The arrowhead is not drawn explicitly,
    but added as an SVG marker object.
    '''
    def __init__(self, p=ORIGIN, v1=EX, v2=EY, radius=1, id=None, linewidth=LINEWIDTH, shape='Arrow1Mend', scale=1, zorder=0, color='k', alpha=1):
        '''Constructor.
        Input
          p         : line starting point coordinates (absolute)
          v1        : arc starting vector, relative to p
          v2        : arc target vector, relative to p
          radius    : radius of arc
          id        : name identifier
          linewidth : line width
          shape     : type of marker to be added
          scale     : scale of line, relative to p
          zorder    : parameter used for depth sorting
          color     : color of line
          alpha     : transparency of line

        The computed unit vectors e1, e2 and e3 span a local vector base in which the
        discretized arc is computed.
        '''
        super().__init__(p, v1, v2, radius, id, linewidth, scale, zorder, color, alpha)
        super().add_marker(shape, color, color, None)

        # set unique gid
        self.gid = 'arcmeasure_' + str(len(arcmeasures)+1)
        self.arc.set_gid(self.gid)

        # remove line from the list of lines
        arcs.pop()

        # add new arc to the list of arcs
        arcmeasures.append(self)

    def adjust_length(self, plot_radius, elev, azim):
        '''Shorten the arc to which the arrowhead is attached such that the tip of the arrowhead
        coincides with the intended end point. Arrowhead markers are defined in such a way that the
        base of the arrowhead (the local origin of the marker path) coincides with the end point of
        the arc's last line segment. If the line segment is longer than the length of the arrowhead,
        the line segment can just be shortened, similar as for the shortening of the single line
        segment defining a vector. However, if the line segment is shorter than the arrowhead (which
        in most situations will be the case) one or even more line segments of the arc need to be
        removed and replaced by a longer line segment that can accomodate the arrowhead. This
        enlarged line segment is then shortened.

        For the replacement operation, we keep the target point of the arc measure, and determine
        the last node of the discretized arc that we can keep, such the entire arrowhead will fit.
        This replacement line segment is then shortened. In this way, we get a precise placement of
        the arc measure, while maintaining a clean representation of the arrowhead.

        See the Vector.adust_length() how to deal with the fact that the arrowhead is a 2D object
        in display coordinate space, while we need to shorteh the arc in 3D model coordinate space.
        '''
        # offset for the used marker
        delta = Marker.deltas[self.shape]*self.linewidth
        # start at tip of arrowhead and walk back on arc
        for i, alpha in enumerate(DEGREES):
            if i == 0: continue                # i = 1, 2, 3, ...
            # chord vector (determined from discrete points)
            s = self.r[:,-1] - self.r[:,-1-i]
            # chord length
            sabs = np.sqrt(s.dot(s))
            # chord unit vector (pointing towards arc end point)
            e = s/sabs
            # length of chord unit vector projected into display coordinate space
            l = projected_length(elev, azim, e)
            # endpoint correction in 3D space
            d = 2*plot_radius*delta/l
#            print(i, alpha, sabs, d)
            if sabs > d: break

        # endpoint correction in 3D space
        self.r[:,-i] = self.r[:,-1] - e*d
        # reset discrete arc data
        if i > 1: # i = 2, 3, ...
            self.arc.set_data_3d(self.r[0,:-i+1], self.r[1,:-i+1], self.r[2,:-i+1])
        else:
            self.arc.set_data_3d(self.r[0,:], self.r[1,:], self.r[2,:])

class Polygon(Object3D):
    '''Class for polygon objects.
    '''
    def __init__(self, p=ORIGIN, v=[[EXYZ]], id=None, linewidth=LINEWIDTH, scale=1, zorder=0, edgecolor='k', facecolor='w', alpha=1, edgecoloralpha=None):
        '''Constructor.
        Draws a polygon with nodal points v specified relative to a reference point p.
        Input
          p              : polygon reference point coordinates, absolute
          v              : polygon nodal point coordinates, relative to p
          id             : name identifier
          linewidth      : line width
          scale          : scale of polygon, relative to p
          zorder         : parameter used for depth sorting
          facecolor      : fill color of polygon
          edgecolor      : line color of polygon
          alpha          : transparency of polygon line and fill colors
          edgecoloralpha : different transparency of polygon line color
        '''
        super().__init__(p, id, linewidth, scale, zorder, edgecolor, facecolor, None, alpha)

#    def __init__(self, p=ORIGIN, id=None, linewidth=LINEWIDTH, shape='Point1M', zorder=0, edgecolor=None, facecolor=None, bgcolor=None, alpha=1):

        # set unique gid
        self.gid = 'polygon_' + str(len(polygons)+1)
        # colors
        self.facecolor = facecolor
        # alpha seems to not be applied to edgecolor and we do it here explicitly
        if edgecoloralpha is None:
            self.edgecoloralpha = alpha
            self.edgecolor = mpl.colors.to_rgba(edgecolor, alpha=alpha)
        else:
            self.edgecoloralpha = edgecoloralpha
            self.edgecolor = mpl.colors.to_rgba(edgecolor, alpha=edgecoloralpha)
        # scale polygon nodal points, relative to p
        self.v = [[vn*scale for vn in v[0]]]
        # compute absolute coordinates of nodal points
        r = [[p + vn for vn in self.v[0]]]
        self.r = r

        # plot the polygon
        pg = art3d.Poly3DCollection(r, facecolors=self.facecolor, edgecolors=self.edgecolor, linewidths=self.linewidth, alpha=self.alpha, closed=False)
        polygon = self.ax.add_collection3d(pg)
        polygon.set_gid(self.gid)
        self.polygon = polygon
        # add new polygon to the list of polygons
        polygons.append(self)

    @classmethod
    def rotated(cls, p=ORIGIN, v=None, file=None, e1=None, e2=None, e3=None, voff=ORIGIN, d=None, linewidth=LINEWIDTH, scale=1, zorder=0, facecolor='w', edgecolor='k', alpha=1, edgecoloralpha=None):
        '''Simulated constructor.
        The polygon is plotted in a vector base (e1, e2, e3), of which at least two axis-diections must be specified.
        The vectors e1, e2 and e3 do not need to be normalized.

        Input
          p         : polygon reference point coordinates, absolute
          v         : polygon nodal point coordinates in (e1, e2, e3) relative to p
          file      : name of file with polygon nodal point coordinates in (e1, e2, e3) relative to p
          e1        : x-direction new vector base
          e2        : y-direction new vector base
          e3        : z-direction new vector base
          voff      : offset applied to nodal points, relative to p
          id        : name identifier
          linewidth : line width
          scale     : scale of polygon, relative to p
          zorder    : parameter used for depth sorting
          facecolor : fill color of polygon
          edgecolor : line color of polygon
          alpha     : transparency of line
        '''
        # complete the vector base
        if e1 is None:
            e1 = np.cross(e2, e3)
        e1abs = np.sqrt(e1.dot(e1))
        e1    = e1/e1abs
        if e2 is None:
            e2 = np.cross(e3, e1)
        e2abs = np.sqrt(e2.dot(e2))
        e2    = e2/e2abs
        if e3 is None:
            e3 = np.cross(e1, e2)
        e3abs = np.sqrt(e3.dot(e3))
        e3    = e3/e3abs

        if  file is not None:

            # read data from file
            data = np.loadtxt(file, skiprows=1)
            # add a zero column if data is 2D
            if data.shape[1] < 3:
                # add column
                col = np.zeros((data.shape[0],1))
                data = np.append(data, col, axis=1)
            v = [list(data)]

        # calculate polygon nodal point coordinates, relative to p
        r = [[(vn[0]+voff[0])*e1 + (vn[1]+voff[1])*e2 + (vn[2]+voff[2])*e3 for vn in v[0]]]
        return cls(p, r, id, linewidth, scale, zorder, facecolor, edgecolor, alpha, edgecoloralpha)


class Marker:
    '''Class for marker objects for use as
    - arrowheads of vectors,
    - arrowheads of arc measures,
    - points.

    The constructor only generates the SVG code to be used for the marker.

    Color mapping: the SVG file markers.svg should contain only the following key-color combinations:

        none is never overwritten
        stroke:#000000 (black) -> edgecolor
        fill:#000000   (black) -> facecolor
        fill:#ffffff   (white) -> bgcolor

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

    It seems that value of the scale() function of the path's transform attribute and the deltas value
    are linearly related. That means that the same scaling factor can be used for scale() and deltas
    values.

    The current SVG standard (1.1) does not allow for inheritance of the color attribute from
    the object to which the marker is attached, to the marker. The new SVG standard (2) does
    but it is not implemented in webbrowsers yet. For that reason, we generate separate
    markers for each combination of marker shape and color.
    '''

    # Dictionary of path offsets (to be determined empirically for each vector marker path)
    deltas = {
        'Arrow1Mend' : 0.0204,
        'Arrow1Lend' : 0.0408
    }

    def __init__(self, shape=None, style=None, edgecolor=None, facecolor=None, bgcolor=None, css_style='overflow:visible', refX=0, refY=0, orient='auto'):
        '''Constructor.
        Input
          shape     : type of marker to be added
          style     : marker id composed of path + (edge)color
          edgecolor : line color of marker
          facecolor : fill color of marker
          bgcolor   : background color
          css_style : CSS style
          refX      : x-displacement of marker
          refY      : y-displacement of marker
          orient    : orientation of marker
        '''
        # shape
        self.shape = shape
        # colors
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.bgcolor = bgcolor
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
        svg = ''
        marker = mroot.find('.//defs/marker[@id="{}"]'.format(shape))
        if marker is None:
            print('Marker ' + shape + ' not found in markers.svg')
        else:
            # Iterate through all paths of the marker
            for path in marker:
                p = copy.deepcopy(path)
                # build a dictionary from the css-style information
                # Remove appended semicolon which is valid CSS but crashes the following splitting algorithm
                if p.attrib['style'][-1] == ';': p.attrib['style'] = p.attrib['style'][:-1]
                style = {pair.split(":")[0]:pair.split(":")[1] for pair in p.attrib['style'].split(";")}
                if 'stroke' in style:
                    if edgecolor is not None and style['stroke'] != 'none':
                        style['stroke'] = mpl.colors.to_hex(edgecolor)
                if 'fill' in style:
                    if mpl.colors.to_hex(style['fill']) == '#ffffff':
                        if bgcolor is not None and style['fill'] != 'none':
                            style['fill'] = mpl.colors.to_hex(bgcolor)
                    elif mpl.colors.to_hex(style['fill']) == '#000000':
                        if facecolor is not None and style['fill'] != 'none':
                            style['fill'] = mpl.colors.to_hex(facecolor)
                    else:
                        print('Color ' + str(style['fill']) + ' in marker ' + str(marker.attrib['id']) + ' in file markers.svg is not changed')
                p.attrib['style'] = ';'.join(f'{key}:{value}' for key, value in style.items())
                svg = svg + '<' + p.tag + ' ' + ' '.join(f'{key}="{value}"' for key, value in p.attrib.items()) + '/>\n'
        marker_start = '<marker id="' + self.id + '" style="' + self.css_style + '" refX="' + self.refX + '" refY="' + self.refY + '" orient="' + self.orient + '">'
        marker_end   = '</marker>'
        self.svg = marker_start + '\n' + svg + marker_end

def save_svg(file='unnamed.svg'):
    '''Function for modifying the generated SVG code.

    Inspired by https://matplotlib.org/stable/gallery/misc/svg_filter_line.html and other sources.

    Before saving the drawing, lines for vectors and polylines for arc measures are shortened.
    See also https://stackoverflow.com/a/50797203

    '''
    # get current Axes instance and prepare axes for plotting
    ax = plt.gca()
    plot_radius = set_axes_equal(ax)
    ax.set_box_aspect([1,1,1], zoom=3) # requires matplotlib 3.3.0

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

    # Remove the "pt" units from the width and height attributes
    tree.attrib['width']  = tree.attrib['width'].removesuffix('pt')
    tree.attrib['height'] = tree.attrib['height'].removesuffix('pt')

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
