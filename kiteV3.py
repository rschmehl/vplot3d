#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:03:53 2024

Programmatically create 3D vector diagrams for SVG output. This module expands
the vplot3d package by a a 3D model of the TU Delft V3 kite as a set of
triangular meshes and line segments read from Wavefront OBJ-files. The meshes
are combined in a single Poly3DCollection to achieve a reasonable depth sorting
(which is limited by the limited depth sorting capability of matplotlib). Depth
sorting problems can be mitigated visually by applying a degree of transparency
to the mesh objects. The bridle line system is drawn without any z-order
considerations, which works well together with the semi-transparent surfaces.

@author: Roland Schmehl
"""

from vplot3d import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import trimesh
from matplotlib.colors import LightSource
from transforms3d.axangles import axangle2mat

# Lists for geometrical objects
kiteV3 = []

class KiteV3(Object3D):
    '''Class for a graphical representation of the TU Delft V3 kite.
    The kite consists of a wing, a suspended kite control unit (KCU), and a
    bridle line system. The wing is composed of a canopy, a leading edge tube
    and several strut tubes.
    '''
    def __init__(self, p=ORIGIN, e1=None, e2=None, e3=None, voff=ORIGIN,
                 v=[[EXYZ]], id=None, linewidth=LINEWIDTH,
                 canopycolor='#dcdcdc4d', tubecolor='#000000',
                 kcucolor='#484848b3', scale=1, zorder=0, alpha=1,
                 azdeg=225.0, altdeg=40.0):
        '''Constructor.
        Draws the kite with nodal points v specified relative to a reference point p.
        Input
          p              : polygon reference point coordinates, absolute
          e1, e2, e3     : Vector base in which to plot, anchored in p
          voff           : Offset relative to p
          id             : name identifier
          linewidth      : line width for the bridle lines
          canopycolor    : color of the canopy
          tubecolor      : color of the tubes
          kcucolor       : color of the KCU
          scale          : scale of polygon, relative to p
          zorder         : parameter used for depth sorting
          alpha          : transparency of polygon line and fill colors
          azdeg          : azimuth (0-360, degrees clockwise from North) of the light source
          altdeg         : altitude (0-90, degrees up from horizontal) of the light source
        '''
        super().__init__(p, id, linewidth, scale, zorder, alpha)

        # set unique gid
        self.gid = 'kiteV3_' + str(len(kiteV3)+1)

        # Read all meshes from obj-files
        mesh1 = trimesh.load('../data/kite_V3_canopy.obj')
        mesh2 = trimesh.load('../data/kite_V3_LE.obj')
        mesh3 = trimesh.load('../data/kite_V3_struts.obj')
        mesh4 = trimesh.load('../data/kite_V3_KCU.obj')
        mesh5 = LineSystem.load('../data/kite_V3_bridle.obj')

        # Add bridle line system separately
        nodes = mesh5.vertices
        lines = mesh5.segments

        # Fix rotation of bridle line system
        m = axangle2mat([1, 0, 0], np.radians(90))
        for i in range(len(nodes)):
            nodes[i] = m.dot(nodes[i])

        # Get coordinates of bridle point from bridle line system
        bp = nodes[np.argmin(nodes[:,2]),:]

        # Translate and scale entire mesh
        nodes = (nodes - bp) * scale

        self.polyline_nodes = nodes
        self.polyline_lines = lines

        lc = Line3DCollection([nodes[lines[i,:]] for i in range(len(lines))],
                               edgecolor='k', linewidths=1, linestyles='solid')

        ln = self.ax.add_collection3d(lc)
        ln.set_gid(self.gid + '_bridle')
        self.lines = ln

        kiteV3.append(self)

        # Move KCU to correct position
        mesh4.vertices[:,2] = mesh4.vertices[:,2] - 5.7
        #mesh4.vertices[:,1] = mesh4.vertices[:,1] + 0.2

        mesh  = trimesh.util.concatenate((mesh1, mesh2, mesh3, mesh4))
        fc    = ['#dcdcdc4d']*len(mesh1.faces) + \
                ['#000000']*len(mesh2.faces) + \
                ['#000000']*len(mesh3.faces) + \
                ['#484848b3']*len(mesh4.faces)

        # Translate and scale entire mesh
        nodes = mesh.vertices - bp
        nodes = nodes * scale
        faces = mesh.faces

        self.mesh_nodes = nodes
        self.mesh_faces = faces

        ls = LightSource(azdeg=225.0, altdeg=40.0)

        pc = Poly3DCollection([nodes[faces[i,:]] for i in range(len(faces))],
                               edgecolor=fc, facecolors=fc,
                               linewidths=0, shade=True, lightsource=ls)

        ms = self.ax.add_collection3d(pc)
        ms.set_gid(self.gid + '_wing_and_kcu')
        self.mesh = ms

        # Cartesian unit vectors
        self.ax.scatter(0, 0, 0, color='k')
        self.ax.plot([0,1], [0,0], [0,0], color='r')
        self.ax.plot([0,0], [0,1], [0,0], color='g')
        self.ax.plot([0,0], [0,0], [0,1], color='b')

class LineSystem():
    '''Class for line system objects.
    '''

    def __init__(self, verts, segs):
        self.vertices = verts
        self.segments = segs

    @classmethod
    def load(cls, fn):
        """
        Read polylines from an obj-file. Adapted from
        https://github.com/zishun/MeshUtility/blob/main/meshutility/obj_lines_io.py

        Input
          fn: path to obj-file
        Output
          verts: vertex coordinates (n x 3)
          lines: polyline segments
        """
        verts = []
        segs = []
        with open(fn, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            s = lines[i].split()
            if (len(s) == 0):
                continue
            if (s[0] == 'v'):
                verts.append([float(s[1]), float(s[2]), float(s[3])])
            elif (s[0] == 'l'):
                L = list(map(lambda x: int(x),  s[1:]))
                segs.extend([[L[i], L[i+1]] for i in range(len(L)-1)])

        # segs are 1-indexed -> 0-indexed
        return cls(np.array(verts), np.array(segs)-1)