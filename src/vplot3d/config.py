#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:46:25 2024

@author: rschmehl
"""
import numpy as np

class DefaultsClass:
    def __init__(self, raw):
        if raw.get('linewidth'):     self.linewidth     = raw['linewidth']
        if raw.get('fontsize_raw'):  self.fontsize_raw  = raw['fontsize_raw']
        if raw.get('fontfamily'):    self.fontfamily    = raw['fontfamily']
        if raw.get('fontsize'):      self.fontsize      = raw['fontsize']
        if raw.get('baselineskip'):  self.baselineskip  = raw['baselineskip']
        if raw.get('rasterize_dpi'): self.rasterize_dpi = raw['rasterize_dpi']
        if raw.get('origin'):        self.origin        = np.asarray(raw['origin'].split(), dtype=float)
        if raw.get('ex'):            self.ex            = np.asarray(raw['ex'].split(), dtype=float)
        if raw.get('ey'):            self.ey            = np.asarray(raw['ey'].split(), dtype=float)
        if raw.get('ez'):            self.ez            = np.asarray(raw['ez'].split(), dtype=float)
        if raw.get('exyz'):          self.exyz          = np.asarray(raw['exyz'].split(), dtype=float)
        if raw.get('plot_zoom'):     self.plot_zoom:   float = raw['plot_zoom']
        if raw.get('plot_radius'):   self.plot_radius: float = raw['plot_radius']
        if raw.get('eps'):           self.eps:         float = raw['eps']
        if raw.get('xyoff'):         self.xyoff:       float = raw['xyoff'].split()
        if raw.get('ddegrees'):      self.ddegrees:    float = raw['ddegrees']
        
class Config:
    def __init__(self, raw):
        self.defaults = DefaultsClass(raw['defaults'])
