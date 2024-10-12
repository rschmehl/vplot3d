#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:46:25 2024

@author: rschmehl
"""

class DefaultsClass:
    def __init__(self, raw):
        if raw.get('linewidth'):     self.linewidth     = raw['linewidth']
        if raw.get('fontsize_raw'):  self.fontsize_raw  = raw['fontsize_raw']
        if raw.get('fontfamily'):    self.fontfamily    = raw['fontfamily']
        if raw.get('fontsize'):      self.fontsize      = raw['fontsize']
        if raw.get('baselineskip'):  self.baselineskip  = raw['baselineskip']
        if raw.get('plot_zoom'):     self.plot_zoom     = raw['plot_zoom']
        if raw.get('plot_radius'):   self.plot_radius   = raw['plot_radius']
        if raw.get('rasterize_dpi'): self.rasterize_dpi = raw['rasterize_dpi']
        
class Config:
    def __init__(self, raw):
        self.defaults = DefaultsClass(raw['defaults'])
