#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:46:25 2024

@author: rschmehl
"""

class DefaultsClass:
    def __init__(self, raw):
        self.linewidth    = raw['linewidth']    if raw.get('linewidth')    else 1
        self.fontsize_raw = raw['fontsize_raw'] if raw.get('fontsize_raw') else 12

class Config:
    def __init__(self, raw):
        self.defaults = DefaultsClass(raw['defaults'])
