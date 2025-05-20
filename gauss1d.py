#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:30:16 2025

@author: wanghuating
"""

# Datei: gauss1d.py
# --------------------------
# Schritt 2: Rückgabe der Gauß-Integrationspunkte und -gewichte (1D)

def gauss1d(nqp):
    """
    Gibt nqp Gauß-Punkte und die entsprechenden Gewichte zurück (im Standardintervall [-1, 1])
    """
    if nqp == 1:
        return [0.0], [2.0]
    elif nqp == 2:
        xi = [-1.0 / 3.0**0.5, 1.0 / 3.0**0.5]
        w = [1.0, 1.0]
        return xi, w
    elif nqp == 3:
        xi = [-3.0**0.5/5.0**0.5, 0.0, 3.0**0.5/5.0**0.5]
        w = [5/9, 8/9, 5/9]
        return xi, w
    else:
        raise ValueError("Nur nqp = 1, 2, 3 werden unterstützt")