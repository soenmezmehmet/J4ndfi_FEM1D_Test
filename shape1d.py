#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:29:03 2025

@author: wanghuating
"""

# Datei: shape1d.py
# --------------------------
# Schritt 1: Definition der Formfunktionen und ihrer Ableitungen (linear oder quadratisch)

import sympy as sp

def shape1d(xi_val, nen):
    """
    Berechnet die 1D-Formfunktionen und ihre Ableitungen an der Stelle xi_val im natürlichen Koordinatensystem.
    
    Parameter:
        xi_val: float – Auswertungsstelle ξ
        nen: int – Anzahl der Knoten pro Element (2 oder 3)
        
    Rückgabe:
        N: [nen] – Werte der Formfunktionen
        gamma: [nen] – Ableitungen der Formfunktionen bezüglich ξ
    """
    xi = sp.Symbol('xi')

    if nen == 2:
        xi_nodes = [-1, 1]
    elif nen == 3:
        xi_nodes = [-1, 0, 1]
    else:
        raise ValueError("Nur 2 oder 3 Knoten werden unterstützt.")

    N_syms = []
    gamma_syms = []

    for A in range(nen):
        LA = 1
        for B in range(nen):
            if B != A:
                LA *= (xi - xi_nodes[B]) / (xi_nodes[A] - xi_nodes[B])
        N_syms.append(LA)
        gamma_syms.append(sp.diff(LA, xi))

    N = [float(Ni.subs(xi, xi_val)) for Ni in N_syms]
    gamma = [float(dNi.subs(xi, xi_val)) for dNi in gamma_syms]

    return N, gamma
