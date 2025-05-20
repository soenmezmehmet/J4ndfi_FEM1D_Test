#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:30:45 2025

@author: wanghuating
"""

# Datei: assemble.py
# --------------------------
# Schritt 4: Assemblierung der lokalen Steifigkeitsmatrix und Kraftvektors

import numpy as np
from shape1d import shape1d
from gauss1d import gauss1d
from jacobian1d import jacobian1d

def element_stiffness_force(EA, body_force, x_e, nen, nqp):
    """
    Berechnet die lokale Steifigkeitsmatrix und den Kraftvektor für ein Element.
    
    Parameter:
        EA: Axialsteifigkeit (E * A)
        body_force: Körperkraft pro Volumen (z.B. Eigengewicht)
        x_e: Knotenkoordinaten des Elements
        nen: Anzahl der Knoten im Element (2 oder 3)
        nqp: Anzahl der Gauß-Integrationspunkte

    Rückgabe:
        K_e: Lokale Steifigkeitsmatrix
        f_e: Lokaler Kraftvektor
    """
    K_e = np.zeros((nen, nen))
    f_e = np.zeros(nen)

    qp, w = gauss1d(nqp)

    for q in range(nqp):
        N, gamma = shape1d(qp[q], nen)
        J, Jinv = jacobian1d(x_e, gamma, nen)
        dNdx = [g * Jinv for g in gamma]

        # Beitrag zur Steifigkeitsmatrix
        for A in range(nen):
            for B in range(nen):
                K_e[A, B] += EA * dNdx[A] * dNdx[B] * J * w[q]

        # Beitrag zum Kraftvektor (Körperkraft)
        for A in range(nen):
            f_e[A] += N[A] * body_force * J * w[q]

    return K_e, f_e