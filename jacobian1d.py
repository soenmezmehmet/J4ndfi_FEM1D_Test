#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Datei: jacobian1d.py
# --------------------------
# Schritt 3: Berechnung der Jacobi-Matrix (in 1D ist es ein Skalar)

def jacobian1d(xe, gamma, nen):
    """
    Berechnet die Jacobi-Determinante und deren Inverse für ein 1D-Element.
    
    Parameter:
        xe: Physikalische Koordinaten der Elementknoten
        gamma: Ableitungen der Formfunktionen bezüglich ξ
        nen: Anzahl der Knoten im Element

    Rückgabe:
        J, Jinv: Jacobi-Determinante und deren Inverse
    """
    J = sum(xe[A] * gamma[A] for A in range(nen))
    if J <= 0:
        raise ValueError("Die Jacobi-Determinante darf nicht negativ oder null sein")
    return J, 1.0 / J
