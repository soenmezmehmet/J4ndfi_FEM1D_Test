#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:31:10 2025

@author: wanghuating
"""

# Datei: apply_bc.py
# --------------------------
# Schritt 5: Behandlung von Randbedingungen (zwei Methoden unterstützt)

import numpy as np

def apply_bc_slicing(K, f, bc):
    """
    Wendet Dirichlet-Randbedingungen an, indem Zeilen und Spalten entfernt werden.
    
    Parameter:
        K: globale Steifigkeitsmatrix
        f: globaler Kraftvektor
        bc: Liste von Randbedingungen als Tupel (Index, Wert)

    Rückgabe:
        K_mod: reduzierte Steifigkeitsmatrix
        f_mod: reduzierter Kraftvektor
        free_dofs: Liste der freien Freiheitsgrade
    """
    fixed_dofs = [i for (i, val) in bc]
    free_dofs = [i for i in range(len(f)) if i not in fixed_dofs]

    K_mod = K[np.ix_(free_dofs, free_dofs)]
    f_mod = f[free_dofs].copy()

    for i, val in bc:
        for idx_j, j in enumerate(free_dofs):
            f_mod[idx_j] -= K[j, i] * val

    return K_mod, f_mod, free_dofs

def apply_bc_penalty(K, f, bc, penalty=1e20):
    """
    Wendet Randbedingungen mit der Strafmethode (Penalty Method) an.
    
    Parameter:
        K: globale Steifigkeitsmatrix
        f: globaler Kraftvektor
        bc: Liste von Randbedingungen als Tupel (Index, Wert)
        penalty: großer Strafwert (Standard: 1e20)
    
    Rückgabe:
        K, f: modifizierte Steifigkeitsmatrix und Kraftvektor
    """
    for i, val in bc:
        K[i, i] += penalty
        f[i] += penalty * val
    return K, f