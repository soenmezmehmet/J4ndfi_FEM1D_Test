#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:06:29 2025

@author: wanghuating
"""

# Datei: main.py
# --------------------------
# Hauptprozess + Konvergenzanalyse bei unterschiedlichen Elementzahlen

import numpy as np
import matplotlib.pyplot as plt
from shape1d import shape1d
from jacobian1d import jacobian1d  
from assemble import element_stiffness_force
from apply_bc import apply_bc_slicing

# --------------------------
# Globale Parameterdefinition
# --------------------------
L = 70.0           # Seillänge (Länge des Stabs)
E = 2e11           # Elastizitätsmodul (Pa)
A = 89.9e-6           # Querschnittsfläche (m²)
EA = E * A         # Axiale Steifigkeit
rho = 7850         # Dichte (kg/m³)
g = 9.81           # Erdbeschleunigung (m/s²)
f_body = -rho * g  # Körperkraft (Eigengewicht), konstant über die Länge
nen = 2            # Anzahl der Knoten pro Element (2 = lineare Elemente)
nqp = 2            # Anzahl der Gauß-Integrationspunkte (Numerische Integration)


# --------------------------
# Einzelfall-Lösung (auch für Konvergenzvergleich nutzbar)
# --------------------------
def run_case(nel, plot_result=True):
    """
    Löst das 1D FEM-Modell für eine gegebene Anzahl an Elementen (nel).
    Gibt x-Knoten, Verschiebungen u(x) und Spannungen σ zurück.
    """
    nn = nel * (nen - 1) + 1                  # Anzahl der Gesamtknoten
    x_nodes = np.linspace(0, L, nn)           # Gleichmäßige Einteilung der Länge

    K_global = np.zeros((nn, nn))             # Globale Steifigkeitsmatrix
    f_global = np.zeros(nn)                   # Globaler Kraftvektor
    

    # -------- Assemblierung ----------
    for e in range(nel):
        nodes_e = list(range(e * (nen - 1), e * (nen - 1) + nen))  # Knotennummern des Elements
        x_e = x_nodes[nodes_e]                                     # Knotenpositionen des Elements
        K_e, f_e = element_stiffness_force(EA, f_body, x_e, nen, nqp)

        # Einfügen in globale Matrix/Vektor
        for i in range(nen):
            for j in range(nen):
                K_global[nodes_e[i], nodes_e[j]] += K_e[i, j]
            f_global[nodes_e[i]] += f_e[i]
            

    # -------- Randbedingungen u(0) = 0 --------
    bc = [(0, 0.0)]                            # Verschiebung am linken Ende ist 0
    K_mod, f_mod, free_dofs = apply_bc_slicing(K_global, f_global, bc)
    

    # -------- Gleichungssystem lösen --------
    u = np.zeros(nn)
    u_free = np.linalg.solve(K_mod, f_mod)    # Nur freie DOFs lösen
    u[free_dofs] = u_free
    

    # -------- Spannung berechnen σ = E * ε --------
    stress = np.zeros(nel)
    for e in range(nel):
        nodes_e = list(range(e * (nen - 1), e * (nen - 1) + nen))
        u_e = u[nodes_e]
        x_e = x_nodes[nodes_e]
        _, gamma = shape1d(0.0, nen)
        J, Jinv = jacobian1d(x_e, gamma, nen)
        dNdx = [g * Jinv for g in gamma]                    # dN/dx = dN/dξ * dξ/dx
        strain = sum(dNdx[i] * u_e[i] for i in range(nen)) # ε = du/dx = Σ (dNi/dx * ui)
        stress[e] = E * strain                              # σ = E * ε

    # -------- Ergebnis darstellen --------
    if plot_result:
        plt.figure()
        plt.plot(x_nodes, u, marker='o', label=f"nel = {nel}")
        plt.xlabel("x (m)")
        plt.ylabel("u(x) (m)")
        plt.title(f"Verschiebung für nel = {nel}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return x_nodes, u, stress


# --------------------------
# Konvergenzanalyse durch Vergleich mehrerer Elementzahlen
# --------------------------
def run_convergence_study(nel_values):
    """
    Führt Konvergenzstudie durch: zeigt mehrere u(x)-Kurven für unterschiedliche Elementzahlen.

    Idee: Wenn die Anzahl der Elemente steigt, sollten sich die u(x)-Kurven nicht mehr stark ändern.
           → Dann gilt: „Konvergenz erreicht“.
    """
    plt.figure(figsize=(10, 6))

    for nel in nel_values:
        x, u, _ = run_case(nel, plot_result=False)  # Für jede Anzahl von Elementen berechnen
        plt.plot(x, u, marker='o', label=f"{nel} Elemente")

    # ---- Plot vergleichen ----
    plt.xlabel("x (m)")
    plt.ylabel("u(x) (m)")
    plt.title("Konvergenzstudie der Verschiebung")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# --------------------------
# Main: zuerst normal rechnen, dann Konvergenz anzeigen
# --------------------------
if __name__ == "__main__":
    # Teil 1: Standardfall lösen（z.B. mit 10 Elementen）
    print("Starte Standardfall mit nel = 10")
    run_case(10)

    # Teil 2: Konvergenzanalyse mit verschiedenen Elementzahlen
    print("Starte Konvergenzstudie ...")
    nel_values = [2, 5, 8, 15, 20]
    run_convergence_study(nel_values)
