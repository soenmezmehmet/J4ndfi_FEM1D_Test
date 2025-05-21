#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 19:51:37 2025

@author: wanghuating
"""

# Datei: main.py
# --------------------------
# Schritt 6: Hauptprozess
# Frage 5 ：Wird die vorgeschriebene Sicherheit eingehalten?


import numpy as np
import matplotlib.pyplot as plt
from shape1d import shape1d
from jacobian1d import jacobian1d  
from assemble import element_stiffness_force
from apply_bc import apply_bc_slicing

# Parameterdefinition
L = 10.0           # Seillänge
E = 2e11           # Elastizitätsmodul (Pa)
A = 1e-4           # Querschnittsfläche (m²)
EA = E * A
rho = 7850         # Dichte (kg/m³)
g = 9.81
f_body = -rho * g  # Körperkraft (Eigengewicht)
nen = 2            # Anzahl der Knoten pro Element
nel = 10           # Anzahl der Elemente
nqp = 2            # Anzahl der Gauß-Integrationspunkte

nn = nel * (nen - 1) + 1
x_nodes = np.linspace(0, L, nn)

K_global = np.zeros((nn, nn))
f_global = np.zeros(nn)

# Assemblierung der globalen Steifigkeitsmatrix und des Kraftvektors
for e in range(nel):
    nodes_e = list(range(e * (nen - 1), e * (nen - 1) + nen))
    x_e = x_nodes[nodes_e]
    K_e, f_e = element_stiffness_force(EA, f_body, x_e, nen, nqp)

    for i in range(nen):
        for j in range(nen):
            K_global[nodes_e[i], nodes_e[j]] += K_e[i, j]
        f_global[nodes_e[i]] += f_e[i]

# Randbedingung u(0) = 0
bc = [(0, 0.0)]
K_mod, f_mod, free_dofs = apply_bc_slicing(K_global, f_global, bc)

# Lösung des Gleichungssystems
u = np.zeros(nn)
u_free = np.linalg.solve(K_mod, f_mod)
u[free_dofs] = u_free

# Nachbearbeitung: Spannung berechnen
stress = np.zeros(nel)
for e in range(nel):
    nodes_e = list(range(e * (nen - 1), e * (nen - 1) + nen))
    u_e = u[nodes_e]
    x_e = x_nodes[nodes_e]
    _, gamma = shape1d(0.0, nen)
    J, Jinv = jacobian1d(x_e, gamma, nen)
    dNdx = [g * Jinv for g in gamma]
    strain = sum(dNdx[i] * u_e[i] for i in range(nen))
    stress[e] = E * strain

# Visualisierung
plt.plot(x_nodes, u, label="Verschiebung")
plt.xlabel("x (m)")
plt.ylabel("u (m)")
plt.title("1D Finite Element Verschiebung")
plt.grid(True)
plt.legend()
plt.show()


# Frage 5 - Sicherheitsanalyse
F_min = 121000     # Mindestbruchkraft pro Seil (N)
sigma_max = np.max(stress)       # maximale Spannung im Seil
F_berechnet = sigma_max * A      # tatsächliche Zugkraft im Seil

print("\nSicherheitsprüfung:")

# Jetzt für beide Szenarien (2 und 3 Seile mit verschiedenen Faktoren)
for n_seile, sicherheitsfaktor in [(2, 16), (3, 12)]:
    F_erlaubt = (F_min * n_seile) / sicherheitsfaktor
    
    print(f"\n--- Analyse für {n_seile} Seile mit {sicherheitsfaktor}-fachem Sicherheitsfaktor ---")
    print(f"Erlaubte maximale Gesamtkraft: {F_erlaubt:.2f} N")
    print(f"Maximale berechnete Kraft im Seil: {abs(F_berechnet):.2f} N")
    print(f"Maximale Spannung im Seil: {abs(sigma_max / 1e6):.2f} MPa")

    if abs(F_berechnet) <= F_erlaubt:
        print("Sicherheit ist eingehalten.")
    else:
        print("Sicherheit ist NICHT eingehalten.")
