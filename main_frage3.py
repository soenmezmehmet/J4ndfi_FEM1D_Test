#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Datei: main.py
# --------------------------
# Schritt 6: Hauptprozess + Analyse von Frage 3:
# Welche Auswirkung hat die Wahl der Zahl der Gauss-Punkte?

import numpy as np
import matplotlib.pyplot as plt  
from assemble import element_stiffness_force
from apply_bc import apply_bc_slicing

# === Festgelegte Modellparameter ===
L = 70.0           # Seillänge
E = 2e11           # Elastizitätsmodul (Pa)
A = 89.9e-6           # Querschnittsfläche (m²)
EA = E * A
rho = 7850         # Dichte (kg/m³)
g = 9.81
f_body = -rho * g  # Körperkraft durch Eigengewicht
nen = 2            # Anzahl Knoten pro Element (linear)
nel = 20           # Anzahl der Elemente

# === Vergleich: Verschiedene Anzahl von Gauß-Punkten ===
nqp_list = [1, 2, 3]  # Zu testende Gauß-Punkte

plt.figure()

for nqp in nqp_list:
    # Anzahl der Gesamtknoten
    nn = nel * (nen - 1) + 1
    x_nodes = np.linspace(0, L, nn)

    # Initialisierung globaler Matrizen
    K_global = np.zeros((nn, nn))
    f_global = np.zeros(nn)

    # --- Assemblierung ---
    for e in range(nel):
        # Lokale Knotennummern
        nodes_e = list(range(e * (nen - 1), e * (nen - 1) + nen))
        x_e = x_nodes[nodes_e]  # Koordinaten des aktuellen Elements

        # Lokale Steifigkeit und Kraft mit aktuellem nqp berechnen
        K_e, f_e = element_stiffness_force(EA, f_body, x_e, nen, nqp)

        # Beitrag zum globalen System addieren
        for i in range(nen):
            for j in range(nen):
                K_global[nodes_e[i], nodes_e[j]] += K_e[i, j]
            f_global[nodes_e[i]] += f_e[i]

    # --- Randbedingungen anwenden ---
    bc = [(0, 0.0)]  # Festlager bei x = 0
    K_mod, f_mod, free_dofs = apply_bc_slicing(K_global, f_global, bc)

    # --- Lösung des Gleichungssystems ---
    u = np.zeros(nn)
    u_free = np.linalg.solve(K_mod, f_mod)
    u[free_dofs] = u_free

    # --- Plot ---
    plt.plot(x_nodes, u, marker='o', label=f"{nqp} Gauss-Punkte")

# === Ergebnisvisualisierung ===
plt.xlabel("x (m)")
plt.ylabel("u(x) (m)")
plt.title("Einfluss der Anzahl der Gauß-Integrationspunkte auf die Verschiebung")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
