#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Datei: main.py
# --------------------------
# Schritt 6: Hauptprozess (Modell definieren, System aufbauen, Randbedingungen anwenden, lösen, Nachbearbeitung)
# Erweiterung zur Beantwortung der Frage 2:
# Vergleich zwischen linearen (nen=2) und quadratischen (nen=3) Formfunktionen

import numpy as np
import matplotlib.pyplot as plt
from assemble import element_stiffness_force
from apply_bc import apply_bc_slicing

# Parameterdefinition
L = 70.0           # Seillänge
E = 2e11           # Elastizitätsmodul (Pa)
A = 89.9e-6           # Querschnittsfläche (m²)
EA = E * A
rho = 7850         # Dichte (kg/m³)
g = 9.81
f_body = -rho * g  # Körperkraft (Eigengewicht)
nqp = 3            # Anzahl der Gauß-Integrationspunkte (nötig für nen=3)
nel = 3           # Anzahl der Elemente

# Vergleich für nen=2 (linear) und nen=3 (quadratisch)
nen_list = [2, 3]
plt.figure()

for nen in nen_list:
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


    # Visualisierung: Verschiebungskurven vergleichen
    label = f"{'Linear' if nen == 2 else 'Quadratisch'} (nen={nen})"
    plt.plot(x_nodes, u, marker='o', label=label)

# Gesamtdiagramm
plt.xlabel("x (m)")
plt.ylabel("u(x) (m)")
plt.title("Vergleich: Lineare vs. Quadratische Formfunktionen")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Ergebnisanalyse:
# Durch Vergleich der Verschiebungskurven für nen=2 und nen=3 sieht man:
# - Quadratische Formfunktionen (nen=3) approximieren die Krümmung deutlich besser
# - Bei gleicher Elementanzahl (z.B. nel=10) ist die quadratische Lösung glatter und näher an der exakten Lösung
