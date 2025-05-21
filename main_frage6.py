#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Datei: main.py
# Schritt 6: Hauptprozess 
# Frage 6: Wie groß ist der Anteil der Längung durch Eigengewicht, Aufzugskabine und einzelne Person?

import numpy as np
import matplotlib.pyplot as plt
from assemble import element_stiffness_force
from apply_bc import apply_bc_slicing

# Parameterdefinition
L = 70.0                      # Seillänge (m)
E = 2e11                     # Elastizitätsmodul (Pa)
A = 89.9e-6                  # Metallischer Querschnitt (m²)
EA = E * A
rho = 7850                  # Dichte des Seils (kg/m³)
g = 9.81
f_body = -rho * g           # Körperkraft pro m³ (Eigengewicht des Seils)

# Zusatzmassen
m_kabine = 300.0            # Masse der Aufzugskabine (kg)
m_person = 75.0             # Masse einer Person (kg)

nen = 2                     # Anzahl der Knoten pro Element
nel = 10                    # Anzahl der Elemente
nqp = 2                     # Anzahl der Gauß-Integrationspunkte

nn = nel * (nen - 1) + 1
x_nodes = np.linspace(0, L, nn)

def solve_fem_with_force(f_body):
    K_global = np.zeros((nn, nn))
    f_global = np.zeros(nn)

    for e in range(nel):
        nodes_e = list(range(e * (nen - 1), e * (nen - 1) + nen))
        x_e = x_nodes[nodes_e]
        K_e, f_e = element_stiffness_force(EA, f_body, x_e, nen, nqp)

        for i in range(nen):
            for j in range(nen):
                K_global[nodes_e[i], nodes_e[j]] += K_e[i, j]
            f_global[nodes_e[i]] += f_e[i]

    bc = [(0, 0.0)]
    K_mod, f_mod, free_dofs = apply_bc_slicing(K_global, f_global, bc)

    u = np.zeros(nn)
    u_free = np.linalg.solve(K_mod, f_mod)
    u[free_dofs] = u_free
    return u

# Einzellösungen berechnen
u_eigengewicht = solve_fem_with_force(f_body)
kabine_body_force = -m_kabine * g / L / A
person_body_force = -m_person * g / L / A
u_kabine = solve_fem_with_force(kabine_body_force)
u_person = solve_fem_with_force(person_body_force)

total_u = u_eigengewicht + u_kabine + u_person

def calc_elongation(u):
    return u[-1] - u[0]

# --- Ergebnisse: Längung ---
L_e = calc_elongation(u_eigengewicht)
L_k = calc_elongation(u_kabine)
L_p = calc_elongation(u_person)
L_total = calc_elongation(total_u)

# --- Ausgabe ---
print("\n[Frage 6 - Längungsanteile (Meter)]")
print(f"Längung durch Eigengewicht:  {abs(L_e):.6f} m")
print(f"Längung durch Aufzugskabine: {abs(L_k):.6f} m")
print(f"Längung durch eine Person:   {abs(L_p):.6f} m")
print(f"Gesamtlängung:               {abs(L_total):.6f} m")

print("\n[Prozentuale Anteile an der Gesamtlängung]")
print(f"Eigengewicht:  {100 * L_e / L_total:.2f} %")
print(f"Kabine:        {100 * L_k / L_total:.2f} %")
print(f"Person:        {100 * L_p / L_total:.2f} %")


# Visualisierung aller Beiträge
plt.plot(x_nodes, u_eigengewicht, label="Eigengewicht")
plt.plot(x_nodes, u_kabine, label="Kabine")
plt.plot(x_nodes, u_person, label="Person")
plt.plot(x_nodes, total_u, label="Gesamt", linestyle='--')
plt.xlabel("x (m)")
plt.ylabel("u(x) (m)")
plt.title("Längungsanteile entlang des Seils")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
