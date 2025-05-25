import numpy as np

import matplotlib.pyplot as plt





def shape1d(xi, nen):

    if nen == 2:  # Linear

        N = np.array([(1 - xi) / 2, (1 + xi) / 2])

        gamma = np.array([-0.5, 0.5])

    elif nen == 3:  # Quadratic

        N = np.array([

            xi * (xi - 1) / 2,

            (1 - xi ** 2),

            xi * (xi + 1) / 2

        ])

        gamma = np.array([

            (2 * xi - 1) / 2,

            -2 * xi,

            (2 * xi + 1) / 2

        ])

    else:

        raise ValueError("Only 2 or 3 node elements supported.")

    return N, gamma



def gauss1d(nqp):

    if nqp == 1:

        return np.array([0.0]), np.array([2.0])

    elif nqp == 2:

        xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])

        w = np.array([1.0, 1.0])

    elif nqp == 3:

        xi = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])

        w = np.array([5 / 9, 8 / 9, 5 / 9])

    else:

        raise ValueError("Only 1–3 Gauss points supported.")

    return xi, w



def jacobian1d(xe, gamma):

    J = sum(xe[i] * gamma[i] for i in range(len(xe)))

    if J <= 0:

        raise ValueError("Non-positive Jacobian!")

    return J, 1 / J



def body_force_vector(rho, g, A, xe, nen, nqp):

    xi_qp, w_qp = gauss1d(nqp)

    fe = np.zeros(nen)

    b = rho * g



    for i in range(nqp):

        xi, w = xi_qp[i], w_qp[i]

        N, gamma = shape1d(xi, nen)

        detJ, _ = jacobian1d(xe, gamma)

        fe += b * A * N * detJ * w

    return fe



def local_stiffness(E, A, xe, nen=2, nqp=2):

    xi_qp, w_qp = gauss1d(nqp)

    ke = np.zeros((nen, nen))



    for i in range(nqp):

        xi, w = xi_qp[i], w_qp[i]

        N, gamma = shape1d(xi, nen)

        detJ, invJ = jacobian1d(xe, gamma)

        B = invJ * gamma

        ke += E * A * np.outer(B, B) * detJ * w

    return ke



def assemble_system(coords, conn, E, A, rho, g, nen=2, nqp=2):

    n_nodes = len(coords)

    K = np.zeros((n_nodes, n_nodes))

    f = np.zeros(n_nodes)



    for e, nodes in enumerate(conn):

        xe = coords[nodes]

        ke = local_stiffness(E, A, xe, nen, nqp)

        fe = body_force_vector(rho, g, A, xe, nen, nqp)



        for i in range(nen):

            for j in range(nen):

                K[nodes[i], nodes[j]] += ke[i, j]

            f[nodes[i]] += fe[i]

    return K, f



# ---------- Boundary Condition Methods -----------



def apply_slicing(K, f, u_d, dof_d):

    n = K.shape[0]

    dof_f = np.setdiff1d(np.arange(n), dof_d)

    K_FF = K[np.ix_(dof_f, dof_f)]

    K_FD = K[np.ix_(dof_f, dof_d)]

    f_F = f[dof_f] - K_FD @ u_d



    u_F = np.linalg.solve(K_FF, f_F)

    u = np.zeros(n)

    u[dof_f] = u_F

    u[dof_d] = u_d

    f_react = K @ u - f

    return u, f_react



def apply_penalty(K, f, u_d, dof_d, penalty=1e13):

    n = K.shape[0]

    K_mod = K.copy()

    f_mod = f.copy()



    for i, d in enumerate(dof_d):

        K_mod[d, d] += penalty

        f_mod[d] += penalty * u_d[i]



    u = np.linalg.solve(K_mod, f_mod)

    f_react = K @ u - f

    return u, f_react



# --------------- Stress and Safety ---------------



def compute_stress(E, coords, conn, u, nen=2):

    stresses = []

    for nodes in conn:

        xe = coords[nodes]

        ue = u[nodes]

        _, gamma = shape1d(0, nen)

        detJ, invJ = jacobian1d(xe, gamma)

        B = invJ * gamma

        strain = B @ ue

        stress = E * strain

        stresses.append(stress)

    return np.array(stresses)



def check_safety(n_cables, F_min, total_mass, g):

    SF = 12 if n_cables == 3 else 16

    F_required = total_mass * g

    F_allowed = n_cables * F_min / SF



    print(f"\n--- Safety Check: {n_cables} Cables (SF = {SF}) ---")

    print(f"Required load       : {F_required:.1f} N")

    print(f"Permitted by SF     : {F_allowed:.1f} N")

    if F_required <= F_allowed:

        print("--- SAFE ---")

    else:

        print("--- UNSAFE ---")



# --------------- Simulation Runner ---------------



def main():

    # --- Physical Data ---

    L = 70.0

    m_car = 300.0

    m_payload = 630.0

    m_person = 75.0

    rho_lin = 86.1 / 100  # kg/m

    A = 89.9e-6

    E = 210e9

    F_min = 121e3

    g = 9.81

    n_cables = 3



    # --- FEM Mesh ---

    n_el = 10

    nen = 2

    n_nodes = n_el * (nen - 1) + 1

    coords = np.linspace(0, L, n_nodes)

    conn = [np.arange(i, i + nen) for i in range(0, len(coords) - 1, nen - 1)]



    # --- Method Toggle ---

    method = "slicing"  # or "penalty"



    # --- Total Load per Cable ---

    m_cable = rho_lin * L

    m_total = (m_payload + m_car + m_cable) / n_cables



    # Assemble system

    K, f_vol = assemble_system(coords, conn, E, A, rho_lin, g, nen)



    # Point load for payload + car

    f_ext = f_vol.copy()

    f_ext[-1] += (m_payload + m_car) * g / n_cables



    # Apply BC

    u_d = np.array([0.0])

    dof_d = np.array([0])

    if method == "slicing":

        u, f_react = apply_slicing(K, f_ext, u_d, dof_d)

    elif method == "penalty":

        u, f_react = apply_penalty(K, f_ext, u_d, dof_d)

    else:

        raise ValueError("Invalid method specified.")



    # Compute stress

    stresses = compute_stress(E, coords, conn, u, nen)



    # Output

    print(f"\nMax displacement: {np.max(np.abs(u)):.6f} m")

    print(f"Max stress      : {np.max(np.abs(stresses)) / 1e6:.2f} MPa")



    # Safety

    check_safety(n_cables=3, F_min=F_min, total_mass=m_payload + m_car + m_cable, g=g)

    check_safety(n_cables=2, F_min=F_min, total_mass=m_payload + m_car + m_cable, g=g)



    # Plot

    plt.plot(coords, u, label=f"Displacement ({method})")

    plt.xlabel("Position along cable [m]")

    plt.ylabel("Displacement [m]")

    plt.title("Cable Elongation")

    plt.grid(True)

    plt.legend()

    plt.show()



if __name__ == "__main__":

    main()
