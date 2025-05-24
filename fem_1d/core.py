# ========================================================================
# Numerische Implementierung der linearen FEM
# TU Berlin
# Institute für Mechanik
# ========================================================================
# Finite Element Code for 1d elements
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from typing import Tuple

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True
############ Input ###############

# Define coordinates as a column vector
# Hint: row entries - global node number
#       column entries - corresponding x coordinates
# x = [x1 ; x2 ; x3 ; x4 ; x5]
x = torch.reshape(torch.linspace(0, 70, 11), [-1, 1])
print("x: ", x)

# Define connectivity list as a matrix
# between local and global node numbers
# Hint: No. of rows in the 'conn' matrix = No. of elements
#       No. of columns in the 'conn' matrix = Element node numbering
# conn  | Global list
# --------------------
# e = 1 | 1  3  2
# e = 2 | 3  5  4
#
conn_reorder = True  # used to reorder the connectivity list to [1, 2, 3] instead of [1, 3, 2]
conn = torch.from_numpy(np.array([[1,  3,  2], [3,  5,  4], [5,  7,  6], [7,  9,  8], [9,  11,  10]]))
if conn_reorder:
    conn = conn[:, [0, 2, 1]]


# Number of quadrature points per Element, nqp
nqp = 2

# Boundary conditions
drltDofs = torch.from_numpy(np.array([1])) # Global DOF numbers where Dirichlet DOF 's are prescribed
freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])) # Global DOF numbers where displacement is unknown, Free DOF's
u_d = torch.from_numpy(np.array([0.]) )# Value of the displacement at the prescribed nodes

# define force on last node
f_sur = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (300 + 75) * 9.81]))

# Constant body force (constant volume force)
b = 7850 * 9.81

# Material parameters for the two elements
E = 2.1e11 * torch.ones(conn.size()[0], 1)
area = 3 * 89.9e-6 * torch.ones(conn.size()[0], 1) # assign cross-sectional area to all elements

# Fuer die Ausgabe: Verschiebungen um Faktor ueberhoehen
scalingfactor = 1

############ Preprocessing ###############
# Extract nnp, ndm and ndf from the vector 'x'
# Hint: use MATLAB function size()
nnp = x.size()[0] # number of nodal points (nodes)
print("nnp: ", nnp)
ndm = x.size()[1] # number of dimensions (1D)
ndf = 1 # number of degrees of freedom (1D)

# Extract nel and nen from the matrix 'conn'
nel = conn.size()[0]  # number of elements
nen = conn.size()[1]  # number of element nodes (2 or 3)
# NOTE: redundant? already defined as nqp


def gauss1d(nqp: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns Gauss quadrature points and weights for numerical integration over [-1, 1].

    Parameters
    ----------
    nqp : int
        Number of quadrature points (supported: 1, 2, or 3).

    Returns
    -------
    xi : torch.Tensor
        Quadrature points (locations ξ_i where the function is evaluated), shape (nqp,).
    
    w8 : torch.Tensor
        Corresponding weights α_i for the quadrature formula, shape (nqp,).

    Raises
    ------
    ValueError
        If nqp is not 1, 2, or 3.
    """
    if nqp == 1:
        xi = torch.tensor([0.0])
        w8 = torch.tensor([2.0])
    elif nqp == 2:
        xi = torch.tensor([-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)])
        w8 = torch.tensor([1.0, 1.0])
    elif nqp == 3:
        xi = torch.tensor([-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)])
        w8 = torch.tensor([5.0 / 9, 8.0 / 9, 5.0 / 9])
    else:
        raise ValueError("Invalid number of quadrature points. Supported values are 1, 2, or 3.")
    return xi, w8


def shape1d(xi: float, nen: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the Lagrange shape functions and their derivatives at a local coordinate ξ.

    Parameters
    ----------
    xi : float
        Local coordinate within the reference (master) element, ξ ∈ [-1, 1].

    nen : int
        Number of nodes in the element.
        - 2 nodes → linear shape functions
        - 3 nodes → quadratic shape functions

    Returns
    -------
    N : torch.Tensor
        Column vector of shape function values at ξ, shape (nen, 1).

    gamma : torch.Tensor
        Column vector of shape function derivatives with respect to ξ, shape (nen, 1).

    Notes
    -----
    - This function assumes `nen` is either 2 or 3. Input validity should be handled by the caller.
    - For efficiency, caching of intermediate factors could reduce repeated computation,
      but increases memory use slightly.
    """
    # define element nodes (in master element coordinates)
    xi_nodes = [-1, 1] if nen == 2 else [-1, 0, 1]

    # define output arrays
    N = torch.zeros(nen, 1)
    gamma = torch.zeros(nen, 1)

    for i in range(nen):  # iterate i over all nodes
        L = 1.0
        dL = 0.0
        for j in range(nen):  # for shape function N_i
            if i != j:
                L *= (xi - xi_nodes[j]) / (xi_nodes[i] - xi_nodes[j])
        N[i] = L

        for j in range(nen):  # for derivative dN_i/dξ
            if i != j:
                prod = 1.0
                for k in range(nen):
                    if k != i and k != j:
                        prod *= (xi - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k])
                dL += prod / (xi_nodes[i] - xi_nodes[j])

        gamma[i] = dL

    return N, gamma


def jacobian1d(xe: torch.Tensor, gamma: torch.Tensor, nen: int) -> Tuple[float, float]:
    """
    Computes the Jacobian determinant and its inverse for a 1D element.

    Parameters
    ----------
    xe : torch.Tensor
        Global coordinates of the element's nodes, shape (nen,) or (nen, 1).
    
    gamma : torch.Tensor
        Derivatives of the shape functions w.r.t. the local coordinate ξ, shape (nen,) or (nen, 1).
    
    nen : int
        Number of nodes in the element (2 for linear, 3 for quadratic).

    Returns
    -------
    detJq : float
        Determinant of the Jacobian matrix (scalar in 1D).
    
    invJq : float
        Inverse of the Jacobian determinant.

    Raises
    ------
    ValueError
        If the determinant is non-positive or nen is unsupported.
    """
    if nen not in (2, 3):
        raise ValueError("Invalid number of element nodes. Supported values are 2 or 3.")

    # flatten both tensors in case they're (nen, 1)
    xe_flat = xe.view(-1)
    gamma_flat = gamma.view(-1)
    # det(J) = sum_A^nen (gamma_A * xe_A)
    detJq = torch.dot(gamma_flat, xe_flat).item()  # scalar float

    if detJq <= 0:
        raise ValueError("Jacobian determinant is non-positive. Check element connectivity or node order.")

    invJq = 1.0 / detJq
    return detJq, invJq


#############Solver#############
# Initialisation of global vectors and matrix
u = torch.zeros(nnp * ndf, 1)     # displacement vector
K = torch.zeros(nnp * ndf, nnp * ndf)  # global stiffness matrix
fext = torch.zeros(nnp * ndf, 1)  # total external force vector
fvol = torch.zeros(nnp * ndf, 1)  # body force vector
frea = torch.zeros(nnp * ndf, 1)  # reaction forces (Dirichlet nodes)

################ Create stiffness matrix and fvol ############
for e in range(nel):  # loop over all elements
    # get global coordinates of the element nodes
    xe = x[conn[e, :] - 1]  # shape: (nen, 1)
    # NOTE: conn[e, :] - 1 should be cached

    # call the coordinates and weights of the gauss points
    xi, w8 = gauss1d(nqp)  # shape: (nqp,)

    for q in range(nqp):  # loop over all gauss points
        xi_q = xi[q].item()
        wq = w8[q].item()
        # call the shape function and its derivatives
        N, gamma = shape1d(xi_q, nen)  # shape: (nen, 1)

        # calculate the Jacobian and its inverse
        detJq, invJq = jacobian1d(xe, gamma, nen)  # floats

        # gradient of the shape functions wrt. to x
        G = gamma * invJq

        for A in range(nen):  # loop over the number of nodes A
            # add volume force contribution (body force)
            fvol[conn[e, A]-1, 0] = fvol[conn[e, A]-1, 0] + N[A] * b * area[e] * detJq * wq

            for B in range(nen):  # loop over the number of nodes A
                K[conn[e, A]-1, conn[e, B]-1] = K[conn[e, A]-1, conn[e, B]-1] + E[e] * area[e] * G[A] * G[B] * detJq * wq

# combine the force vectors
fext = f_sur.reshape(-1, 1) + fvol
# clone the global stiffness matrix
solve_K = K.clone()
# apply Dirichlet boundary conditions
for i in range(drltDofs.size()[0]):
    solve_K[drltDofs[i]-1, drltDofs[i]-1] = 1e30  # penalty method

# solve the linear system
u = torch.linalg.solve(solve_K, fext)
# enforce exact values at Dirichlet DOFs
u[drltDofs-1, 0] = u_d

###### Process results: Calculate output quantities  #######
# Compute the reaction forces at the Dirichlet boundary
frea = torch.matmul(K, u) - fext
frea[drltDofs-1, 0] = (K @ u - fext)[drltDofs - 1, 0]
#Compute the total force vector
fext = fvol + f_sur.reshape(-1, 1)

# Calculate strains and stresses
eps = torch.zeros(nel*nqp, 1)
sigma = torch.zeros(nel*nqp, 1)
x_eps = torch.zeros(nel*nqp, 1)

for e in range(nel):  # loop over all elements

    # get global coordinates of the element nodes
    xe = x[conn[e, :] - 1]  # (nen, 1)
    (xi, w8) = gauss1d(nqp)  # (nqp,)

    # for all Gauss points
    for q in range(nqp):
        N, gamma = shape1d(xi[q].item(), nen)
        detJq, invJq = jacobian1d(xe, gamma, nen)
        G = gamma*invJq

        # Use derivatives of shape functions, G, to calculate eps as spatial derivative of u
        eps[e*nqp + q] = torch.tensordot(torch.transpose(u[conn[e, :]-1], 0, 1), G[:], dims=[[1], [0]])
        # Calculate stresses from strains and Young's modulus (i.e. apply Hooke's law)
        sigma[e * nqp + q] = E[e, 0]*eps[e*nqp + q]
        # create x-axis vector to plot eps, sigma in the Gauss points (not at the nodes!)
        x_eps[e*nqp + q] = torch.tensordot(xe, N[:])

print("u: ", u)
###### Post-processing/ plots ########
plt.subplot(4,1,1)
plt.plot(x, torch.zeros_like(x), 'ko-')
plt.plot(x[drltDofs], -0.02*torch.ones_like(drltDofs), 'g^')
plt.plot(x+scalingfactor*u, torch.zeros_like(x), 'o-')
plt.plot(0, 1, 'k-')

plt.subplot(4, 1, 2)
plt.plot(x, u, 'x-')
plt.legend("u")

plt.subplot(4,1,3)
plt.plot(x, fext, 'ro')
plt.plot(x, fvol, 'cx')
plt.plot(x, frea, 'bx')
plt.legend(["$f_{ext}$", "$f_{vol}$", "$f_{rea}$"])

plt.subplot(4,1,4)
plt.plot(x_eps, sigma, 'x-')

plt.show()
