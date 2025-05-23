#% ========================================================================
# Numerische Implementierung der linearen FEM
#% TU Berlin
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

# disp_scaling = 1000
# toplot = True
# ############ Input ###############

# # Define coordinates as a column vector
# # Hint: row entries - global node number
# #       column entries - corresponding x coordinates
# # x = [x1 ; x2 ; x3 ; x4 ; x5]
# #
# x = torch.reshape(torch.linspace(0, 70, 11), [-1, 1])
# print("x: ", x)
# # Define connectivity list as a matrix
# # between local and global node numbers
# # Hint: No. of rows in the 'conn' matrix = No. of elements
# #       No. of columns in the 'conn' matrix = Element node numbering
# # conn  | Global list
# # --------------------
# # e = 1 | 1  3  2
# # e = 2 | 3  5  4
# #
# conn = torch.from_numpy(np.array([[1,  3,  2], [3,  5,  4], [5,  7,  6], [7,  9,  8], [9,  11,  10]]))

# # Number of quadrature points per Element, nqp
# nqp = 2

# # Boundary conditions
# drltDofs = torch.from_numpy(np.array([1])) # Global DOF numbers where Dirichlet DOF 's are prescribed
# freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])) # Global DOF numbers where displacement is unknown, Free DOF's
# u_d = torch.from_numpy(np.array([0.]) )# Value of the displacement at the prescribed nodes

# f_sur = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (300 + 75) * 9.81]))

# # Constant body force
# b = 7850 * 9.81

# # Material parameters for the two elements
# E = 2.1e11 * torch.ones(conn.size()[0], 1);
# area = 3 * 89.9e-6 * torch.ones(conn.size()[0], 1);  ##

# #% Fuer die Ausgabe: Verschiebungen um Faktor ueberhoehen
# scalingfactor = 1

# ############ Preprocessing ###############
# # Extract nnp, ndm and ndf from the vector 'x'
# # Hint: use MATLAB function size()
# nnp = #TODO
# print("nnp: ", nnp)
# ndm = #TODO
# ndf = #TODO

# # Extract nel and nen from the matrix 'conn'
# # Hint: use MATLAB function size()
# #TODO
# #TODO
# #############Solver#############
# # Initialisation of global vectors and matrix
# u = #TODO
# K = #TODO
# fext = #TODO
# fvol = #TODO
# frea = #TODO

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


# """
# % Abbildung zum und vom Masterelement:
# % Entspricht der "Jacobian"-Matrix
# % xe: Globale Koordinaten der Knoten eines Elementes
# % gamma: Ableitungen der Formfunktionen (Shape Functions) am Masterelement
# % nen: Number of Element Nodes
# % Ausgabe: Determinante der Jacobian-Matrix und Inverse der Jacobian
# """
# def jacobian1d (xe, gamma, nen):

#     #TODO
#     if detJq <= 0:
#         raise ("Error: detJq = ", detJq, "<= 0")

#     invJq = 1/detJq

#     return (detJq, invJq)


# ########### SOLVER  ############

# #Initialisation of global vectors and matrix
# u = torch.zeros(nnp*ndf, 1)
# K = torch.zeros(nnp*ndf, nnp*ndf)
# fext = torch.zeros(nnp*ndf, 1)
# fvol = torch.zeros(nnp*ndf, 1)
# frea = torch.zeros(nnp*ndf, 1)

# ################ Create stiffness matrix and fvol ############
# for e in range(nel):

#     #Coordinates of the element nodes
#     # Hint: to be extracted from the global coordinate list 'x'
#     #       considering the 'conn' matrix
#     xe = #TODO
#     (xi, w8) = #TODO

#     # Call the coordinates and weights of the gauss points
#     # Hint: input parameters - nqp
#     #            output parameters - xi, w8
#     #            function name - gauss1d
#     #TODO

#     # Loop over the gauss points, Summation over all gauss points 'q'
#     for q in range(nqp):
#         #% Call the shape functions and its derivatives
#         # Hint: input parameters - xi(q), nen
#         #       output parameters - N, gamma
#         #       function name - shape1d
#         #TODO

#         #% Determinant of Jacobian and the inverse of Jacobian
#         # at the quadrature points q
#         # Hint: For this 1d-case the Jacobian is a scalar of 1x1
#         #       and its inverse also a scalar
#         #TODO

#         #Gradient of the shape functions wrt. to x
#         G = gamma*invJq

#         #Loop over the number of nodes A
#         for A in range(nen):
#             # Volume force contribution
#             fvol[conn[e, A]-1, 0] += #TODO

#             # Loop over the number of nodes A
#             for B in range(nen):
#                 K[conn[e, A]-1, conn[e, B]-1] += #TODO

# ##### Include BC, solve the linear system K u = f ######
# # Calculate nodal displacements => solve linear system
# # Use "array slicing" or penalty method
# solve_K = K
# #print("solve_K:  ", solve_K)
# for i in range(drltDofs.size()[0]):
#     solve_K[drltDofs[i]-1, drltDofs[i]-1] = #TODO


# u [:, 0] = #TODO
# u[drltDofs-1, 0] = #TODO

# ###### Process results: Calculate output quantities  #######
# # Compute the reaction forces at the Dirichlet boundary
# frea[drltDofs-1, 0] = #TODO
# #Compute the total force vector
# fext = #TODO

# # Calculate strains and stresses
# eps = torch.zeros(nel*nqp, 1)
# sigma = torch.zeros(nel*nqp, 1)
# x_eps = torch.zeros(nel*nqp, 1)

# # for all elements
# for e in range(nel):

#     #Coordinates of the element nodes
#     # Hint: to be extracted from the global coordinate list 'x'
#     #       considering the 'conn' matrix
#     xe = #TODO
#     (xi, w8) = #TODO

#     # for all Gauss points
#     for q in range(nqp):
#         # evaluate shape functions (as above)
#         # evaluate jacobian

#         # Gradient of the shape functions wrt. to x
#         G = gamma*invJq

#         # Use derivatives of shape functions, G, to calculate eps as spatial derivative of u
#         eps[e*nqp + q] = torch.tensordot(torch.transpose(u[conn[e, :]-1], 0, 1), G[:], dims=[[1],[0]])
#         # Calculate stresses from strains and Young's modulus (i.e. apply Hooke's law)
#         sigma[e * nqp + q] = E[e, 0]*eps[e*nqp + q]
#         # create x-axis vector to plot eps, sigma in the Gauss points (not at the nodes!)
#         x_eps[e*nqp + q] = torch.tensordot(xe, N[:])

# print("u: ", u)
# ###### Post-processing/ plots ########
# plt.subplot(4,1,1)
# plt.plot(x, torch.zeros_like(x), 'ko-')
# plt.plot(x[drltDofs], -0.02*torch.ones_like(drltDofs), 'g^')
# plt.plot(x+scalingfactor*u, torch.zeros_like(x), 'o-')
# plt.plot(0, 1, 'k-')

# plt.subplot(4, 1, 2)
# plt.plot(x, u, 'x-')
# plt.legend("u")

# plt.subplot(4,1,3)
# plt.plot(x, fext, 'ro')
# plt.plot(x, fvol, 'cx')
# plt.plot(x, frea, 'bx')
# plt.legend(["$f_{ext}$", "$f_{vol}$", "$f_{rea}$"])

# plt.subplot(4,1,4)
# plt.plot(x_eps, sigma, 'x-')

# plt.show()
