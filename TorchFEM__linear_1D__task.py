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

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 1000
toplot = True
############ Input ###############

# Define coordinates as a column vector
# Hint: row entries - global node number
#       column entries - corresponding x coordinates
# x = [x1 ; x2 ; x3 ; x4 ; x5]
#
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
conn = torch.from_numpy(np.array([[1,  3,  2], [3,  5,  4], [5,  7,  6], [7,  9,  8], [9,  11,  10]]))

# Number of quadrature points per Element, nqp
nqp = 2

# Boundary conditions
drltDofs = torch.from_numpy(np.array([1])) # Global DOF numbers where Dirichlet DOF 's are prescribed
freeDofs = torch.from_numpy(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])) # Global DOF numbers where displacement is unknown, Free DOF's
u_d = torch.from_numpy(np.array([0.]) )# Value of the displacement at the prescribed nodes

f_sur = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (300 + 75) * 9.81]))

# Constant body force
b = 7850 * 9.81

# Material parameters for the two elements
E = 2.1e11 * torch.ones(conn.size()[0], 1);
area = 3 * 89.9e-6 * torch.ones(conn.size()[0], 1);  ##

#% Fuer die Ausgabe: Verschiebungen um Faktor ueberhoehen
scalingfactor = 1

############ Preprocessing ###############
# Extract nnp, ndm and ndf from the vector 'x'
# Hint: use MATLAB function size()
nnp = #TODO
print("nnp: ", nnp)
ndm = #TODO
ndf = #TODO

# Extract nel and nen from the matrix 'conn'
# Hint: use MATLAB function size()
#TODO
#TODO
#############Solver#############
# Initialisation of global vectors and matrix
u = #TODO
K = #TODO
fext = #TODO
fvol = #TODO
frea = #TODO

"""
% Parameter fuer die Gauss-Quadratur
%% xi sind die Stellen, an denen die Funktion ausgewertet wird
%% w8 die "weights" -- zugehörige Gewichte in der Summenformel
%% Eingabe: nqp, die Anzahl der Gauss-Quadratur-Punkte (nicht Element-Knoten!!)
"""
def gauss1d (nqp):
    #TODO
    return (xi, w8)


"""
% Auswertung der Formfunktionen (Lagrange-Polynome) [N] und ihrer Ableitungen [gamma]
%% an der (elementlokalen) Positionen [xi] bei [nen] Knoten im Element
%% für das Masterelement
"""
def shape1d(xi, nen):
    N = torch.zeros(nen, 1)
    gamma = torch.zeros(nen, 1)

    if nen == 2:
        #TODO
    elif nen == 3:
        #TODO
    else:
        raise("unknown #nen")

    return (N, gamma)

"""
% Abbildung zum und vom Masterelement:
% Entspricht der "Jacobian"-Matrix
% xe: Globale Koordinaten der Knoten eines Elementes
% gamma: Ableitungen der Formfunktionen (Shape Functions) am Masterelement
% nen: Number of Element Nodes
% Ausgabe: Determinante der Jacobian-Matrix und Inverse der Jacobian
"""
def jacobian1d (xe, gamma, nen):

    #TODO
    if detJq <= 0:
        raise ("Error: detJq = ", detJq, "<= 0")

    invJq = 1/detJq

    return (detJq, invJq)


########### SOLVER  ############

#Initialisation of global vectors and matrix
u = torch.zeros(nnp*ndf, 1)
K = torch.zeros(nnp*ndf, nnp*ndf)
fext = torch.zeros(nnp*ndf, 1)
fvol = torch.zeros(nnp*ndf, 1)
frea = torch.zeros(nnp*ndf, 1)

################ Create stiffness matrix and fvol ############
for e in range(nel):

    #Coordinates of the element nodes
    # Hint: to be extracted from the global coordinate list 'x'
    #       considering the 'conn' matrix
    xe = #TODO
    (xi, w8) = #TODO

    # Call the coordinates and weights of the gauss points
    # Hint: input parameters - nqp
    #            output parameters - xi, w8
    #            function name - gauss1d
    #TODO

    # Loop over the gauss points, Summation over all gauss points 'q'
    for q in range(nqp):
        #% Call the shape functions and its derivatives
        # Hint: input parameters - xi(q), nen
        #       output parameters - N, gamma
        #       function name - shape1d
        #TODO

        #% Determinant of Jacobian and the inverse of Jacobian
        # at the quadrature points q
        # Hint: For this 1d-case the Jacobian is a scalar of 1x1
        #       and its inverse also a scalar
        #TODO

        #Gradient of the shape functions wrt. to x
        G = gamma*invJq

        #Loop over the number of nodes A
        for A in range(nen):
            # Volume force contribution
            fvol[conn[e, A]-1, 0] += #TODO

            # Loop over the number of nodes A
            for B in range(nen):
                K[conn[e, A]-1, conn[e, B]-1] += #TODO

##### Include BC, solve the linear system K u = f ######
# Calculate nodal displacements => solve linear system
# Use "array slicing" or penalty method
solve_K = K
#print("solve_K:  ", solve_K)
for i in range(drltDofs.size()[0]):
    solve_K[drltDofs[i]-1, drltDofs[i]-1] = #TODO


u [:, 0] = #TODO
u[drltDofs-1, 0] = #TODO

###### Process results: Calculate output quantities  #######
# Compute the reaction forces at the Dirichlet boundary
frea[drltDofs-1, 0] = #TODO
#Compute the total force vector
fext = #TODO

# Calculate strains and stresses
eps = torch.zeros(nel*nqp, 1)
sigma = torch.zeros(nel*nqp, 1)
x_eps = torch.zeros(nel*nqp, 1)

# for all elements
for e in range(nel):

    #Coordinates of the element nodes
    # Hint: to be extracted from the global coordinate list 'x'
    #       considering the 'conn' matrix
    xe = #TODO
    (xi, w8) = #TODO

    # for all Gauss points
    for q in range(nqp):
        # evaluate shape functions (as above)
        # evaluate jacobian

        # Gradient of the shape functions wrt. to x
        G = gamma*invJq

        # Use derivatives of shape functions, G, to calculate eps as spatial derivative of u
        eps[e*nqp + q] = torch.tensordot(torch.transpose(u[conn[e, :]-1], 0, 1), G[:], dims=[[1],[0]])
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
