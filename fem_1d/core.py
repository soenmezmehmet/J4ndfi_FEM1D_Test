"""
This is a simple implementation of a 1d FEM solver based on
a code template by Stefan Hildebrand (TU Berlin, Institute of Mechanics)
"""
import math
from typing import Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


@dataclass
class MaterialProperties:
    """
    Material properties for the 1D FEM model.

    Attributes
    ----------
    E : torch.Tensor
        Young's modulus per element.
    area : torch.Tensor
        Cross-sectional area per element.
    b : float
        Body force per unit volume.
    """
    E: torch.Tensor
    area: torch.Tensor
    b: float


@dataclass
class BoundaryConditions:
    """
    Boundary conditions for the 1D FEM model.

    Attributes
    ----------
    u_d : torch.Tensor
        Dirichlet boundary condition values.
    drltDofs : torch.Tensor
        Indices of prescribed degrees of freedom (1-based).
    f_sur : torch.Tensor
        Surface force vector.
    """
    u_d: torch.Tensor
    drlt_dofs: torch.Tensor
    f_sur: torch.Tensor


class Fem1D:
    """
    A class-based implementation of a 1D finite element method (FEM) solver
    for axial deformation in bars or cables using Lagrange shape functions
    with linear or quadratic elements.

    Attributes
    ----------
    x : torch.Tensor
        Node coordinates.
    conn : torch.Tensor
        Connectivity matrix (element-to-node mapping).
    nqp : int
        Number of Gauss quadrature points.
    num_threads : int
        Number of threads used for computation.
    """

    def __init__(
        self,
        x: torch.Tensor,
        conn: torch.Tensor,
        material: MaterialProperties,
        bc: BoundaryConditions,
        nqp: int = 2,
        num_threads: int = 4
    ):
        """
        Initialize a 1D FEM model with material and boundary conditions.
        """
        # global settings
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(num_threads)
        # input parameters
        self.x = x.view(-1, 1)
        self.conn = conn
        self.material = material
        self.bc = bc
        self.nqp = nqp
        # derived parameters
        self.shape = {
            "nnp": x.size(0),  # number of nodal points
            "ndf": 1,  # number of degrees of freedom per node (1D problem)
            "ndm": 1,  # number of dimensions (1D problem)
            "nel": conn.size(0),  # number of elements
            "nen": conn.size(1)  # number of nodes per element (2 or 3)
        }
        # solver initialization
        self.K = torch.zeros(self.shape["nnp"], self.shape["nnp"])  # global stiffness matrix
        self.u = torch.zeros(self.shape["nnp"], 1)  # displacement vector
        self.fext = torch.zeros(self.shape["nnp"], 1)  # total external force vector
        self.fvol = torch.zeros(self.shape["nnp"], 1)  # body force vector
        self.frea = torch.zeros(self.shape["nnp"], 1)  # reaction forces (Dirichlet nodes)
        # initialize result tensors
        self.results = {
            "eps": torch.zeros(self.shape["nel"] * self.nqp, 1),  # strains at Gauss points
            "sigma": torch.zeros(self.shape["nel"] * self.nqp, 1),  # stresses at Gauss points
            "x_eps": torch.zeros(self.shape["nel"] * self.nqp, 1)  # spatial locations at Gauss points
        }

    def gauss1d(self, nqp: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return Gauss-Legendre quadrature points and weights for [-1, 1].

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

    def shape1d(self, xi: float, nen: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Lagrange shape functions and derivatives at given xi.

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

    def jacobian1d(self, xe: torch.Tensor, gamma: torch.Tensor, nen: int) -> Tuple[float, float]:
        """
        Compute 1D Jacobian determinant and its inverse from node positions.

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
        det_jq = torch.dot(gamma_flat, xe_flat).item()

        if det_jq <= 0:
            raise ValueError("Jacobian determinant is non-positive. Check element connectivity or node order.")

        inv_jq = 1.0 / det_jq
        return det_jq, inv_jq

    def preprocess(self) -> None:
        """
        Assemble the global stiffness matrix and external force vectors.
        """
        for e in range(self.shape["nel"]):  # loop over all elements
            # get global coordinates of the element nodes
            e_mask = self.conn[e, :] - 1
            xe = self.x[e_mask]  # shape: (nen, 1)

            # call the coordinates and weights of the Gauss points
            xi, w8 = self.gauss1d(self.nqp)

            for q in range(self.nqp):  # loop over all Gauss points
                xi_q = xi[q].item()
                wq = w8[q].item()
                # call the shape function and its derivatives
                N, gamma = self.shape1d(xi_q, self.shape["nen"])  # shape: (nen, 1)

                # calculate the Jacobian and its inverse
                det_jq, inv_jq = self.jacobian1d(xe, gamma, self.shape["nen"])

                # gradient of the shape functions wrt. to x
                # G = dN/dx = dN/dξ * dξ/dx = γ * invJq	
                G = gamma * inv_jq

                for A in range(self.shape["nen"]):  # loop over the number of nodes A
                    # add volume force contribution (body force)
                    a = e_mask[A]
                    self.fvol[e_mask[A]] += N[A] * self.material.b * self.material.area[e] * det_jq * wq

                    for B in range(self.shape["nen"]):  # loop over the number of nodes B
                        b = e_mask[B]
                        self.K[a, b] = self.K[a, b] + self.material.E[e] * self.material.area[e] * G[A] * G[B] * det_jq * wq

        # combine the force vectors
        self.fext = self.fvol + self.bc.f_sur.reshape(-1, 1)

    def solve(self) -> None:
        """
        Apply Dirichlet boundary conditions and solve the system.
        """
        solve_K = self.K.clone()
        # apply Dirichlet boundary conditions using penalty method
        for i in range(self.bc.drlt_dofs.size(0)):
            solve_K[self.bc.drlt_dofs[i] - 1, self.bc.drlt_dofs[i] - 1] = 1e30
            self.fext[self.bc.drlt_dofs[i] - 1, 0] = 1e30 * self.bc.u_d[i]

        # solve the linear system
        self.u = torch.linalg.solve(solve_K, self.fext)
        # enforce exact values at Dirichlet DOFs
        self.u[self.bc.drlt_dofs - 1] = self.bc.u_d

    def postprocess(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute strains, stresses, and spatial locations at Gauss points.
        """
        # calculate reaction forces at Dirichlet boundary conditions
        self.frea = self.K @ self.u - self.fext

        for e in range(self.shape["nel"]):  # loop over all elements
            # get global coordinates of the element nodes
            xe = self.x[self.conn[e, :] - 1]
            xi, _ = self.gauss1d(self.nqp)

            # for all Gauss points
            for q in range(self.nqp):
                N, gamma = self.shape1d(xi[q].item(), self.shape["nen"])
                _, inv_jq = self.jacobian1d(xe, gamma, self.shape["nen"])
                G = gamma * inv_jq

                # use derivatives of shape functions, G, to calculate eps as spatial derivative of u
                self.results["eps"][e * self.nqp + q] = torch.tensordot(self.u[self.conn[e, :] - 1].T, G, dims=[[1], [0]])
                # calculate stresses from strains and Young's modulus (i.e. apply Hooke's law)
                self.results["sigma"][e * self.nqp + q] = self.material.E[e, 0] * self.results["eps"][e * self.nqp + q]
                # create x-axis vector to plot eps, sigma in the Gauss points (not at the nodes!)
                self.results["x_eps"][e * self.nqp + q] = torch.tensordot(xe, N, dims=[[0], [0]])

        return self.results["eps"], self.results["sigma"], self.results["x_eps"]

    def report(self) -> None:
        """
        Print a summary of the FEM simulation results.
        """
        print(
                f"""
    ===== FEM Simulation Summary =====
    Displacement @ node {self.bc.drlt_dofs.item()}: {self.u[self.bc.drlt_dofs - 1].item():.4e} m
    Max displacement         : {torch.max(torch.abs(self.u)).item():.4e} m
    Max stress (σ)           : {torch.max(torch.abs(self.results["sigma"])).item():.4e} Pa
    Reaction force           : {self.frea[self.bc.drlt_dofs - 1].item():.2f} N
    ==================================
    """
            )

    def plot(self, scalingfactor: float = None, filepath: str = None) -> None:
        """
        Visualize FEM results including:
        - deformed shape,
        - nodal displacements,
        - force vectors,
        - stress distribution at Gauss points.

        Parameters
        ----------
        scalingfactor : float, optional
            Scaling factor for displacements in the deformed shape plot.
            If None, it is computed automatically to make deformations visible.
        """
        # Compute automatic scaling factor if not set
        if scalingfactor is None:
            scalingfactor = 0.05 * (self.x[-1] - self.x[0]) / self.u.abs().max()

        # Create subplots
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[0.45, 0.45, 0.45, 1])
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
        fig.suptitle("1D FEM Simulation Results", fontsize=16, fontweight="bold")

        x_min = self.x.min().item()
        x_max = self.x.max().item()

        # 1 — Deformed shape
        # Highlight elements
        for e in range(self.conn.size(0)):
            idx = self.conn[e, :] - 1
            x_elem = self.x[idx].view(-1)
            u_elem = self.u[idx].view(-1)
            x_def = x_elem + scalingfactor * u_elem
            # Generate a color based on the element index, equally spaced on the color wheel
            color = plt.cm.hsv(e / self.conn.size(0))
            axes[0].plot(x_def, torch.zeros_like(x_def), '-', lw=6, alpha=0.3, color=color)
        # Plot undeformed and deformed shapes
        axes[0].plot(self.x, torch.zeros_like(self.x), 'o-', label="Undeformed", color='black')
        axes[0].plot(self.x + scalingfactor * self.u, torch.zeros_like(self.x), 'mo:', label="Deformed (scaled)")
        axes[0].plot(self.x[self.bc.drlt_dofs - 1], -0.02 * torch.ones_like(self.bc.drlt_dofs), 'g^', markersize=15, label="Dirichlet BC")
        # Set plot limits and labels
        axes[0].set_title("Deformed Shape (scaled)")
        axes[0].set_xlabel("x [m]")
        axes[0].set_yticks([])
        axes[0].legend(loc="lower right")
        axes[0].grid(True)
        axes[0].set_xlim([x_min - 1, x_max * 1.1])
        axes[0].set_ylim([-0.02, 0.005])

        # 2 — Displacement
        axes[1].plot(self.x, self.u, 'x-', color='blue')
        axes[1].set_title("Displacement")
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("u(x) [m]")
        axes[1].grid(True)
        axes[1].set_xlim([x_min - 1, x_max * 1.1])
        # Highlight maximum displacement
        max_u = torch.max(self.u).item()
        max_u_idx = torch.argmax(self.u).item()
        x_max_u = self.x[max_u_idx].item()
        axes[1].annotate(
            f"Max: {max_u:.3e} m",
            xy=(x_max_u, max_u),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            color='blue',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='blue', facecolor='white', alpha=0.7)
        )

        # 3 — Stress distribution at Gauss points
        axes[2].plot(self.results["x_eps"], self.results["sigma"], 'mx-', label="σ(x)")
        axes[2].set_title("Stress Distribution (Gauss Points)")
        axes[2].set_xlabel("x [m]")
        axes[2].set_ylabel("σ(x) [Pa]")
        axes[2].grid(True)
        axes[2].legend(loc="best")
        axes[2].set_xlim([x_min - 1, x_max * 1.1])
        # Highlight maximum stress
        max_sigma = torch.max(self.results["sigma"]).item()
        max_sigma_idx = torch.argmax(self.results["sigma"]).item()
        x_max_sigma = self.results["x_eps"][max_sigma_idx].item()
        axes[2].annotate(
            f"Max: {max_sigma:.3e} Pa",
            xy=(x_max_sigma, max_sigma),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            color='magenta',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='magenta', facecolor='white', alpha=0.7)
        )

        # 4 — Nodal force vectors
        axes[3].scatter(self.x, self.frea, color='b', marker='s', label="$f_{rea}$")
        axes[3].scatter(self.x, self.fext, color='r', marker='o', label="$f_{ext}$")
        axes[3].scatter(self.x, self.fvol, color='c', marker='x', label="$f_{vol}$")
        # accumulated body force
        accumulated_fvol = torch.cumsum(self.fvol.flip(0), dim=0).flip(0)
        axes[3].scatter(self.x, accumulated_fvol, color='black', marker='^', label="Σ$f_{vol}$")
        # Highlight max external force
        max_fext = torch.max(self.fext).item()
        max_fext_idx = torch.argmax(self.fext).item()
        x_max = self.x[max_fext_idx].item()
        y_max = self.fext[max_fext_idx].item()
        axes[3].annotate(
            f"Max: {max_fext:.2f} N",
            xy=(x_max, y_max),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            color='r',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='red', facecolor='white', alpha=0.7)
        )
        # Highlight minimum reaction force
        min_frea = torch.min(self.frea).item()
        min_frea_idx = torch.argmin(self.frea).item()
        x_min = self.x[min_frea_idx].item()
        y_min = self.frea[min_frea_idx].item()
        axes[3].annotate(
            f"Min: {min_frea:.2f} N",
            xy=(x_min, y_min),
            xytext=(0, -12),
            textcoords="offset points",
            ha='center',
            va='top',
            fontsize=9,
            color='b',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='blue', facecolor='white', alpha=0.7)
        )
        # Highlight max accumulated body force
        max_accumulated_fvol = torch.max(accumulated_fvol).item()
        max_accumulated_fvol_idx = torch.argmax(accumulated_fvol).item()
        x_max_accum = self.x[max_accumulated_fvol_idx].item()
        y_max_accum = accumulated_fvol[max_accumulated_fvol_idx].item()
        axes[3].annotate(
            f"Max: {max_accumulated_fvol:.2f} N",
            xy=(x_max_accum, y_max_accum),
            xytext=(0, 12),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white', alpha=0.7)
        )
        # Set plot limits and labels
        axes[3].set_title("Nodal Forces")
        axes[3].set_xlabel("x [m]")
        axes[3].set_ylabel("Force [N]")
        axes[3].legend(loc="lower right")
        axes[3].grid(True)
        axes[3].set_xlim([x_min - 1, x_max * 1.1])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if filepath is not None:
            plt.savefig(filepath)
        plt.show()



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    nodes = torch.linspace(0, 70, 11).view(-1, 1)
    conn_list = torch.tensor([[1, 3, 2], [3, 5, 4], [5, 7, 6], [7, 9, 8], [9, 11, 10]])
    conn_list = conn_list[:, [0, 2, 1]]  # Reorder to [left, mid, right]

    E = 2.1e11 * torch.ones(conn_list.size(0), 1)
    # NOTE: Three cables with the same cross-sectional area
    area = 3 * 89.9e-6 * torch.ones(conn_list.size(0), 1)
    # NOTE: Density is dervied from weight per unit length
    RHO = 0.861 / 89.9e-6
    BODY_FORCE = RHO * 9.81
    #BODY_FORCE = 0
    mat = MaterialProperties(E=E, area=area, b=BODY_FORCE)

    f_sur = torch.zeros(11)
    f_sur[-1] = (300 + 630) * 9.81
    #f_sur[-1] = 0
    u_d = torch.tensor([0.])
    drlt_dofs = torch.tensor([1])
    boundary_conditions = BoundaryConditions(u_d=u_d, drlt_dofs=drlt_dofs, f_sur=f_sur)

    fem = Fem1D(nodes, conn_list, mat, boundary_conditions, nqp=2)
    fem.preprocess()
    fem.solve()
    fem.postprocess()
    fem.report()
    fem.plot()
