import numpy as np

class Fem1D:
    def __init__(self, x, conn, material, bc, nqp=2, num_threads=1):
        self.x = np.array(x).reshape(-1, 1)       # ensure x is a column vector (nnp x 1)
        self.conn = np.array(conn)                # ensure conn is a 2D numpy array
        self.mat = material
        self.bc = bc
        self.nqp = nqp
        self.num_threads = num_threads

        self.shape = {
            "nnp": self.x.shape[0],               # number of nodal points
            "ndf": 1,                              # degrees of freedom per node
            "nel": self.conn.shape[0],             # number of elements
            "nen": self.conn.shape[1],             # number of nodes per element (2 or 3)
        }

        # Initialize system arrays
        self.K = np.zeros((self.shape["nnp"], self.shape["nnp"]))  # global stiffness matrix
        self.u = np.zeros((self.shape["nnp"], 1))                  # displacement vector
        self.fext = np.zeros((self.shape["nnp"], 1))               # total external force vector

    def preprocess(self):
        # Example preprocessing: compute element lengths
        self.lengths = np.zeros(self.shape["nel"])
        for e, conn_e in enumerate(self.conn):
            x_e = self.x[conn_e].flatten()
            self.lengths[e] = np.abs(x_e[-1] - x_e[0])

    def assemble(self):
        # Assemble global stiffness matrix (placeholder linear element implementation)
        E = self.mat["E"]
        A = self.mat["A"]

        for e, conn_e in enumerate(self.conn):
            i, j = conn_e
            le = self.lengths[e]
            ke = (E * A / le) * np.array([[1, -1], [-1, 1]])

            self.K[i, i] += ke[0, 0]
            self.K[i, j] += ke[0, 1]
            self.K[j, i] += ke[1, 0]
            self.K[j, j] += ke[1, 1]

    def apply_boundary_conditions(self):
        for node, value in self.bc.get("dirichlet", []):
            self.K[node, :] = 0
            self.K[node, node] = 1
            self.fext[node] = value
            self.u[node] = value

        for node, value in self.bc.get("neumann", []):
            self.fext[node] += value

    def solve(self):
        self.u = np.linalg.solve(self.K, self.fext)

    def run(self):
        self.preprocess()
        self.assemble()
        self.apply_boundary_conditions()
        self.solve()
        return self.u



import numpy as np

# Node coordinates (column vector)
nodes = np.linspace(0, 70, 11).reshape(-1, 1)

# Connectivity (5 elements, each with 3 nodes) — converted to 0-based indexing
conn_list = np.array([[0, 2, 1],
                      [2, 4, 3],
                      [4, 6, 5],
                      [6, 8, 7],
                      [8, 10, 9]])

# Material properties
E = 2.1e11  # Young's modulus (Pa)
area = 3 * 89.9e-6  # Total cross-sectional area (m^2)
rho = 0.861 / 89.9e-6  # Density derived from weight per unit length
body_force = rho * 9.81  # Body force (N/m)

material = {
    "E": E,
    "A": area,
    "b": body_force
}

# External (Neumann) force vector
f_sur = np.zeros((11, 1))
f_sur[-1] = (300 + 630) * 9.81  # Load at last node

# Boundary conditions
dirichlet = [(0, 0.0)]  # u(0) = 0
neumann = [(i, f_sur[i, 0]) for i in range(len(f_sur)) if f_sur[i, 0] != 0]

boundary_conditions = {
    "dirichlet": dirichlet,
    "neumann": neumann
}

# Run the Fem1D solver
fem = Fem1D(nodes, conn_list, material, boundary_conditions, nqp=2)
u = fem.run()

# Print displacements
print("Displacements (u):")
print(u)
