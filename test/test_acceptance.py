import unittest
import torch
from fem_1d.core import Fem1D, MaterialProperties, BoundaryConditions

torch.set_default_dtype(torch.float64)

class TestFem1DAcceptance(unittest.TestCase):
    def setUp(self):
        self.length = 70.0  # meters
        self.rope_mass = 0.861  # kg/m
        self.rope_num = 3
        self.n_nodes = 11
        self.x = torch.linspace(0, self.length, self.n_nodes).view(-1, 1)
        self.conn = torch.tensor([[1, 3, 2], [3, 5, 4], [5, 7, 6], [7, 9, 8], [9, 11, 10]])
        self.conn = self.conn[:, [0, 2, 1]]
        self.g = 9.81  # m/s^2, gravitational acceleration
        self.E_val = 2.1e11  # Pa, Young's modulus for steel
        self.A = 89.9e-6  # m^2, cross-sectional area of the rope
        self.A_total = self.A * self.rope_num
        self.E = self.E_val * torch.ones(self.conn.size(0), 1)
        self.area = self.A_total * torch.ones(self.conn.size(0), 1)
        self.nqp = 2
        self.drlt_dofs = torch.tensor([1])
        self.u_d = torch.tensor([0.])
        self.tol = 1e-5

    def compute_numerical_displacement(self, f_sur, b):
        bc = BoundaryConditions(u_d=self.u_d, drlt_dofs=self.drlt_dofs, f_sur=f_sur)
        mat = MaterialProperties(E=self.E, area=self.area, b=b)
        fem = Fem1D(x=self.x, conn=self.conn, material=mat, bc=bc, nqp=self.nqp)
        fem.preprocess()
        fem.solve()
        return fem.u[-1].item()  # u at bottom of rope

    def test_point_load_only(self):
        f_sur = torch.zeros(self.n_nodes)
        payload_mass = 375.0  # kg
        f_sur[-1] = payload_mass * self.g
        b = 0.0
        u_num = self.compute_numerical_displacement(f_sur, b)
        u_expected = f_sur[-1] * self.length / (self.E_val * self.A_total)
        self.assertAlmostEqual(u_num, u_expected, delta=self.tol,
            msg=f"Expected {u_expected:.6e}, got {u_num:.6e} for point load only")

    def test_body_force_only(self):
        f_sur = torch.zeros(self.n_nodes)
        RHO = self.rope_mass / self.A
        BODY_FORCE = RHO * self.g
        u_num = self.compute_numerical_displacement(f_sur, BODY_FORCE)
        u_expected = (self.rope_num * self.rope_mass * self.g * self.length**2) / (2 * self.E_val * self.A_total)
        self.assertAlmostEqual(u_num, u_expected, delta=self.tol,
            msg=f"Expected {u_expected:.6e}, got {u_num:.6e} for body force only")

    def test_combined_load(self):
        f_sur = torch.zeros(self.n_nodes)
        f_sur[-1] = (300 + 630) * self.g  # 300 kg payload + 630 kg rope mass
        RHO = self.rope_mass / self.A
        BODY_FORCE = RHO * self.g
        u_num = self.compute_numerical_displacement(f_sur, BODY_FORCE)
        u_point_load = f_sur[-1] * self.length / (self.E_val * self.A_total)
        u_body_force = (self.rope_num * self.rope_mass * self.g * self.length**2) / (2 * self.E_val * self.A_total)
        u_expected = u_point_load + u_body_force
        self.assertAlmostEqual(u_num, u_expected, delta=self.tol,
            msg=f"Expected {u_expected:.6e}, got {u_num:.6e} for combined load")

if __name__ == "__main__":
    unittest.main()
