import unittest
import torch
import math
from fem_1d.core import shape1d, gauss1d, jacobian1d


class TestShape1D(unittest.TestCase):
    def setUp(self):
        # Common test data for nen = 3
        self.xi_test = 0.3
        self.nen = 3
        self.n_expected = torch.tensor([[-0.105], [0.765], [0.34]])
        self.gamma_expected = torch.tensor([[-0.2], [-1.3], [1.5]])

    def test_sum_of_shape_functions_is_one(self):
        N, _ = shape1d(self.xi_test, self.nen)
        self.assertTrue(torch.allclose(torch.sum(N), torch.tensor(1.0), atol=1e-12),
                        msg=f"Sum of shape functions is {torch.sum(N)} instead of 1.")

    def test_sum_of_shape_function_derivatives_is_zero(self):
        _, gamma = shape1d(self.xi_test, self.nen)
        self.assertTrue(torch.allclose(torch.sum(gamma), torch.tensor(0.0), atol=1e-12),
                        msg=f"Sum of derivatives is {torch.sum(gamma)} instead of 0.")

    def test_shape_function_value_N0(self):
        N, _ = shape1d(self.xi_test, self.nen)
        self.assertTrue(torch.allclose(N[0], self.n_expected[0], atol=1e-3),
                        msg=f"N_0 is {N[0]} but expected ≈ {self.n_expected[0]}")

    def test_shape_function_derivative_gamma0(self):
        _, gamma = shape1d(self.xi_test, self.nen)
        self.assertTrue(torch.allclose(gamma[0], self.gamma_expected[0], atol=1e-3),
                        msg=f"gamma_0 is {gamma[0]} but expected ≈ {self.gamma_expected[0]}")


class TestGauss1D(unittest.TestCase):
    def test_nqp_1(self):
        xi, w8 = gauss1d(1)
        self.assertTrue(torch.allclose(xi, torch.tensor([0.0])), "xi for nqp=1 is incorrect.")
        self.assertTrue(torch.allclose(w8, torch.tensor([2.0])), "weights for nqp=1 are incorrect.")
        self.assertAlmostEqual(torch.sum(w8).item(), 2.0, places=12, msg="sum of weights should be 2 for nqp=1")

    def test_nqp_2(self):
        xi_expected = torch.tensor([-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)])
        w8_expected = torch.tensor([1.0, 1.0])
        xi, w8 = gauss1d(2)
        self.assertTrue(torch.allclose(xi, xi_expected, atol=1e-12), "xi for nqp=2 is incorrect.")
        self.assertTrue(torch.allclose(w8, w8_expected, atol=1e-12), "weights for nqp=2 are incorrect.")
        self.assertAlmostEqual(torch.sum(w8).item(), 2.0, places=12, msg="sum of weights should be 2 for nqp=2")

    def test_nqp_3(self):
        xi_expected = torch.tensor([
            -math.sqrt(3 / 5),
            0.0,
            math.sqrt(3 / 5)
        ])
        w8_expected = torch.tensor([
            5.0 / 9,
            8.0 / 9,
            5.0 / 9
        ])
        xi, w8 = gauss1d(3)
        self.assertTrue(torch.allclose(xi, xi_expected, atol=1e-12), "xi for nqp=3 is incorrect.")
        self.assertTrue(torch.allclose(w8, w8_expected, atol=1e-12), "weights for nqp=3 are incorrect.")
        self.assertAlmostEqual(torch.sum(w8).item(), 2.0, places=12, msg="sum of weights should be 2 for nqp=3")

    def test_invalid_nqp_raises(self):
        with self.assertRaises(ValueError):
            gauss1d(4)
           

class TestJacobian1D(unittest.TestCase):

    def test_linear_element(self):
        # Nodes at x = 2 and x = 4 → element length = 2
        xe = torch.tensor([2.0, 4.0])  # shape (2,)
        gamma = torch.tensor([-0.5, 0.5])  # derivatives for linear shape functions at any xi
        detJq, invJq = jacobian1d(xe, gamma, nen=2)
        self.assertAlmostEqual(detJq, 1.0, places=12, msg="detJq should be 1.0 for uniform linear element")
        self.assertAlmostEqual(invJq, 1.0, places=12, msg="invJq should be 1.0 for uniform linear element")

    def test_quadratic_element(self):
        # Nodes at x = 1, 2, 3 (symmetric spacing)
        xe = torch.tensor([1.0, 2.0, 3.0])
        # gamma at xi = 0 for 3-node element
        gamma = torch.tensor([-0.5, 0.0, 0.5])
        detJq, invJq = jacobian1d(xe, gamma, nen=3)
        self.assertAlmostEqual(detJq, 1.0, places=12, msg="detJq should be 1.0 for symmetric quadratic element")
        self.assertAlmostEqual(invJq, 1.0, places=12, msg="invJq should be 1.0 for symmetric quadratic element")

    def test_negative_determinant_raises(self):
        xe = torch.tensor([4.0, 2.0])  # flipped nodes
        gamma = torch.tensor([-0.5, 0.5])
        with self.assertRaises(ValueError):
            jacobian1d(xe, gamma, nen=2)

    def test_invalid_nen_raises(self):
        xe = torch.tensor([0.0, 1.0, 2.0, 3.0])
        gamma = torch.tensor([0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            jacobian1d(xe, gamma, nen=4)

if __name__ == "__main__":
    unittest.main()
