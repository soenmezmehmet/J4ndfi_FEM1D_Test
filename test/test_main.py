import unittest
import torch
import math
from fem_1d.core import shape1d, gauss1d


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

if __name__ == "__main__":
    unittest.main()
