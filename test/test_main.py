import unittest
import torch
from fem_1d.core import shape1d


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


if __name__ == "__main__":
    unittest.main()
