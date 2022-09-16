import unittest
import numpy as np
from swiftt.taylor import factory_taylor
from swiftt.taylor.real_multivar_taylor import RealMultivarTaylor
from swiftt.math_algebra import cos, sin, exp, sqrt


tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


def intermediate(order: int) -> RealMultivarTaylor:
    x, y, z = factory_taylor.create_unknown_map(order=order, consts=[1., 2., -1.], var_names=["x", "y", "z"])
    g = exp((sin(x) * cos(y) + 1.) / sqrt(1. + x**2 + y**2 + z**2))
    g = g.deriv_once_wrt_var(1).integ_once_wrt_var(0).integ_once_wrt_var(2)
    return g


class TestAgainst(unittest.TestCase):

    def test_non_regression(self):
        g = intermediate(order=4)
        regre_coeff = [0.,     0.,     0.,     0.,     0.,     0.,
                       -0.45942301,  0.,     0.,     0.,     0.,     0.,
                       -0.02996078,  0.,     0.57760928, -0.05369175,  0.,     0.,
                       0.,     0.,     0.,     0.,     0.06033602,  0.,
                       0.04807805,  0.01270756,  0.,    -0.19353301,  0.11813837,  0.0072345,
                       0.,     0.,     0.,     0.,     0.]
        if not np.allclose(regre_coeff, g.coeff):
            self.fail()

    def test_pyaudi(self):
        try:
            order = 4
            g1 = intermediate(order)
            from pyaudi.core import gdual_double
            x = gdual_double(1., "x", order)
            y = gdual_double(2., "y", order)
            z = gdual_double(-1., "z", order)
            g2 = ((x.sin() * y.cos() + 1.) / ((x**2 + y**2 + z**2 + 1.).sqrt())).exp()
            g2 = g2.partial("y").integrate("x").integrate("z")
            for exponent in g1.get_mapping_monom().keys():
                dict_deriv = {"dx": exponent[0], "dy": exponent[1], "dz": exponent[2]}
                self.assertAlmostEqual(g1.get_partial_deriv(exponent), g2.get_derivative(dict_deriv), delta=1.e-15)
        except ImportError:
            pass

    def test_sympy(self):
        try:
            from sympy import poly
            str1 = "x**4 - 2 * x**3 + x**2 - x + 2 + x * y + 4 * y**2 + x * y * z +" \
                   " z**3 + x * z**3 - 3 * x**2 * y**2 - 5 * y**4"
            str2 = "3 * x**4 - x**2 - z * x + 1 + x**2 * y + 4 * y**4 + x * y **2 * z +" \
                   " y * z**3 - x * z**3 - 4 * x**2 * y * z + 3 * y**3"
            poly1, poly2 = poly(str1), poly(str2)
            poly_prod = poly1 * poly2
            expans1 = factory_taylor.from_string(str1, order=8)
            expans2 = factory_taylor.from_string(str2, order=8)
            expans_prod = expans1 * expans2
            coeff = expans_prod.coeff
            mapping = expans1.get_mapping_monom()
            for el, monom in zip(poly_prod.coeffs(), poly_prod.monoms()):
                if coeff[mapping[monom]] != el:
                    self.fail()
        except ImportError:
            pass


if __name__ == '__main__':
    unittest.main()
