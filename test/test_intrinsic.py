import unittest
import numpy as np
from typing import Callable
from swiftt.taylor import taylor_map, factory_taylor
from swiftt.math_algebra import tan, cos, sin, tanh, cosh, sinh, asin, asinh, acos, acosh, atan, atanh, exp, log,\
    erf, sqrt, cbrt, arctan, arctanh, arccos, arcsin, arcsinh, scalar_inversion


tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


def test_func_inverse(func: Callable, inv_func: Callable, const: float, order: int = 10):
    expansion = factory_taylor.zero_expansion(n_var=1, order=order)
    coeff = np.zeros(order + 1)
    coeff[1] = 1.
    expansion.coeff = coeff
    v1 = inv_func(func(const) + expansion)
    v2 = scalar_inversion(func, const + expansion)
    return np.allclose(v1.coeff, v2.coeff)


def test_inverse_pow(power: float):
    return test_func_inverse(lambda x: x**power, lambda y: y**(1. / power), 2.)


def test_map_intrinsic(func) -> bool:
    expansion1 = null_expansion_2var_order2.copy()
    coeff1 = [0.5, -1., 4., 2., 5., 1.]
    expansion1.coeff = coeff1

    expansion2 = null_expansion_2var_order2.copy()
    coeff2 = [0.1, 6., 3., -2., 1., -1.]
    expansion2.coeff = coeff2

    map1 = taylor_map.RealTaylorMap([expansion1, expansion2])
    on_map = func(map1)
    for el1, el2 in zip(map1, on_map):
        if el2 != func(el1):
            return False
    return True


class TestIntrinsic(unittest.TestCase):

    def test_cos0(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0., 1., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        cosinus = cos(expansion1)
        coeff = cosinus.coeff

        if not np.array_equal(coeff, [1., 0., -0.5, 0., 1. / 24., 0.]):
            self.fail()

    def test_cosh0(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0., 1., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        cosinush = cosh(expansion1)
        coeff = cosinush.coeff

        if not np.array_equal(coeff, [1., 0., 0.5, 0., 1. / 24., 0.]):
            self.fail()

    def test_sin0(self):
        expansion1 = factory_taylor.zero_expansion(1, 6)
        coeff1 = [0., 1., 0., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        sinus = sin(expansion1)
        coeff = sinus.coeff

        if not np.array_equal(coeff, [0., 1., 0., -1. / 6., 0., 1. / 120., 0.]):
            self.fail()

    def test_sinh0(self):
        expansion1 = factory_taylor.zero_expansion(1, 6)
        coeff1 = [0., 1., 0., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        sinush = sinh(expansion1)
        coeff = sinush.coeff

        if not np.array_equal(coeff, [0., 1., 0., 1. / 6., 0., 1. / 120., 0.]):
            self.fail()

    def test_cos_sin(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = cos(expansion1)**2 + sin(expansion1)**2
        coeff2 = expansion2.coeff

        if np.fabs(1. - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff2[1:], np.zeros(len(coeff2) - 1)):
            self.fail()

    def test_asin0(self):
        expansion1 = factory_taylor.zero_expansion(1, 6)
        coeff1 = [0., 1., 0., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        arcsin = asin(expansion1)
        coeff = arcsin.coeff

        if not np.array_equal(coeff, [0., 1., 0., 1. / 6., 0., 3. / 40., 0.]):
            self.fail()

    def test_atan0(self):
        expansion1 = factory_taylor.zero_expansion(1, 6)
        coeff1 = [0., 1., 0., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        arctan = atan(expansion1)
        coeff = arctan.coeff

        if not np.array_equal(coeff, [0., 1., 0., -1. / 3., 0., 0.2, 0.]):
            self.fail()

    def test_sin_asin(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [0.5, 3., 4., -5., -1., -3.]
        expansion.coeff = coeff

        expansion1 = asin(sin(expansion))
        coeff1 = expansion1.coeff

        expansion2 = sin(asin(expansion))
        coeff2 = expansion2.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff1[1:], coeff2[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_cos_acos(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [0.5, 3., 4., -5., -1., -3.]
        expansion.coeff = coeff

        expansion1 = acos(cos(expansion))
        coeff1 = expansion1.coeff

        expansion2 = cos(acos(expansion))
        coeff2 = expansion2.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff1[1:], coeff2[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_sinh_asinh(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [2., 3., 4., -5., -1., -3.]
        expansion.coeff = coeff

        expansion1 = asinh(sinh(expansion))
        coeff1 = expansion1.coeff

        expansion2 = sinh(asinh(expansion))
        coeff2 = expansion2.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff1[1:], coeff2[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_cosh_acosh(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [2., 3., 4., -5., -1., -3.]
        expansion.coeff = coeff

        expansion1 = acosh(cosh(expansion))
        coeff1 = expansion1.coeff

        expansion2 = cosh(acosh(expansion))
        coeff2 = expansion2.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff1[1:], coeff2[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_tan_atan(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [0.5, 3., 4., -5., -1., -3.]
        expansion.coeff = coeff

        expansion1 = atan(tan(expansion))
        coeff1 = expansion1.coeff

        expansion2 = tan(atan(expansion))
        coeff2 = expansion2.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff1[1:], coeff2[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_cosh_sinh(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [0.5, 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = cosh(expansion1)**2 - sinh(expansion1)**2
        coeff2 = expansion2.coeff

        if np.fabs(1. - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(np.zeros(len(coeff2) - 1), coeff2[1:]):
            self.fail()

    def test_cosh(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [3., 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        cosinush = cosh(expansion)
        coeff1 = cosinush.coeff
        cosinush2 = (exp(expansion) + exp(-expansion)) / 2.
        coeff2 = cosinush2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_sinh(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [3., 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        sinush = sinh(expansion)
        coeff1 = sinush.coeff
        sinush2 = (exp(expansion) - exp(-expansion)) / 2.
        coeff2 = sinush2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_tanh_atanh(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [0.5, 3., 4., -5., -1., -3.]
        expansion.coeff = coeff

        expansion1 = atanh(tanh(expansion))
        coeff1 = expansion1.coeff

        expansion2 = tanh(atanh(expansion))
        coeff2 = expansion2.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff1[1:], coeff2[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_exp0(self):
        expansion1 = factory_taylor.zero_expansion(1, 10)
        coeff1 = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        expon = exp(expansion1)
        coeff = expon.coeff

        if not np.allclose(coeff, np.append([1.], 1. / np.cumprod(np.arange(1., len(coeff))))):
            self.fail()

    def test_exp_div(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [3., 1.5, -4., 2., -1., 7.]
        expansion1.coeff = coeff1

        expon_inv = exp(-expansion1)
        one_over_expon = 1. / exp(expansion1)

        coeff = expon_inv.coeff
        coeff1 = one_over_expon.coeff

        if not np.allclose(coeff, coeff1):
            self.fail()

    def test_tan(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [3., 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        tan1 = tan(expansion)
        tan2 = sin(expansion) / cos(expansion)
        coeff1 = tan1.coeff
        coeff2 = tan2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_tanh(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [3., 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        tan1 = tanh(expansion)
        tan2 = sinh(expansion) / cosh(expansion)
        coeff1 = tan1.coeff
        coeff2 = tan2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_sqrt0(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [1., 1., 0., 0.]
        expansion1.coeff = coeff1

        sqroot = sqrt(expansion1)
        coeff = sqroot.coeff

        if not np.array_equal(coeff, [1., 0.5, -0.125, 0.0625]):
            self.fail()

    def test_sqrt_sq(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [0.5, 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        expansion1 = sqrt(expansion**2)
        coeff1 = expansion1.coeff

        expansion2 = sqrt(expansion)**2
        coeff2 = expansion2.coeff

        expansion = abs(expansion)
        coeff = expansion.coeff

        if np.fabs(coeff1[0] - coeff2[0]) > tol_coeff:
            self.fail()
        if not np.allclose(coeff[1:], coeff1[1:]):
            self.fail()
        if not np.allclose(coeff[1:], coeff2[1:]):
            self.fail()

    def test_sqrt_exp_log_float(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [0.5, 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        expansion1 = sqrt(expansion)
        coeff1 = expansion1.coeff

        expansion2 = exp(0.5 * log(expansion))
        coeff2 = expansion2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_sqrt_exp_log_complex(self):
        expansion = factory_taylor.zero_expansion(1, 5, dtype=np.complex128)
        coeff = [complex(0.5, 1.), complex(1.5, -1.), complex(-4., 0.), complex(0., 2.), complex(-1., 3.), complex(7., 0.)]
        expansion.coeff = coeff

        expansion1 = sqrt(expansion)
        coeff1 = expansion1.coeff

        expansion2 = exp(0.5 * log(expansion))
        coeff2 = expansion2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_sqrt_pow(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [0.5, 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        expansion1 = sqrt(expansion)
        coeff1 = expansion1.coeff

        expansion2 = expansion**0.5
        coeff2 = expansion2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_sqrt_iter(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [0.5, 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        expansion1 = sqrt(expansion)
        coeff1 = expansion1.coeff

        sqrt_const = np.sqrt(expansion.const)
        factor = 0.5 / sqrt_const
        nilpo = expansion.get_nilpo_part()
        expansion2 = nilpo * factor
        if expansion2.total_depth != 1:
            self.fail()
        for i in range(2, expansion.order + 1):
            save = expansion2.copy()
            expansion2 = (nilpo - expansion2**2) * factor
            if (expansion2 - save).total_depth != i:
                self.fail()
        expansion2 += sqrt_const
        coeff2 = expansion2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_cbrt_pow(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [0.5, 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        expansion1 = cbrt(expansion)
        coeff1 = expansion1.coeff

        expansion2 = expansion**(1. / 3.)
        coeff2 = expansion2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_log(self):
        expansion1 = factory_taylor.zero_expansion(1, 10)
        coeff1 = [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        expansion1.coeff = coeff1

        logar = log(expansion1)
        coeff = logar.coeff

        if coeff[0] != 0.:
            self.fail()
        if not np.array_equal(coeff[1:], [(-1.)**(i - 1) / float(i) for i in range(1, len(coeff))]):
            self.fail()

    def test_log_exp(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [3., 1.5, -4., 2, -1., 7.]
        expansion1.coeff = coeff1

        expansion2 = log(exp(expansion1))
        coeff = expansion2.coeff

        if not np.allclose(coeff1, coeff):
            self.fail()

    def test_asinh(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0.5, 1.5, -4., 2., -1., 7.]
        expansion1.coeff = coeff1

        asinh1 = asinh(expansion1)
        asinh2 = log(expansion1 + sqrt(expansion1**2 + 1.))

        coeff = asinh1.coeff
        coeff1 = asinh2.coeff

        if not np.allclose(coeff1, coeff):
            self.fail()

    def test_acosh(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [3., 1.5, -4., 2., -1., 7.]
        expansion1.coeff = coeff1

        acosh1 = acosh(expansion1)
        acosh2 = log(expansion1 + sqrt(expansion1**2 - 1.))

        coeff = acosh1.coeff
        coeff1 = acosh2.coeff

        if not np.allclose(coeff1, coeff):
            self.fail()

    def test_atanh(self):
        expansion = factory_taylor.zero_expansion(1, 5)
        coeff = [0.5, 1.5, -4., 2., -1., 7.]
        expansion.coeff = coeff

        atanh1 = atanh(expansion)
        atanh2 = 0.5 * log((expansion + 1.) / (1. - expansion))

        coeff1 = atanh1.coeff
        coeff2 = atanh2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_erf0(self):
        expansion = factory_taylor.zero_expansion(1, 9)
        coeff = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
        expansion.coeff = coeff

        error_fun = erf(expansion)
        expansion1 = error_fun.deriv_once_wrt_var(0)
        expansion2 = 2. * exp(-expansion**2) / np.sqrt(np.pi)

        coeff1 = expansion1.coeff
        coeff2 = expansion2.coeff

        if not np.allclose(coeff1, coeff2):
            self.fail()

    def test_inverse_exp(self):
        if not test_func_inverse(exp, log, 2.):
            return self.fail()

    def test_inverse_log(self):
        if not test_func_inverse(log, exp, 3.):
            return self.fail()

    def test_inverse_cos(self):
        if not test_func_inverse(cos, acos, 1.):
            return self.fail()

    def test_inverse_acos(self):
        if not test_func_inverse(acos, cos, -0.5):
            return self.fail()

    def test_inverse_sin(self):
        if not test_func_inverse(sin, asin, 1.):
            return self.fail()

    def test_inverse_asin(self):
        if not test_func_inverse(asin, sin, -0.5):
            return self.fail()

    def test_inverse_tan(self):
        if not test_func_inverse(tan, atan, 1.):
            return self.fail()

    def test_inverse_atan(self):
        if not test_func_inverse(atan, tan, -0.5):
            return self.fail()

    def test_inverse_cosh(self):
        if not test_func_inverse(cosh, acosh, 1.):
            return self.fail()

    def test_inverse_acosh(self):
        if not test_func_inverse(acosh, cosh, 2.):
            return self.fail()

    def test_inverse_sinh(self):
        if not test_func_inverse(sinh, asinh, 1.):
            return self.fail()

    def test_inverse_asinh(self):
        if not test_func_inverse(asinh, sinh, -0.5):
            return self.fail()

    def test_inverse_tanh(self):
        if not test_func_inverse(tanh, atanh, 1.):
            return self.fail()

    def test_inverse_atanh(self):
        if not test_func_inverse(atanh, tanh, -0.5):
            return self.fail()

    def test_inverse_pow2(self):
        if not test_inverse_pow(2.):
            return self.fail()

    def test_inverse_pow3(self):
        if not test_inverse_pow(3.):
            return self.fail()

    def test_modulo(self):
        expansion1 = null_expansion_3var_order2.copy()
        coeff1 = [8., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        arg_mod = 3.
        if expansion1 % arg_mod != expansion1 - 6.:
            self.fail()

    def test_map_sqrt(self):
        if not test_map_intrinsic(sqrt):
            self.fail()

    def test_map_exp(self):
        if not test_map_intrinsic(exp):
            self.fail()

    def test_map_log(self):
        if not test_map_intrinsic(log):
            self.fail()

    def test_map_cos(self):
        if not test_map_intrinsic(cos):
            self.fail()

    def test_map_sin(self):
        if not test_map_intrinsic(sin):
            self.fail()

    def test_map_cosh(self):
        if not test_map_intrinsic(cosh):
            self.fail()

    def test_map_sinh(self):
        if not test_map_intrinsic(sinh):
            self.fail()

    def test_map_tan(self):
        if not test_map_intrinsic(tan):
            self.fail()

    def test_map_tanh(self):
        if not test_map_intrinsic(tanh):
            self.fail()

    def test_map_arctan(self):
        if not test_map_intrinsic(arctan):
            self.fail()

    def test_map_arctanh(self):
        if not test_map_intrinsic(arctanh):
            self.fail()

    def test_map_arcsin(self):
        if not test_map_intrinsic(arcsin):
            self.fail()

    def test_map_arccos(self):
        if not test_map_intrinsic(arccos):
            self.fail()

    def test_map_arcsinh(self):
        if not test_map_intrinsic(arcsinh):
            self.fail()


if __name__ == '__main__':
    unittest.main()
