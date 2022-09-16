import unittest
import numpy as np
from swiftt.taylor import taylor_map, factory_taylor

tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


class TestDifferentiation(unittest.TestCase):

    def test_deriv_wrt_map(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [0., 1., 4., -5., 3.5, 1., -2., 3., 0., 7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order3.copy()
        coeff2 = [0., -3., 0., 6., -1.5, 2., 3., -4., 8., 1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        if map1.deriv_wrt_var(1, 2) != taylor_map.RealTaylorMap([expansion1.deriv_wrt_var(1, 2),
                                                                 expansion2.deriv_wrt_var(1, 2)]):
            self.fail()

    def test_integ_wrt_map(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [0., 1., 4., -5., 3.5, 1., -2., 3., 0., 7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order3.copy()
        coeff2 = [0., -3., 0., 6., -1.5, 2., 3., -4., 8., 1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        if map1.integ_wrt_var(1, 2) != taylor_map.RealTaylorMap([expansion1.integ_wrt_var(1, 2),
                                                                 expansion2.integ_wrt_var(1, 2)]):
            self.fail()

    def test_integ_deriv_integ_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0., 1.5, -4., 2., -1., 3.]
        expansion1.coeff = coeff1

        deriv = expansion1.deriv_once_wrt_var()
        same = deriv.integ_once_wrt_var()

        if same != expansion1:
            self.fail()

    def test_rigorous_integ_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0., 1.5, -4., 2., -1., 3.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.rigorous_integ_once_wrt_var()

        if expansion2 != expansion1.prolong_one_order().integ_once_wrt_var():
            self.fail()

        if expansion2.truncated(expansion1.order) != expansion1.integ_once_wrt_var():
            self.fail()

    def test_rigorous_integ_univariate_map(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0., 1.5, -4., 2., -1., 3.]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(1, 5)
        coeff2 = [3., 1.5, -4., 2, -1., 6.]
        expansion2.coeff = coeff2

        map_expans = taylor_map.RealTaylorMap([expansion1, expansion2])
        map_integ = taylor_map.RealTaylorMap([expansion1.rigorous_integ_once_wrt_var(),
                                              expansion2.rigorous_integ_once_wrt_var()])

        if map_expans.rigorous_integ_once_wrt_var(0) != map_integ:
            self.fail()

    def test_rigorous_integ_bivariate(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.rigorous_integ_once_wrt_var(1)

        if expansion2 != expansion1.prolong_one_order().integ_once_wrt_var(1):
            self.fail()

        if expansion2.truncated(expansion1.order) != expansion1.integ_once_wrt_var(1):
            self.fail()

    def test_rigorous_integ_bivariate_map(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order3.copy()
        coeff2 = [0., -1., 2., -2., 4., 5., 2., -1., -7., 8.]
        expansion2.coeff = coeff2

        map_expans = taylor_map.RealTaylorMap([expansion1, expansion2])
        map_integ = taylor_map.RealTaylorMap([expansion1.rigorous_integ_once_wrt_var(0),
                                              expansion2.rigorous_integ_once_wrt_var(0)])

        if map_expans.rigorous_integ_once_wrt_var(0) != map_integ:
            self.fail()

    def test_deriv_bivariate(self):
        expansion = null_expansion_2var_order3.copy()
        coeff = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion.coeff = coeff

        deriv1 = expansion.deriv_once_wrt_var(0)
        coeff1 = deriv1.coeff
        deriv2 = expansion.deriv_once_wrt_var(1)
        coeff2 = deriv2.coeff

        if not np.array_equal(coeff1[:6], [3., -10., -1., -6., 2., 6.]):
            self.fail()
        if not np.array_equal(coeff2[:6], [4., -1., -6., 1., 12., -21.]):
            self.fail()
        if not np.array_equal(coeff1[6:], np.zeros(deriv1.dim_alg - 6)):
            self.fail()
        if not np.array_equal(coeff2[6:], np.zeros(deriv1.dim_alg - 6)):
            self.fail()

    def test_deriv_trivariate(self):
        expansion = null_expansion_3var_order2.copy()
        coeff = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion.coeff = coeff

        deriv1 = expansion.deriv_once_wrt_var(0)
        coeff1 = deriv1.coeff

        if not np.array_equal(coeff1[:4], [3., -2., -3., -2.]):
            self.fail()
        if not np.array_equal(coeff1[4:], np.zeros(deriv1.dim_alg - 4)):
            self.fail()

    def test_integ_bivariate(self):
        expansion = null_expansion_2var_order3.copy()
        coeff = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion.coeff = coeff

        integ1 = expansion.integ_once_wrt_var(0)
        coeff1 = integ1.coeff
        integ2 = expansion.integ_once_wrt_var(1)
        coeff2 = integ2.coeff

        if not np.array_equal(coeff1[:10], [0., 2., 0., 1.5, 4., 0., -5./3., -0.5, -3., 0.]):
            self.fail()
        if not np.array_equal(coeff2[:10], [0., 0., 2., 0., 3., 2., 0., -5., -0.5, -1.]):
            self.fail()

    def test_integ_trivariate(self):
        expansion1 = null_expansion_3var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_3var_order2.copy()
        coeff2 = [1., -4., 0., 5., 6., 8., -3., 2., 9., -1.]
        expansion2.coeff = coeff2

        integ1 = expansion1.integ_once_wrt_var(0)
        integ2 = expansion2.integ_once_wrt_var(0)
        integ_map = taylor_map.RealTaylorMap([expansion1, expansion2]).integ_once_wrt_var(0)

        if integ1 != integ_map[0] or integ2 != integ_map[1]:
            self.fail()

    def test_schwartz_bivariate(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        deriv12 = expansion1.deriv_once_wrt_var(0).deriv_once_wrt_var(1)

        deriv21 = expansion1.deriv_once_wrt_var(1).deriv_once_wrt_var(0)

        if deriv12 != deriv21:
            self.fail()


if __name__ == '__main__':
    unittest.main()
