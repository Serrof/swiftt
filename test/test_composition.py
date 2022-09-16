import unittest
import numpy as np
from swiftt.taylor import taylor_map, factory_taylor
from swiftt.math_algebra import cos, sin, atan2, sqrt

tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


class TestComposition(unittest.TestCase):

    def test_comp_expansion_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff2 = [0., 7., -6., 0.]
        expansion2.coeff = coeff2

        comp = expansion1(expansion2)
        coeff = comp.coeff

        if coeff[0] != 2. or coeff[1] != -21. or coeff[2] != 263. or coeff[3] != 952.:
            self.fail()

    def test_inverted_map_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [0., 2., 1., 7., -4., 5.]
        expansion1.coeff = coeff1

        inverted = taylor_map.RealTaylorMap([expansion1]).compo_inverse()[0]
        composed = inverted(expansion1)
        coeff = composed.coeff

        if coeff[0] != 0. or coeff[1] != 1.:
            self.fail()
        if not np.array_equal(coeff[2:], np.zeros(len(coeff) - 2)):
            self.fail()

    def test_inverted_polar_coord(self):
        x1 = null_expansion_2var_order3.copy()
        coeff1 = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
        x1.coeff = coeff1

        x2 = null_expansion_2var_order3.copy()
        coeff2 = [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
        x2.coeff = coeff2

        a = 0.5
        b = -1.
        expansion1 = x1 + a
        expansion2 = x2 + b
        radius = sqrt(expansion1**2 + expansion2**2)
        theta = atan2(expansion2, expansion1)

        polar_coord = taylor_map.RealTaylorMap([radius, theta])
        inverted = polar_coord.get_nilpo_part().compo_inverse()
        inverted[0] += a
        inverted[1] += b

        expansion1 = x1 + sqrt(a**2 + b**2)
        expansion2 = x2 + atan2(b, a)
        analytical = taylor_map.RealTaylorMap([expansion1 * cos(expansion2), expansion1 * sin(expansion2)])

        for i in range(0, 2):
            if not np.allclose(inverted[i].coeff, analytical[i].coeff):
                self.fail()

    def test_inverted_map_bivariate(self):
        x1 = null_expansion_2var_order3.copy()
        coeff1 = [0., 1., 4., -5., 3.5, 1., -2., 3., 0., 7.]
        x1.coeff = coeff1

        x2 = null_expansion_2var_order3.copy()
        coeff2 = [0., -3., 0., 6., -1.5, 2., 3., -4., 8., 1.]
        x2.coeff = coeff2

        map_origin = taylor_map.RealTaylorMap([x1, x2])
        inverted = map_origin.compo_inverse()
        composed = map_origin.compose(inverted)

        for i in range(0, 2):
            coeff = composed[i].coeff
            if coeff[i + 1] != 1.:
                self.fail()
            for j in range(3, len(coeff)):
                if np.fabs(coeff[j]) > tol_coeff:
                    self.fail()

    def test_truncated_inverse_univariate(self):
        old_order = 5
        expansion = factory_taylor.zero_expansion(1, old_order)
        coeff = np.random.random(old_order + 1)
        coeff[0] = 0.
        expansion.coeff = coeff

        new_order = 4
        trunc = expansion.compo_inverse().truncated(new_order)
        same = expansion.truncated(new_order).compo_inverse()

        coeff1 = trunc.coeff
        coeff2 = same.coeff
        if not np.array_equal(coeff1[:new_order + 1], coeff2[:new_order + 1]):
            self.fail()

    def test_comp_map_1(self):
        expansion1 = factory_taylor.zero_expansion(1, 2)
        coeff1 = [3., -1., 2.]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(1, 2)
        coeff2 = [-2., 5., 1.]
        expansion2.coeff = coeff2

        lhs = taylor_map.RealTaylorMap([expansion1, expansion2])

        expansion = null_expansion_2var_order2.copy()
        coeff_rhs = [0., 3., 4., -5., -1., -3.]
        expansion.coeff = coeff_rhs
        rhs = taylor_map.RealTaylorMap([expansion])

        composed = lhs(rhs)
        coeff = composed[0].coeff

        if not np.array_equal(coeff, [3., -3., -4., 23., 49., 35.]):
            self.fail()

    def test_comp_map_2(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [3., -1., 0., 2., 0., 1.]
        expansion.coeff = coeff

        lhs = taylor_map.RealTaylorMap([expansion])

        expansion1 = factory_taylor.zero_expansion(1, 2)
        coeff1 = [0., 3., 4.]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(1, 2)
        coeff2 = [0., -1., 2.]
        expansion2.coeff = coeff2

        rhs = taylor_map.RealTaylorMap([expansion1, expansion2])

        composed = lhs(rhs)
        coeff = composed[0].coeff

        if not np.array_equal(coeff, [3., -3., 15.]):
            self.fail()

    def test_comp_expansion(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [3., -1., 0., 2., 0., 1.]
        expansion.coeff = coeff

        lhs = expansion

        expansion1 = factory_taylor.zero_expansion(1, 2)
        coeff1 = [0., 3., 4.]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(1, 2)
        coeff2 = [0., -1., 2.]
        expansion2.coeff = coeff2

        rhs = taylor_map.RealTaylorMap([expansion1, expansion2])

        composed = lhs(rhs)
        coeff = composed.coeff

        if not np.array_equal(coeff, [3., -3., 15.]):
            self.fail()

    def test_real_root_coeff(self):
        order = 4
        root1_nominal = 2.
        root2_nominal = -3.
        real_map = factory_taylor.create_unknown_map(order, [root1_nominal, root2_nominal])
        root1 = real_map[0]
        root2 = real_map[1]

        bc = taylor_map.RealTaylorMap([-(root1 + root2), root1 * root2])
        roots_from_inversion = bc.get_nilpo_part().compo_inverse()
        roots_from_inversion.const = [root1_nominal, root2_nominal]

        root1_from_inversion = roots_from_inversion[0]
        root2_from_inversion = roots_from_inversion[1]

        b = real_map[0].copy()
        b.const = -(root2_nominal + root1_nominal)
        c = real_map[1].copy()
        c.const = root2_nominal * root1_nominal
        sqrt_delta = sqrt(b**2 - 4. * c)
        root1_formula = 0.5 * (sqrt_delta - b)
        root2_formula = (-0.5) * (b + sqrt_delta)

        coeff1_formula = root1_formula.coeff
        coeff1_from_inversion = root1_from_inversion.coeff
        coeff2_formula = root2_formula.coeff
        coeff2_from_inversion = root2_from_inversion.coeff
        if not np.allclose(coeff1_formula, coeff1_from_inversion):
            self.fail()
        if not np.allclose(coeff2_formula, coeff2_from_inversion):
            self.fail()

    def test_complex_root_coeff(self):
        order = 4

        root1_nominal = complex(0., 1.)
        root2_nominal = complex(0., -1.)
        complex_map = factory_taylor.create_unknown_map(order, [root1_nominal, root2_nominal])

        root1 = complex_map[0].copy()
        root2 = complex_map[1].copy()

        bc = taylor_map.ComplexTaylorMap([-(root1 + root2), root1 * root2])
        roots_from_inversion = bc.get_nilpo_part().compo_inverse()

        root1_from_inversion = roots_from_inversion[0] + root1_nominal
        root2_from_inversion = roots_from_inversion[1] + root2_nominal

        b = complex_map[0].copy()
        b.const = -(root2_nominal + root1_nominal)
        c = complex_map[1].copy()
        c.const = root2_nominal * root1_nominal
        delta = b ** 2 - 4. * c
        sqrt_delta = sqrt(delta)

        root1_formula = 0.5 * (sqrt_delta - b)
        root2_formula = (-0.5) * (b + sqrt_delta)

        if root1_from_inversion != root1_formula:
            self.fail()
        if root2_from_inversion != root2_formula:
            self.fail()


if __name__ == '__main__':
    unittest.main()
