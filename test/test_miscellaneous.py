import unittest
import numpy as np
from swiftt.interval import Interval
from swiftt.taylor import taylor_map, factory_taylor
from swiftt.math_algebra import exp


tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


class TestMiscellaneous(unittest.TestCase):

    def test_depth_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 6)
        coeff1 = [0., 0., 0., 4, 0., 5., 8.]
        expansion1.coeff = coeff1

        if expansion1.create_null_expansion().total_depth != 7:
            self.fail()
        if expansion1.total_depth != 3:
            self.fail()

    def test_depth_bivariate(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [0., 0., 0., 4, 0., 5.]
        expansion1.coeff = coeff1

        if expansion1.create_null_expansion().total_depth != expansion1.order + 1:
            self.fail()
        if expansion1.total_depth != 2:
            self.fail()

    def test_effective_order_bivariate(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 1., 0., 0, 0., 0.]
        expansion1.coeff = coeff1

        if expansion1.create_null_expansion().effective_order != 0:
            self.fail()
        if expansion1.effective_order != 1:
            self.fail()

    def test_const_nilpo(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., 1.5, -4., 2, -1., 7.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.get_const_part() + expansion1.get_nilpo_part()

        if expansion1 != expansion2:
            self.fail()

    def test_low_high_order_part(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [3., 1.5, -4., 2., -1., 7., 0., -3., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.get_low_order_part(2) + expansion1.get_high_order_part(3)

        if expansion1 != expansion2:
            self.fail()

    def test_low_order_part_map(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        if map1.get_low_order_part(1) != taylor_map.RealTaylorMap([expansion1.get_low_order_part(1),
                                                                   expansion2.get_low_order_part(1)]):
            self.fail()

    def test_high_order_part_map(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        if map1.get_high_order_part(2) != taylor_map.RealTaylorMap([expansion1.get_high_order_part(2),
                                                                    expansion2.get_high_order_part(2)]):
            self.fail()

    def test_bounder_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [-0.5, 1.5, -4., 2., 1., 3.]
        expansion1.coeff = coeff1

        bounds = expansion1.bounder(Interval(-1., 1.))

        expected_lb = coeff1[0] - coeff1[1] + coeff1[2] - coeff1[3] - 0. - coeff1[5]
        expected_ub = coeff1[0] + coeff1[1] + 0. + coeff1[3] + coeff1[4] + coeff1[5]
        if bounds.lb != expected_lb:
            self.fail()
        if bounds.ub != expected_ub:
            self.fail()

    def test_bounder_univariate_map(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [-0.5, 1.5, -4., 2., 1., 3.]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(1, 5)
        coeff2 = [2., 1., 4., 0., 8., -5.]
        expansion2.coeff = coeff2

        map_expans = taylor_map.RealTaylorMap([expansion1, expansion2])

        bounds1 = expansion1.bounder(Interval(-1., 1.))
        bounds2 = expansion2.bounder(Interval(-1., 1.))
        bounds_map = map_expans.bounder([Interval(-1., 1.)])

        if bounds1 != bounds_map[0]:
            self.fail()
        if bounds2 != bounds_map[1]:
            self.fail()

    def test_bounder_bivariate_map(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order3.copy()
        coeff2 = [0., -1., 2., -2., 4., 5., 2., -1., -7., 8.]
        expansion2.coeff = coeff2

        intervals = [Interval(-1., 1.), Interval(-2., 3.)]

        bounders_map = taylor_map.RealTaylorMap([expansion1, expansion2]).bounder(intervals)

        if bounders_map[0] != expansion1.bounder(intervals):
            self.fail()

        if bounders_map[1] != expansion2.bounder(intervals):
            self.fail()

    def test_truncated_prolong_univariate(self):
        old_order = 5
        expansion1 = factory_taylor.zero_expansion(1, old_order)
        coeff1 = [3., 1.5, -4., 2, -1., 6.]
        expansion1.coeff = coeff1

        new_order = 2
        trunc = expansion1.truncated(new_order)
        same = trunc.prolong(expansion1.order)
        coeff = same.coeff

        if not np.array_equal(coeff[:new_order + 1], coeff1[:new_order + 1]):
            self.fail()
        if not np.array_equal(coeff[new_order + 1:], np.zeros(old_order - new_order)):
            self.fail()

    def test_remove_contribution(self):
        expansion = null_expansion_2var_order2.copy()
        coeff = [0., 3., 4., -5., 0., -3.]

        expansion.coeff = coeff

        expansion1 = expansion.contrib_removed([0])
        expansion2 = expansion.contrib_removed([1])

        if expansion1 + expansion2 != expansion:
            self.fail()

    def test_remove_contributions(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        if expansion1.pointwise_eval([1., 0.]) != expansion1.contrib_removed([1]).pointwise_eval([1., 1.]):
            self.fail()

    def test_remove_contributions2(self):
        expansion1 = null_expansion_4var_order5.copy()
        expansion1.coeff = np.random.rand(expansion1.dim_alg)

        if expansion1.contrib_removed([1, 2]) != expansion1.contrib_removed([2, 1]):
            self.fail()

    def test_remove_last_contribution_map(self):

        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        if map1.last_contrib_removed() != taylor_map.RealTaylorMap([expansion1.last_contrib_removed(),
                                                                    expansion2.last_contrib_removed()]):
            self.fail()

    def test_remove_dependency(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        dx1 = -3.
        dx2 = 2.
        if expansion1.var_eval(0, dx1).var_eval(1, dx2).const != \
                expansion1.pointwise_eval([dx1, dx2]):
            self.fail()

    def test_remove_last_dependency(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order3.copy()
        coeff2 = [-58., 25., 0., -3., 0., 0., -2., 0., 0., 0.]
        expansion2.coeff = coeff2

        if expansion1.last_var_eval(2.) != expansion2:
            self.fail()

    def test_remove_dependency_contribution(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        if expansion1.last_contrib_removed() != expansion1.last_var_eval(0.):
            self.fail()

    def test_remove_last_dependency_map(self):

        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        dx2 = 2.
        if map1.last_var_eval(dx2) != taylor_map.RealTaylorMap([expansion1.last_var_eval(dx2),
                                                                expansion2.last_var_eval(dx2)]):
            self.fail()

    def test_gradient(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        grad = [coeff1[1], coeff1[2]]
        for i, el in enumerate(expansion1.gradient):
            if el != grad[i]:
                self.fail()

    def test_hessian(self):
        n_var = 2
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [3., -1., 4., 2., 5., 3., 0., 0., 1., -2.]
        expansion1.coeff = coeff1

        expected_hessian = np.array([[2. * coeff1[3], coeff1[4]], [coeff1[4], 2. * coeff1[5]]])
        hessian = expansion1.hessian

        for i in range(0, n_var):
            for j in range(0, n_var):
                if hessian[i, j] != expected_hessian[i, j]:
                    self.fail()

    def test_monomial(self):
        expansion = null_expansion_2var_order3.copy()

        coeff1 = [0., 0., 0., 0., 0., 0., 0., 0., 5., 0.]
        expansion1 = expansion.create_expansion_with_coeff(coeff1)

        expansion2 = expansion.create_monomial_expansion_from_exponent([1, 2], 5.)

        if expansion1 != expansion2:
            self.fail()

    def test_truncate_bivariate(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.truncated(2)
        coeff2 = expansion2.coeff

        coeff = expansion1.get_low_order_part(2).coeff

        if not np.array_equal(coeff1[:len(coeff2)], coeff[:len(coeff2)]):
            self.fail()
        if not np.array_equal(coeff2[:len(coeff2)], coeff[:len(coeff2)]):
            self.fail()
        if len(coeff2) > expansion2.dim_alg:
            self.fail()

    def test_truncate_prolong_trivariate(self):
        expansion = null_expansion_3var_order2.copy()
        coeff = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion.coeff = coeff

        expansion2 = expansion.truncated(expansion.order - 1).prolong(expansion.order)

        if expansion != expansion2 + expansion.get_high_order_part(expansion.order):
            self.fail()

    def test_prolong_trivariate(self):
        expansion = null_expansion_3var_order2.copy()
        coeff = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion.coeff = coeff

        if expansion.prolong_one_order() != expansion.prolong(expansion.order + 1):
            self.fail()

    def test_truncate_taylor(self):
        expansion1 = null_expansion_4var_order5.copy()
        coeff = np.random.rand(expansion1.dim_alg)
        expansion1.coeff = coeff
        expansion2 = expansion1.truncated(expansion1.order - 1)

        def fun(p):
            return exp(p**0.8 - 3. / p)

        expansion1 = fun(expansion1)
        expansion1 = expansion1.truncated(expansion1.order - 1)
        expansion2 = fun(expansion2)

        coeff1 = expansion1.coeff
        coeff2 = expansion2.coeff

        if not np.array_equal(coeff2[:len(coeff1)], coeff1):
            self.fail()

    def test_remove_unknown_univariate(self):
        expansion = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff = [2., -3., 5., 4]
        expansion.coeff = coeff

        expansion2 = null_expansion_2var_order3.copy()
        expansion2.coeff = [2., -3., 0., 5., 0., 0., 4., 0., 0., 0.]

        if expansion2.last_var_removed() != expansion:
            self.fail()

    def test_append_unknown_univariate(self):
        expansion = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff = [2., -3., 5., 4]
        expansion.coeff = coeff

        expansion2 = null_expansion_2var_order3.copy()
        expansion2.coeff = [2., -3., 0., 5., 0., 0., 4., 0., 0., 0.]

        if expansion.var_appended("x2") != expansion2:
            self.fail()

    def test_append_unknown_bivariate(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.var_appended("x3")
        coeff2 = expansion2.coeff

        if not np.array_equal(coeff1[:3], coeff2[:3]):
            self.fail()
        if coeff2[4] != -5. or coeff2[5] != -1. or coeff2[7] != -3.:
            self.fail()
        if coeff2[3] != 0 or coeff2[6] != 0 or coeff2[8] != 0 or coeff2[9] != 0:
            self.fail()

    def test_another_unknown(self):
        expansion1 = null_expansion_3var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.var_appended("x4")
        expansion = expansion2.last_var_removed()

        if expansion != expansion1:
            self.fail()

    def test_remove_last_unknown(self):
        expansion1 = null_expansion_3var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [2., 3., 4., -1., -3., 1.]
        expansion2.coeff = coeff2

        if expansion1.last_var_removed() != expansion2:
            self.fail()

    def test_expansion_from_smaller_algebra(self):
        expansion = factory_taylor.zero_expansion(1, 3, ["x1"])
        coeff1 = [1., 2., 3., -4.]
        expansion.coeff = coeff1

        expansion = null_expansion_2var_order3.copy().create_expansion_from_smaller_algebra(expansion)
        coeff2 = expansion.coeff
        expected_coeff = [coeff1[0], coeff1[1], 0., coeff1[2], 0., 0., coeff1[3], 0., 0., 0.]
        if not np.array_equal(coeff2, expected_coeff):
            self.fail()

    def test_map_from_smaller_algebra(self):
        expansion1 = factory_taylor.zero_expansion(1, 3, ["x1"])
        coeff1 = [1., 2., 3., -4.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.copy()
        coeff2 = [0., -2., 0., 5.]
        expansion2.coeff = coeff2

        expansions = taylor_map.RealTaylorMap([expansion1, expansion2])
        new_expansions = taylor_map.RealTaylorMap([null_expansion_2var_order3.copy()]).create_map_from_smaller_algebra(expansions)
        for i, expansion in enumerate(new_expansions):
            if expansion != null_expansion_2var_order3.copy().create_expansion_from_smaller_algebra(expansions[i]):
                self.fail()


if __name__ == '__main__':
    unittest.main()
