import unittest
import numpy as np
from swiftt.taylor import taylor_map, factory_taylor


tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


class TestBasicOperation(unittest.TestCase):

    def test_eval_univariate(self):
        expansion = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff = [2., -3., 5., 4]
        expansion.coeff = coeff

        if expansion.pointwise_eval(2.) != 48. or expansion(2.) != 48.:
            self.fail()

    def test_add_expansion_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff2 = [1., 7., -6., 0.]
        expansion2.coeff = coeff2

        sum = expansion1 + expansion2
        coeff = sum.coeff

        if not np.array_equal(coeff, np.array(coeff1) + np.array(coeff2)):
            self.fail()

    def test_add_const_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        expansion2 = expansion1 + 6.
        coeff = expansion2.coeff

        if coeff[0] != 8.:
            self.fail()
        if not np.array_equal(coeff[1:], coeff1[1:]):
            self.fail()

    def test_linearly_combine_expansion_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff2 = [1., 7., -6., 0.]
        expansion2.coeff = coeff2

        a = 2.
        b = -1.
        expansion = expansion1.linearly_combine_with_another(a, expansion2, b)
        coeff = expansion.coeff

        if not np.array_equal(coeff, a * np.array(coeff1) + b * np.array(coeff2)):
            self.fail()

        expansion = expansion1.linearly_combine_with_many(a, [expansion2], [b])
        coeff = expansion.coeff

        if not np.array_equal(coeff, a * np.array(coeff1) + b * np.array(coeff2)):
            self.fail()

    def test_sub_expansion_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff2 = [1., 7., -6., 0.]
        expansion2.coeff = coeff2

        diff = expansion1 - expansion2
        coeff = diff.coeff

        if not np.array_equal(coeff, np.array(coeff1) - np.array(coeff2)):
            self.fail()

    def test_sub_const_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        a = 6.
        expansion2 = expansion1 - a
        coeff = expansion2.coeff

        if coeff[0] != coeff1[0] - a:
            self.fail()
        if not np.array_equal(coeff1[1:], coeff[1:]):
            self.fail()

    def test_radd_const_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [2., -3., 5., 4, 0., 1.]
        expansion1.coeff = coeff1

        a = 1
        sum = a + expansion1
        coeff = sum.coeff

        if coeff[0] != coeff1[0] + a:
            self.fail()
        if not np.array_equal(coeff1[1:], coeff[1:]):
            self.fail()

    def test_rsub_const_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [2., -3., 5., 4, 0., 1.]
        expansion1.coeff = coeff1

        a = 1.
        diff = a - expansion1
        coeff = diff.coeff

        if coeff[0] != a - coeff1[0]:
            self.fail()
        if not np.array_equal(coeff1[1:], -np.array(coeff[1:])):
            self.fail()

    def test_rmul_const_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        factor = -2.
        mul = factor * expansion1
        coeff = mul.coeff

        if not np.array_equal(coeff, factor * np.array(coeff1)):
            self.fail()

    def test_mul_expansion_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff2 = [1., 7., -6., 0.]
        expansion2.coeff = coeff2

        mul = expansion1 * expansion2
        coeff = mul.coeff

        if coeff[0] != 2. or coeff[1] != 11. or coeff[2] != -28. or coeff[3] != 57.:
            self.fail()

    def test_div_scalar_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [2., -3., 5., 4]
        expansion1.coeff = coeff1

        factor = 2.
        div = expansion1 / factor
        coeff = div.coeff

        if not np.array_equal(coeff, np.array(coeff1) / factor):
            self.fail()

    def test_div_expansion_univariate(self):
        expansion1 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff1 = [1., 0., 0., 0]
        expansion1.coeff = coeff1

        expansion2 = factory_taylor.zero_expansion(n_var=1, order=3)
        coeff2 = [0.5, 1., 0., 0.]
        expansion2.coeff = coeff2

        div = expansion1 / expansion2
        coeff = div.coeff

        if coeff[0] != 2. or coeff[1] != -4. or coeff[2] != 8. or coeff[3] != -16.:
            self.fail()

    def _template_div(self, univar: bool):
        expansion1 = factory_taylor.zero_expansion(1, 9) if univar else null_expansion_2var_order3
        coeff1 = [2., -3., 5., 4, 0., 1., -2., 6., -1., 0.]
        expansion1.coeff = coeff1

        expansion2 = expansion1 ** (-1)

        expansion3 = 1. / expansion1

        expansion4 = expansion1.reciprocal()

        if expansion2 != expansion3 or expansion2 != expansion4:
            self.fail()
        if expansion1 * expansion4 != expansion1.create_const_expansion(1.):
            self.fail()

    def test_div_univariate(self):
        self._template_div(univar=True)

    def test_div_bivariate(self):
        self._template_div(univar=False)

    def _template_pown(self, univar: bool):
        expansion1 = factory_taylor.zero_expansion(1, 9) if univar else null_expansion_2var_order3
        coeff1 = [2., -3., 5., 4, 0.5, 1., -2., 6., -1., 0.]
        expansion1.coeff = coeff1

        for i in range(4, 6):
            product = expansion1.copy()
            expansion2 = expansion1**i
            for __ in range(i - 1):
                product *= expansion1
            if product != expansion2:
                self.fail()

    def test_pown_univariate(self):
        self._template_pown(univar=True)

    def test_pown_bivariate(self):
        self._template_pown(univar=False)

    def _template_pow3(self, univar: bool):
        expansion1 = factory_taylor.zero_expansion(1, 9) if univar else null_expansion_2var_order3
        coeff1 = [2., -3., 5., 4, 0.5, 1., -2., 6., -1., 0.]
        expansion1.coeff = coeff1

        product = expansion1**2
        expansion2 = expansion1**3
        product *= expansion1
        if product != expansion2:
            self.fail()

    def test_pow3_univariate(self):
        self._template_pow3(univar=True)

    def test_pow3_bivariate(self):
        self._template_pow3(univar=False)

    def _template_pow(self, univar: bool):
        expansion1 = factory_taylor.zero_expansion(1, 9) if univar else null_expansion_2var_order3.copy()
        coeff1 = [2., -3., 5., 4, 0.5, 1., -2., 6., -1., 0.]
        expansion1.coeff = coeff1

        product = expansion1.copy()
        for i in range(2, 6):
            expansion2 = expansion1**float(i)
            product *= expansion1
            if product != expansion2:
                self.fail()

    def test_pow_univariate(self):
        self._template_pow(univar=True)

    def test_pow_bivariate(self):
        self._template_pow(univar=False)

    def _template_test_complex_versus_float(self, expans, expans_complex):
        for func in (lambda x: x**2, lambda x: 1. / x):
            eval_func = func(expans)
            eval_func_complex = func(expans_complex)
            if not np.array_equal(eval_func.coeff, np.real(eval_func_complex.coeff)):
                self.fail()

    def test_univar_complex_versus_float(self):
        nvar = 1
        order = 10
        expans = factory_taylor.zero_expansion(nvar, order=order)
        coeff = np.zeros(order + 1)
        coeff[0] = coeff[1] = 1.
        expans.coeff = coeff
        expans_complex = factory_taylor.zero_expansion(nvar, order=order, dtype=complex)
        coeff_complex = np.zeros(order + 1, dtype=complex)
        coeff_complex[0] = coeff_complex[1] = 1.
        expans_complex.coeff = coeff_complex
        self._template_test_complex_versus_float(expans, expans_complex)

    def test_multivar_complex_versus_float(self):
        nvar = 4
        order = 5
        expans = factory_taylor.zero_expansion(nvar, order=order)
        coeff = np.zeros(expans.dim_alg)
        coeff[:order + 1] = 1.
        expans.coeff = coeff
        expans_complex = factory_taylor.zero_expansion(nvar, order=order, dtype=complex)
        coeff_complex = np.zeros(expans.dim_alg, dtype=complex)
        coeff_complex[:order + 1] = 1.
        expans_complex.coeff = coeff_complex
        self._template_test_complex_versus_float(expans, expans_complex)

    def test_lin_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5)
        coeff1 = [2., -3., 5., 4, 0., 1.]
        expansion1.coeff = coeff1

        expansion2 = expansion1.get_linear_part()
        coeff = expansion2.coeff

        if coeff[1] != coeff1[1]:
            self.fail()
        if not np.array_equal(coeff[2:], np.zeros(len(coeff) - 2)):
            self.fail()

    def test_lin_nonlin(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., 1.5, -4., 2, -1., 7.]
        expansion1.coeff = coeff1

        if expansion1 != 3. + expansion1.get_linear_part() + expansion1.get_nonlin_part():
            self.fail()
        if expansion1 != expansion1.get_affine_part() + expansion1.get_nonlin_part():
            self.fail()

    def test_add_complex_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 5, ["x"], np.complex128)
        coeff1 = [complex(3., 1.), 1.5, -4., 2, -1., 6.]
        expansion1.coeff = coeff1

        el = complex(-1.5, 10.)
        added = expansion1 + el
        coeff = added.coeff
        if coeff[0] != coeff1[0] + el:
            self.fail()
        if not np.array_equal(coeff[1:], coeff1[1:]):
            self.fail()

    def test_mul_complex_univariate(self):
        expansion1 = factory_taylor.zero_expansion(1, 3, ["x"], np.complex128)
        coeff1 = [complex(3., 2.), 1.5, -4., 2.]
        expansion1.coeff = coeff1

        factor = complex(-2., 3.)
        multiplied = expansion1 * factor
        coeff = multiplied.coeff

        if not np.array_equal(coeff, factor * np.array(coeff1)):
            self.fail()

    def test_multiplicative_inverse_univariate(self):
        expansion = factory_taylor.zero_expansion(n_var=1, order=10)
        expansion.coeff = [2., 1., 3., -1., 5., -4., 7., -0.5, 1., -3., 8.]

        inv_const = 1. / expansion.const
        scaled_expans = expansion * inv_const
        iter_mul_inverse = 2. - scaled_expans
        for __ in range(1, expansion.order // 2 + 1):
            iter_mul_inverse *= 2. - scaled_expans * iter_mul_inverse
        iter_mul_inverse *= inv_const

        if not np.array_equal(iter_mul_inverse.coeff, (1. / expansion).coeff):
            self.fail()

    def test_multiplicative_inverse_bivariate(self):
        expansion = null_expansion_2var_order3.copy()
        expansion.coeff = [2., 1., 3., -1., 5., -4., 7., -0.5, 1., 2.]

        inv_const = 1. / expansion.const
        scaled_expans = expansion * inv_const
        iter_mul_inverse = 2. - scaled_expans
        for __ in range(1, expansion.order // 2 + 1):
            iter_mul_inverse *= 2. - scaled_expans * iter_mul_inverse
        iter_mul_inverse *= inv_const

        if not np.array_equal(iter_mul_inverse.coeff, (1. / expansion).coeff):
            self.fail()

    # def test_complex_map_inversion(self):
    #     order = 20
    #     z = complex(0.5, 1.)
    #     complex_map = factory.create_unknown_map(order, [z])
    #     square = expansion_map.ComplexMap([complex_map[0] * complex_map[0]])
    #     inverted = square.get_nilpo_part().inverse()
    #     composed = square(inverted)
    #     coeff = composed[0].coeff
    #     if cnp.polar(coeff[1] - complex(1., 0.))[0] > tol_coeff:
    #         return self.fail()
    #     for i in range(2, len(coeff)):
    #         if cnp.polar(coeff[i])[0] > tol_coeff:
    #             self.fail()

    def test_pow2_bivariate(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        if expansion1**2 != expansion1 * expansion1:
            self.fail()

    def test_divide_by_var(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [0., 0., 4., 0., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [4., -1., -3., 0., 0., 0.]
        expansion2.coeff = coeff2

        if expansion1.divided_by_var(1) != expansion2:
            self.fail()

    def test_eval_bivariate(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        x = [1., -2.]
        if expansion1.pointwise_eval(x) != -18. or expansion1(x) != -18.:
            self.fail()

    def test_massive_eval_univariate(self):
        expans = factory_taylor.zero_expansion(1, 6).create_expansion_with_coeff([1., 2., -3., 4., 5., -6., 7.])
        Xs = np.array([0., 1., -5., 10., -2.])
        evaluated = expans.massive_eval(Xs)
        for el, x in zip(evaluated, Xs):
            if expans.pointwise_eval(x) != el:
                self.fail()

    def test_massive_eval_trivariate(self):
        expansion1 = null_expansion_2var_order3.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        x = np.random.rand(4, 2)
        eval_all = expansion1.massive_eval(x)
        for i, el in enumerate(eval_all):
            if np.fabs(el - expansion1.pointwise_eval(x[i, :])) > 1.e-8:
                print(expansion1.pointwise_eval(x[i, :]))
                self.fail()

    def test_massive_eval_map(self):
        expansion1 = null_expansion_3var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3., -2., 1., 6., -7.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_3var_order2.copy()
        coeff2 = [1., -2., 1., 4., 2., 0., -6., 1., -2., 3.]
        expansion2.coeff = coeff2

        list_expansion = [expansion1, expansion2]
        map_expansion = taylor_map.RealTaylorMap(list_expansion)

        x = np.random.rand(5, 3)
        eval_both = map_expansion.massive_eval(x)
        for i, expans in enumerate(list_expansion):
            for j, el in enumerate(x):
                evaluated = expans.pointwise_eval(el)
                if np.fabs(eval_both[j, i] - evaluated) > 1.e-8:
                    self.fail()

    def test_mul_bivariate(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-1., 0., 1., 2., 3., -1.]

        expansion1.coeff = coeff1
        expansion2.coeff = coeff2

        expansion = expansion1 * expansion2
        coeff = expansion.coeff

        if not np.array_equal(coeff, [-2., -3., -2., 9., 10., 5.]):
            self.fail()

    def test_add_maps(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])
        map2 = taylor_map.RealTaylorMap([expansion2, expansion1])

        sum1 = map1 + map2
        sum2 = map2 + map1
        if sum1 != sum2:
            return self.fail()

    def test_sub_maps(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])
        map2 = taylor_map.RealTaylorMap([expansion2, expansion1])

        diff1 = map1 - map2
        diff2 = map2 - map1
        if diff1 != -diff2:
            return self.fail()

    def test_add_map_nparray(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        added = np.array([2., 4.])
        sum1 = map1 + added
        map1.const = map1.const + added
        if sum1 != map1:
            self.fail()

    def test_add_map_float(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        added = 6.
        sum1 = map1 + added
        map1.const = map1.const + np.ones(2) * added
        if sum1 != map1:
            self.fail()

    def test_add_map_expansion(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        expansion = null_expansion_2var_order2.copy()
        coeff = [2., 1., -3., 4., 0., 7.]
        expansion.coeff = coeff

        sum1 = map1 + expansion
        if sum1 != taylor_map.RealTaylorMap([expansion1 + expansion, expansion2 + expansion]):
            self.fail()

    def test_sub_map_nparray(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        subtracted = np.array([2., 4.])
        sum1 = map1 - subtracted
        map1.const = map1.const - subtracted
        if sum1 != map1:
            self.fail()

    def test_sub_map_float(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        subtracted = 6.
        sum1 = map1 - subtracted
        map1.const = map1.const - np.ones(2) * subtracted
        if sum1 != map1:
            self.fail()

    def test_sub_map_expansion(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        expansion = null_expansion_2var_order2.copy()
        coeff = [2., 1., +3., 4., 0., 7.]
        expansion.coeff = coeff

        sub = map1 - expansion
        if sub != taylor_map.RealTaylorMap([expansion1 - expansion, expansion2 - expansion]):
            self.fail()

    def test_mul_map_float(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        factor = 6.
        product = map1 * factor
        if product != taylor_map.RealTaylorMap([factor * expansion1, factor * expansion2]):
            self.fail()

    def test_mul_map_expansion(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map1 = taylor_map.RealTaylorMap([expansion1, expansion2])

        expansion = null_expansion_2var_order2.copy()
        coeff = [2., 1., -3., -4., 0., 7.]
        expansion.coeff = coeff

        product = map1 * expansion
        if product != taylor_map.RealTaylorMap([expansion1 * expansion, expansion2 * expansion]):
            self.fail()

    def test_pow2_map_expansion(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        squared_map = taylor_map.RealTaylorMap([expansion1, expansion2]) ** 2
        if squared_map[0] != expansion1 ** 2:
            self.fail()
        if squared_map[1] != expansion2 ** 2:
            self.fail()

    def test_eval_map(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [3., -1., 4., 2., 5., 1.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map = taylor_map.RealTaylorMap([expansion1, expansion2])

        eval0 = map.pointwise_eval(np.zeros(2))
        expected = [coeff1[0], coeff2[0]]
        for i, el in enumerate(map.const):
            if el != eval0[i] or el != expected[i]:
                self.fail()

        x = np.array([0.5, 3.])
        evalx = map.pointwise_eval(x)
        for i, el in enumerate(evalx):
            if el != map[i].pointwise_eval(x):
                self.fail()

    def test_truediv_map_float(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        factor = 2.
        map_expansion = taylor_map.RealTaylorMap([expansion1, expansion2])
        divided_map = map_expansion / factor

        for i, expansion in enumerate(divided_map):
            if expansion != map_expansion[i] / factor:
                self.fail()

    def test_truediv_map_expansion(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        expansion = null_expansion_2var_order2.copy()
        coeff = [2., 1., -3., -4., 0., 7.]
        expansion.coeff = coeff

        map_expansion = taylor_map.RealTaylorMap([expansion1, expansion2])
        divided_map = map_expansion / expansion

        for i, new_expansion in enumerate(divided_map):
            if new_expansion != map_expansion[i] / expansion:
                self.fail()

    def test_dot_map_float(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map_expansion = taylor_map.RealTaylorMap([expansion1, expansion2])

        dotted = np.array([2., -1.])
        if map_expansion.dot(dotted) != dotted[0] * expansion1 + dotted[1] * expansion2:
            return self.fail()

    def test_dot_map_pow2(self):
        expansion1 = null_expansion_2var_order2.copy()
        coeff1 = [2., 3., 4., -5., -1., -3.]
        expansion1.coeff = coeff1

        expansion2 = null_expansion_2var_order2.copy()
        coeff2 = [-2., 6., 3., -2., 1., -1.]
        expansion2.coeff = coeff2

        map_expansion = taylor_map.RealTaylorMap([expansion1, expansion2])

        if map_expansion.dot(map_expansion) != expansion1**2 + expansion2**2:
            return self.fail()


if __name__ == '__main__':
    unittest.main()
