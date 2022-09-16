# real_multivar_taylor.py: class implementing Taylor expansions of multiple real variables
# Copyright 2022 Romain Serra

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Iterable, Union
import math
import numpy as np
from swiftt.taylor.complex_multivar_taylor import TaylorExpansAbstract, ComplexMultivarTaylor
from swiftt.taylor.taylor_map import RealTaylorMap
from swiftt.interval import Interval


class RealMultivarTaylor(ComplexMultivarTaylor):
    """Class for Taylor expansions of real variables.

    """

    # intrinsic functions for reals
    _sqrt_cst = math.sqrt
    _log_cst, _exp_cst = math.log, math.exp
    _cos_cst, _sin_cst = math.cos, math.sin
    _cosh_cst, _sinh_cst = math.cosh, math.sinh
    _tan_cst, _tanh_cst = math.tan, math.tanh

    def __init__(self, n_var: int, order: int, var_names: List[str]) -> None:
        """Constructor for Taylor expansions of multiple real variables. Result has no assigned coefficients.

        Args:
             n_var (int): number of variables in algebra.
             order (int): order of algebra.
             var_names (List[str]): name of variables in algebra.

        """
        TaylorExpansAbstract.__init__(self, n_var, order, np.float64, var_names)

    def __float__(self) -> float:
        """Method to cast Taylor expansion as a scalar (according to its constant part).

        Returns:
            float: constant part of expansion seen as a scalar.

        """
        return float(self.const)

    def __int__(self) -> int:
        """Method to cast Taylor expansion as a integer (according to its constant part).

        Returns:
            int: constant part of expansion casted as an integer.

        """
        return int(self.const)

    @classmethod
    def from_complex_expansion(cls, expansion: ComplexMultivarTaylor) -> "RealMultivarTaylor":
        """Method to recast a complex Taylor expansion into a real one.

        Args:
            expansion (ComplexMultivarTaylor): original Taylor expansion.

        Returns:
            RealMultivarTaylor: Taylor expansion where all coefficients are real parts of the input's.

        """
        real_expansion = cls(expansion.n_var, expansion.order, expansion.var_names)
        real_expansion.coeff = np.real(expansion.coeff)
        return real_expansion

    def __str__(self) -> str:
        """Method to cast Taylor expansion as a string. Overwrites parent implementation assuming complex coefficients.

        Returns:
            str: string representing the Taylor expansion.

        """
        string = str(self.const)
        var_names = self.var_names
        if not self.is_trivial():
            for exponent, index_coeff in self.get_mapping_monom().items():
                if self._coeff[index_coeff] != 0. and index_coeff != 0:
                    sign = " + " if self._coeff[index_coeff] > 0. else " - "
                    string += sign + str(abs(self._coeff[index_coeff]))
                    for index_var, power_var in enumerate(exponent):
                        if power_var != 0:
                            string += " * " + var_names[index_var]
                            if power_var > 1:
                                string += "**" + str(power_var)
        return string + " + " + self.remainder_term

    def bounder(self, domains: Iterable[Interval]) -> Interval:
        """Method to evaluate the polynomial part of the Taylor expansion on a Cartesian product of segments via
        interval arithmetic.

        Args:
            domains (Iterable[Interval]): input intervals for expansion's variables.

        Returns:
            Interval: image of inputted intervals through polynomial part.

        """

        if self.is_trivial():
            return Interval.singleton(self.const)

        # order of at least one
        output = Interval.singleton(self.const)
        for i, domain in enumerate(domains, 1):
            output += self._coeff[i] * domain
        # non-linear terms
        if self._order > 1:
            powers = []
            for domain in domains:
                powers_xi = [1., domain]
                for j in range(2, self._order + 1):
                    powers_xi.append(domain ** j)
                powers.append(powers_xi)

            for exponent, index_coeff in self.get_mapping_monom().items():
                if self._coeff[index_coeff] != 0. and index_coeff > self._n_var:
                    product = powers[0][exponent[0]]
                    for index_var, power_var in enumerate(exponent[1:], 1):
                        product *= powers[index_var][power_var]
                    output += self._coeff[index_coeff] * product

        return output

    def __call__(self, *args, **kwargs):
        """Method for calling the Taylor expansion. Wraps several possibilities: evaluation on a point or a Cartesian
        product of intervals as well as composition with a map.

        Returns:
            Union[RealTaylorMap, Interval, numpy.ndarray]: Taylor expansion called on input.

        """
        if isinstance(args[0], RealTaylorMap):
            # composition
            return RealMultivarTaylor.compose(self, *args, **kwargs)
        if isinstance(args[0][0], Interval):
            # bounder
            return self.bounder(args[0])
        # evaluation
        return self.pointwise_eval(args[0])

    def __lt__(self, other: Union["RealMultivarTaylor", float]) -> bool:
        """Method enabling "<" inequality comparison with other Taylor expansions and scalars.

        Args:
            other (Union[RealMultivarTaylor, float])): quantity to be compared with.

        Returns:
            bool: if the input is a Taylor expansion in same algebra, compares coefficients pairwise
                (according to sorted monomials) until one is strictly smaller than the other. If the input is a
                scalar, performs comparison on Taylor expansion with only a constant part equal to it.

        """
        if isinstance(other, RealMultivarTaylor):
            if self.is_in_same_algebra(other):
                for coeff1, coeff2 in zip(self._coeff, other._coeff):
                    if coeff1 != coeff2:
                        return coeff1 < coeff2
            return False

        # scalar case
        return self < self.create_const_expansion(other)

    def __le__(self, other: Union["RealMultivarTaylor", float]) -> bool:
        """Method enabling "<=" inequality comparison with other Taylor expansions and scalars.

        Args:
            other (Union[RealMultivarTaylor, float])): quantity to be compared with.

        Returns:
            bool: returns logical union of "=" and "<".

        """
        return not self.__gt__(other)

    def __gt__(self, other: Union["RealMultivarTaylor", float]) -> bool:
        """Method enabling ">" inequality comparison with other Taylor expansions and scalars.

        Args:
            other (Union[RealMultivarTaylor, float])): quantity to be compared with.

        Returns:
            bool: if the input is a Taylor expansion in same algebra, compares coefficients pairwise
                (according to sorted monomials) until one is strictly greater than the other. If the input is a
                scalar, performs comparison on Taylor expansion with only a constant part equal to it.

        """
        if isinstance(other, RealMultivarTaylor):
            if self.is_in_same_algebra(other):
                for coeff1, coeff2 in zip(self._coeff, other._coeff):
                    if coeff1 != coeff2:
                        return coeff1 > coeff2
            return False

        # scalar case
        return self > self.create_const_expansion(other)

    def __ge__(self, other: Union["RealMultivarTaylor", float]) -> bool:
        """Method enabling ">=" inequality comparison with other Taylor expansions and scalars.

        Args:
            other (Union[RealMultivarTaylor, float])): quantity to be compared with.

        Returns:
            bool: returns logical union of "=" and ">".

        """
        return not self.__lt__(other)

    def __abs__(self) -> "RealMultivarTaylor":
        """Method implementation the absolute value for real Taylor expansions.

        Returns:
            RealMultivarTaylor: Taylor expansion composed with absolute value from the left hand side.

        """

        const = self.const
        if const == 0.:
            raise ValueError("The absolute value is not differentiable at zero.")
        return self.copy() if const > 0. else -self

    def __floor__(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the floor function.

        Returns:
            RealMultivarTaylor: floor of Taylor expansion.

        """
        return self.create_const_expansion(math.floor(self.const))

    def __ceil__(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the ceil function.

        Returns:
            RealMultivarTaylor: ceil of Taylor expansion.

        """
        return self.create_const_expansion(math.ceil(self.const))

    def sign(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the sign function.

        Returns:
            RealMultivarTaylor: sign of Taylor expansion.

        """
        return self.create_const_expansion(np.sign(self.const))

    def cbrt(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the cubic root function.

        Returns:
            RealMultivarTaylor: cubic root of Taylor expansion.

        """
        return self ** (1. / 3.)

    @staticmethod
    def seq_deriv_atan_atanh(c: float, eps: float, order: int) -> np.ndarray:
        csq = c * c
        cd = 2. * c
        aux = -1. / (csq + eps)
        seq = np.zeros(order)
        seq[0] = 1. / (1. + eps * csq)
        if order > 1:
            seq[1] = seq[0] * cd * aux
            for i in range(2, order):
                seq[i] = (seq[i - 2] + cd * seq[i - 1]) * aux
        return seq

    def arctan(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the inverse tangent function. Here so that Numpy can use it on arrays.

        Returns:
            RealMultivarTaylor: inverse tangent of Taylor expansion.

        """
        if self.is_trivial():
            return self.create_const_expansion(math.atan(self.const))

        order = self.order
        const = self.const
        sequence = self.seq_deriv_atan_atanh(const, 1., order)
        seq = sequence / np.arange(1, order + 1)
        nilpo = self.get_nilpo_part()
        arctan = seq[-1] * nilpo
        for el in seq[-2::-1]:
            arctan.const = el
            arctan *= nilpo
        arctan.const = math.atan(const)
        return arctan

    def arctanh(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the inverse hyperbolic tangent function.
        Here so that Numpy can use it on arrays.

        Returns:
            RealMultivarTaylor: inverse hyperbolic tangent of Taylor expansion.

        """

        if self.is_trivial():
            return self.create_const_expansion(math.atanh(self.const))

        order = self.order
        const = self.const
        sequence = self.seq_deriv_atan_atanh(const, -1., order)
        seq = sequence / np.arange(1, order + 1)
        nilpo = self.get_nilpo_part()
        arctanh = seq[-1] * nilpo
        for el in seq[-2::-1]:
            arctanh.const = el
            arctanh *= nilpo
        arctanh.const = math.atanh(const)
        return arctanh

    @staticmethod
    def seq_deriv_asin_asinh(c: float, eps: float, order: int) -> np.ndarray:
        csq = c * c
        aux = -1. / (csq + eps)
        seq = np.zeros(order)
        seq[0] = 1. / math.sqrt(1. + eps * csq)
        if order > 1:
            seq[1] = seq[0] * c * aux
            inter = aux / np.arange(2, order)
            for i, el in enumerate(inter, 2):
                seq[i] = ((i - 1.) * seq[i - 2] + c * (2. * i - 1.) * seq[i - 1]) * el
        return seq

    def arcsin(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the inverse sine function.

        Returns:
            RealMultivarTaylor: inverse sine of Taylor expansion.

        """
        if self.is_trivial():
            return self.create_const_expansion(math.asin(self.const))

        order = self.order
        const = self.const
        sequence = self.seq_deriv_asin_asinh(const, -1., order)
        nilpo = self.get_nilpo_part()
        seq = sequence / np.arange(1, order + 1)
        arcsine = seq[-1] * nilpo
        for el in seq[-2::-1]:
            arcsine.const = el
            arcsine *= nilpo
        arcsine.const = math.asin(const)
        return arcsine

    def arccos(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the inverse cosine function.

        Returns:
            RealMultivarTaylor: inverse cosine of Taylor expansion.

        """
        return -self.arcsin() + math.pi / 2.

    def arcsinh(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the inverse hyperbolic sine function.

        Returns:
            RealMultivarTaylor: inverse hyperbolic sine of Taylor expansion.

        """
        if self.is_trivial():
            return self.create_const_expansion(math.asinh(self.const))

        order = self.order
        const = self.const
        sequence = self.seq_deriv_asin_asinh(const, 1., order)
        seq = sequence / np.arange(1, order + 1)
        nilpo = self.get_nilpo_part()
        arcsineh = seq[-1] * nilpo
        for el in seq[-2::-1]:
            arcsineh.const = el
            arcsineh *= nilpo
        arcsineh.const = math.asinh(const)
        return arcsineh

    def arccosh(self) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the inverse hyperbolic cosine function.

        Returns:
            RealMultivarTaylor: inverse hyperbolic cosine of Taylor expansion.

        """
        if self.is_trivial():
            return self.create_const_expansion(math.acosh(self.const))

        order = self.order
        const = self.const
        c2 = const * const
        aux = 1. / (1. - c2)
        seq = np.zeros(order)
        seq[0] = 1. / math.sqrt(c2 - 1.)
        if order > 1:
            seq[1] = const * seq[0] * aux
            for i in range(2, order):
                seq[i] = ((i - 1.) * seq[i - 2] + (2. * i - 1.) * const * seq[i - 1]) * aux / i
        seq /= np.arange(1, order + 1)
        nilpo = self.get_nilpo_part()
        arccosineh = seq[-1] * nilpo
        for el in seq[-2::-1]:
            arccosineh.const = el
            arccosineh *= nilpo
        arccosineh.const = math.acosh(const)
        return arccosineh

    def arctan2(self, q: Union["RealMultivarTaylor", float]) -> "RealMultivarTaylor":
        """Version for Taylor expansions of the arctan2.

        Args:
            Union["RealMultivarTaylor", float]: second argument of arctan2

        Returns:
            RealMultivarTaylor: arctan2(self, q).

        """
        if q != 0.:
            output = (self / q).arctan()
        elif self.const != 0.:
            output = -(q / self).arctan()
        else:
            raise ValueError
        output.const = math.atan2(self.const, float(q))
        return output
