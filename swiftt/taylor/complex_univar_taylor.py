# complex_univar_taylor.py: class implementing Taylor expansions of a unique complex variable
# Copyright 2022-2023 Romain Serra

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

from typing import List, Iterable, Optional
import numpy as np
from numba import njit
from swiftt.taylor.complex_multivar_taylor import ComplexMultivarTaylor, Scalar
from swiftt.taylor.taylor_expans_abstract import TaylorExpansAbstract, default_unknown_name, default_inverse_name
from swiftt.taylor.tables import factorials


class ComplexUnivarTaylor(ComplexMultivarTaylor):
    """Class for Taylor expansions of a single complex variable.

    """

    def __init__(self, order: int, var_names: List[str] = default_unknown_name) -> None:
        """Constructor for Taylor expansions of a single complex variable. Result has no assigned coefficients.

        Args:
            order (int): order of expansion.
            var_names (List[str]): name of variable.

        """
        ComplexMultivarTaylor.__init__(self, 1, order, var_names)

    def create_expansion_with_coeff(self, coeff: Iterable) -> "ComplexUnivarTaylor":
        """Method to create a new Taylor expansion (in same algebra than self) from given coefficients. Overwrites
        multivariate implementation to make sure the correct object is created (dedicated to univariate expansions).

        Args:
            coeff (Iterable[complex]): coefficients of new expansion.

        Returns:
            ComplexUnivarTaylor: Taylor expansion corresponding to inputted coefficients.

        """
        expansion = self.__class__(self._order, self._var_names)
        expansion.coeff = coeff
        return expansion

    def get_mth_term_coeff(self, order: int) -> Scalar:
        """Method to get the coefficient corresponding to a given order.

        Args:
            order (int): order of requested coefficient.

        Returns:
            Union[complex, float]: coefficient for input order.

        """
        if 0 <= order <= self._order:
            return self._coeff[order]

        raise IndexError("Input order must be a positive integer at most equal to the order of the algebra")

    def __mul__(self, other) -> "ComplexUnivarTaylor":
        """Method defining right-hand side multiplication. It works for the external multiplication i.e. with scalars
        and for the internal one, that is between expansions. Overwrites the multivariate implementation for
        performance.

        Args:
            other (Union[ComplexUnivarTaylor, complex, float]): quantity to be multiplied with.

        Returns:
            ComplexUnivarTaylor: multiplied objects.

        """

        if isinstance(other, ComplexUnivarTaylor):

            if self.is_in_same_algebra(other):
                multiplied_coeff = self.mul_univar(self._coeff, other._coeff)
                return self.create_expansion_with_coeff(multiplied_coeff)

            raise ValueError("Expansions to be multiplied are not from the same algebra")

        # scalar case
        return self.create_expansion_with_coeff(other * self._coeff)

    @staticmethod
    @njit(cache=True)
    def mul_univar(coeff: np.ndarray, other_coeff: np.ndarray) -> np.ndarray:
        """Static method transforming two series of coefficients into the coefficients of the product of univariate
        Taylor expansions. It emulates the polynomial product and the truncation at the same time.

        Args:
            coeff (numpy.ndarray): first set of coefficients.
            other_coeff (numpy.ndarray): second set of coefficients.

        Returns:
            numpy.ndarray: coefficient corresponding to product.

        """

        multiplied_coeff = coeff[0] * other_coeff
        multiplied_coeff[1:] += coeff[1:] * other_coeff[0]

        other_nonconst_coeff_reverted = np.flip(other_coeff[1:])
        for i in range(2, coeff.shape[0]):
            multiplied_coeff[i] += coeff[1:i].dot(other_nonconst_coeff_reverted[-i + 1:])

        return multiplied_coeff

    @staticmethod
    @njit(cache=True)
    def pow2_univar(coeff: np.ndarray, half_order: int) -> np.ndarray:
        """Static method transforming coefficients into the coefficients of the square of a univariate
        Taylor expansion. It emulates the polynomial product and the truncation at the same time.

        Args:
            coeff (numpy.ndarray): first set of coefficients.
            half_order (int): half order.

        Returns:
            numpy.ndarray: coefficients corresponding to square.

        """

        twice_coeff = 2. * coeff
        new_coeff = coeff[0] * twice_coeff
        squared_coeff = coeff[:half_order]**2
        new_coeff[0] = squared_coeff[0]

        for i in range(1, half_order):
            index = 2 * i
            new_coeff[index] += squared_coeff[i]
            new_coeff[i + 1:index] += twice_coeff[i] * coeff[1:i]

        dim = coeff.shape[0]
        for i in range(half_order, dim - 1):
            new_coeff[i + 1:] += twice_coeff[i] * coeff[1:dim-i]

        return new_coeff

    def pow2(self) -> "ComplexUnivarTaylor":
        """Method to raise Taylor expansion to the power 2 i.e. compute its multiplicative square.

        Returns:
            ComplexUnivarTaylor: Taylor expansion raised to power 2.

        """
        coeff = self.pow2_univar(self._coeff, self.half_order + 1)
        return self.create_expansion_with_coeff(coeff)

    def __truediv__(self, other) -> "ComplexUnivarTaylor":
        if isinstance(other, self.__class__):
            new_coeff = self._div_expansion(self._coeff, other._coeff)
            divided = self.create_expansion_with_coeff(new_coeff)
            return divided

        # scalar case
        return self * (1. / other)

    @staticmethod
    @njit(cache=True)
    def _div_expansion(coeff: np.ndarray, coeff_other: np.ndarray) -> np.ndarray:
        """Method computing the coefficients of an expansion divided by another from their respective coefficients.
        Uses (univariate) recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): first set of coefficients from the numerator.
            coeff_other (numpy.ndarray): second set of coefficients from the denominator.

        Returns:
            numpy.ndarray: coefficients corresponding to expansion obtained by dividing first expansion by second one.

        """
        divided_coeff = coeff / coeff_other[0]
        flipped = np.flip(coeff_other) / coeff_other[0]
        for i in range(1, coeff.shape[0]):
            divided_coeff[i] -= divided_coeff[:i].dot(flipped[-i-1:-1])
        return divided_coeff

    def reciprocal(self) -> "ComplexUnivarTaylor":
        new_coeff = self._reciprocal_univar(self._coeff)
        return self.create_expansion_with_coeff(new_coeff)

    @staticmethod
    @njit(cache=True)
    def _reciprocal_univar(coeff: np.ndarray) -> np.ndarray:
        """Method computing the coefficients of the reciprocal (multiplicative inverse) of an expansion.
        Uses (univariate) recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): set of coefficients from the original expansion.

        Returns:
            numpy.ndarray: coefficients corresponding to reciprocal.

        """
        reciprocal_coeff = np.empty(len(coeff))
        reciprocal_coeff[0] = 1. / coeff[0]
        inter = -np.flip(coeff) / coeff[0]
        for i in range(1, coeff.shape[0]):
            reciprocal_coeff[i] = reciprocal_coeff[:i].dot(inter[-i-1:-1])
        return reciprocal_coeff

    def exp(self) -> "ComplexUnivarTaylor":
        first_term = self._exp_cst(self._coeff[0])  # this prevents from having the method static
        preprocessed_coeff = first_term * self._coeff
        preprocessed_coeff[0] = first_term
        return self.create_expansion_with_coeff(self._exp_expansion(self._coeff, preprocessed_coeff))

    @staticmethod
    @njit(cache=True)
    def _exp_expansion(coeff: np.ndarray, preprocessed_coeff: np.ndarray) -> np.ndarray:
        """Method computing the coefficients of the exponential of an expansion from their coefficients.
        Uses (univariate) recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): expansion's coefficients.
            preprocessed_coeff (numpy.ndarray): coefficients with some pre-computations done.

        Returns:
            numpy.ndarray: coefficients corresponding to the exponential of the expansion.

        """
        integers = np.arange(1., coeff.shape[0] + 1., 1.)
        for i in range(2, coeff.shape[0]):
            preprocessed_coeff[i] += (coeff[1:i] * integers[:i-1]).dot(preprocessed_coeff[i-1:0:-1]) / float(i)
        return preprocessed_coeff

    def log(self) -> "ComplexUnivarTaylor":
        first_term = self._log_cst(self._coeff[0])  # this prevents from having the method static
        preprocessed_coeff = self._coeff / self._coeff[0]
        preprocessed_coeff[0] = first_term
        return self.create_expansion_with_coeff(self._log_expansion(self._coeff, preprocessed_coeff))

    @staticmethod
    @njit(cache=True)
    def _log_expansion(coeff: np.ndarray, preprocessed_coeff: np.ndarray) -> np.ndarray:
        """Method computing the coefficients of the natural logarithm of an expansion from their coefficients.
        Uses (univariate) recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): expansion's coefficients.
            preprocessed_coeff (numpy.ndarray): coefficients with some pre-computations done.

        Returns:
            numpy.ndarray: coefficients corresponding to the logarithm of the expansion.

        """
        integers = np.arange(1., coeff.shape[0] + 1., 1.)
        scaled_coeff = coeff / coeff[0]
        for i in range(2, coeff.shape[0]):
            preprocessed_coeff[i] -= (preprocessed_coeff[1:i] * integers[:i-1]).dot(scaled_coeff[i-1:0:-1]) / float(i)
        return preprocessed_coeff

    def sqrt(self) -> "ComplexUnivarTaylor":
        first_term = self._sqrt_cst(self._coeff[0])  # this prevents from having the method static
        preprocessed_coeff = self._coeff / (2. * first_term)
        preprocessed_coeff[0] = first_term
        return self.create_expansion_with_coeff(self._sqrt_expansion(self._coeff, preprocessed_coeff))

    @staticmethod
    @njit(cache=True)
    def _sqrt_expansion(coeff: np.ndarray, preprocessed_coeff: np.ndarray) -> np.ndarray:
        """Method computing the coefficients of the square root of an expansion from their coefficients.
        Uses (univariate) recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): expansion's coefficients.
            preprocessed_coeff (numpy.ndarray): coefficients with some pre-computations done.

        Returns:
            numpy.ndarray: coefficients corresponding to the square root of the expansion.

        """
        inter = np.arange(1., coeff.shape[0] + 1., 1.) / preprocessed_coeff[0]
        for i in range(2, coeff.shape[0]):
            preprocessed_coeff[i] -= (preprocessed_coeff[1:i] * inter[:i-1]).dot(preprocessed_coeff[i-1:0:-1]) / \
                                     float(i)
        return preprocessed_coeff

    def __call__(self, *args, **kwargs):
        """Method for calling the Taylor expansion. Wraps several possibilities: evaluation and composition with another
        expansion.

        Returns:
            Union[ComplexMultivarTaylor, complex, float]: Taylor expansion called on input.

        """

        other = args[0]
        return self.compose(other) if isinstance(other, ComplexMultivarTaylor) else self.pointwise_eval(other)

    def compose(self, other: ComplexMultivarTaylor) -> ComplexMultivarTaylor:
        """Method performing composition with inputted univariate Taylor expansion (must have the same order).

        Args:
            other (ComplexMultivarTaylor): Taylor expansion to be composed on the right-hand side.

        Returns:
            ComplexMultivarTaylor: composed expansion.

        """

        if other.is_const():
            return other.create_const_expansion(self.pointwise_eval(other.const))

        if other.order != self._order:
            raise ValueError("Expansions to be composed must have same order")

        if not other.is_nilpotent():
            raise ValueError("Right-hand-side expansion for composition must be nilpotent")

        # Horner's scheme
        composed = self._coeff[-1] * other
        composed.const = self._coeff[-2]
        for el in self._coeff[-3::-1]:
            composed *= other
            composed.const = el
        return composed

    def pointwise_eval(self, x: Scalar) -> Scalar:
        """Method for the evaluation of the Taylor expansion on a given point.

        Args:
            x (Union[complex, float]): point of evaluation.

        Returns:
            Union[complex, float]: Taylor expansion evaluated at given point.

        """
        if self.is_trivial():
            return self.const

        # Horner's scheme
        output = self._coeff[-1] * x + self._coeff[-2]
        for el in self._coeff[-3::-1]:
            output = output * x + el
        return output

    def massive_eval(self, x: np.ndarray) -> np.ndarray:
        """Method for the evaluation of the Taylor expansion on a range of points (vectorized evaluation).

        Args:
            x (numpy.ndarray): points of evaluation.

        Returns:
            numpy.ndarray: Taylor expansion evaluated at given points.

        """
        # in the uni-variate case, the point-wise evaluation method is already vectorized
        return np.array(self.pointwise_eval(x))

    def truncated(self, new_order: int) -> "ComplexUnivarTaylor":
        """Method for the truncation at a given order. Output lives in another algebra, of lower dimension.
        Overwrites multivariate implementation for performance.

        Args:
            new_order (int): order of algebra in which to truncate the Taylor expansion.

        Returns:
            ComplexUnivarTaylor: Taylor expansion truncated at input order.

        """
        if 0 <= new_order < self._order:
            truncated = self.__class__(new_order, self._var_names)
            truncated.coeff = self._coeff[:truncated.dim_alg]
            return truncated

        raise ValueError("Input order must be an integer between zero and order of initial algebra")

    def prolong(self, new_order: int) -> "ComplexUnivarTaylor":
        """Method for the prolongation (opposite of truncation) at a given order. Output lives in another algebra, of
        higher dimension. It is not a rigorous operation as the possible contributions within the former remainder are
        ignored. Overwrites multivariate implementation for performance.

        Args:
            new_order (int): order of algebra in which to prolong the Taylor expansion.

        Returns:
            ComplexUnivarTaylor: Taylor expansion prolonged at input order.

        """

        if new_order > self._order:
            prolonged = self.__class__(new_order, self._var_names)
            coeff = np.zeros(prolonged._dim_alg, dtype=self.var_type)
            coeff[:self._dim_alg] = np.array(self._coeff, dtype=self.var_type)
            prolonged.coeff = coeff
            return prolonged

        raise ValueError("New order must be larger than old one")

    def prolong_one_order(self) -> "ComplexUnivarTaylor":
        """Method for the prolongation (opposite of truncation) of one order. Output lives in another algebra, of
        higher dimension. It is not a rigorous operation as the possible contributions within the former remainder are
        ignored. Overwrites multivariate implementation for performance.

        Returns:
            ComplexUnivarTaylor: Taylor expansion prolonged by one order.

        """
        prolonged = self.__class__(self._order + 1, self._var_names)
        coeff = np.zeros(prolonged._dim_alg, dtype=self.var_type)
        coeff[:self._dim_alg] = np.array(self._coeff, dtype=self.var_type)
        prolonged.coeff = coeff
        return prolonged

    def rigorous_integ_once_wrt_var(self, index_var: int = 0) -> "ComplexUnivarTaylor":
        """Method performing integration with respect to a given unknown variable. The integration constant is zero.
        This transformation is rigorous as the order of the expansion is increased by one. In other words, the output
        lives in another algebra, of higher dimension. Overwrites multivariate implementation for performance.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            ComplexUnivarTaylor: integrated Taylor expansion w.r.t. input variable number.

        """
        integrated = self.__class__(self._order + 1, self._var_names)
        coeff = np.zeros(integrated._dim_alg, dtype=self._var_type)
        coeff[1:] = self._coeff / np.arange(1, integrated.dim_alg)
        integrated.coeff = coeff
        return integrated

    def integ_once_wrt_var(self, index_var: int = 0) -> "ComplexUnivarTaylor":
        """Method performing integration while remaining in the same algebra. This transformation is not rigorous in the
        sense that the order of the expansion should be increased by one. Overwrites multivariate implementation for
        performance.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed. It is not used
                as the expansion is univariate and there is no choice.

        Returns:
            ComplexUnivarTaylor: integrated Taylor expansion.

        """
        new_coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        new_coeff[1:] = self._coeff[:-1] / np.arange(1, self.dim_alg)
        return self.create_expansion_with_coeff(new_coeff)

    def deriv_once_wrt_var(self, index_var: int = 0) -> "ComplexUnivarTaylor":
        """Method performing differentiation while remaining in the same algebra. This transformation is not rigorous in
         the sense that the order of the expansion should be decreased by one. Overwrites multivariate implementation
         for performance.

        Args:
            index_var (int): variable number w.r.t. which differentiation needs to be performed. It is not used
                as the expansion is univariate and there is no choice.

        Returns:
            ComplexUnivarTaylor: differentiated Taylor expansion.

        """
        new_coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        new_coeff[:-1] = self._coeff[1:] * np.arange(1, self.dim_alg)
        return self.create_expansion_with_coeff(new_coeff)

    def get_all_partial_deriv_up_to(self, order: int) -> np.ndarray:
        """Method returning the derivatives of the expansion up to a given order. In other words, it converts
        the polynomial coefficients by multiplying them with factorials. Overwrites the multivariate implementation
        for performance.

        Args:
            order (int): order (included) up to which derivatives need to be outputted.

        Returns:
            numpy.ndarray: derivatives up to input order.

        """

        if 0 <= order <= self._order:
            output = np.array(self._coeff[:order + 1], dtype=self.var_type)
            return output * factorials(order)

        raise ValueError("Order of differentiation must be non-negative and not exceed order of algebra")

    def get_mth_deriv(self, m: int) -> Scalar:
        """Method returning a specific derivative of the expansion i.e. at a specified order.

        Args:
            m (int): order of requested derivative.

        Returns:
            Union[complex, float]: derivative at specified order.

        """
        return self.get_all_partial_deriv_up_to(m)[-1]

    def divided_by_var(self, index_var: int = 0) -> "ComplexUnivarTaylor":
        """Method returning the Taylor expansion divided by the variable. This is not a rigorous operation as the order
        should be decreased by one.

        Args:
            index_var (str): index of variable, not used in the univariate case.

        Returns:
            ComplexUnivarTaylor: Taylor expansion divided by variable.

        """

        if self.const != 0.:
            raise ValueError("Cannot divide by variable if constant term is not zero.")

        coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        coeff[:self.order] = np.array(self._coeff[1:])
        return self.create_expansion_with_coeff(coeff)

    def compo_inverse(self, name_inverse: Optional[str] = None) -> "ComplexUnivarTaylor":
        """Method returning the inverse of the univariate Taylor expansion from the point of view of composition, i.e.
        so that composed with self it gives the monomial X. This is also known as series reversion.
        The computation is done iteratively, starting from the inversion of the first-order truncated expansion.
        See the work of M. Berz.

        Args:
            name_inverse (str): name of inverse variable.

        Returns:
            ComplexUnivarTaylor: composition-inverse of Taylor expansion.

        """

        if not self.is_nilpotent():
            raise ValueError("Expansion to be inverted cannot have non-zero constant term")

        if name_inverse is None:
            name_inverse = default_inverse_name

        coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        coeff[1] = 1. / self._coeff[1]
        invlin = self.create_expansion_with_coeff(coeff)

        if self._order == 1:
            invlin.var_names = [name_inverse]
            return invlin

        # order is at least two
        inter = -self.get_nonlin_part()
        coeff[1] = 1.
        identity = self.create_expansion_with_coeff(coeff)
        inverted_expansion = invlin.compose(identity + inter.compose(invlin))
        for __ in range(3, self._dim_alg):
            inverted_expansion = invlin.compose(identity + inter.compose(inverted_expansion))
        inverted_expansion.var_names = [name_inverse]
        return inverted_expansion

    @property
    def effective_order(self) -> int:
        """Method returning the effective order of the expansion, that is the highest order with non-zero coefficients.
        Overwrites the multivariate implementation for simplicity.

        Returns:
            int: effective order of Taylor expansion.

        """
        for i, el in zip(range(self._dim_alg - 1, -1, -1), self._coeff[::-1]):
            if el != 0.:
                return i
        return 0

    @property
    def total_depth(self) -> int:
        """Method returning the total depth of the Taylor expansion, that is the lowest order with non-zero coefficient.
        Overwrites the multivariate implementation for simplicity.

        Returns:
            int: total depth of Taylor expansion.

        """
        for i, el in enumerate(self._coeff):
            if el != 0.:
                return i
        return self._dim_alg

    @property
    def remainder_term(self) -> str:
        return TaylorExpansAbstract.landau_univar(self._order, self._var_names)

    def __str__(self) -> str:
        """Method to cast Taylor expansion as a string. Overwrites parent implementation that assumes more than one
        variable.

        Returns:
            str: string representing the Taylor expansion.

        """
        string = str(self.const)
        if not self.is_trivial():
            var_name = self._var_names[0]
            if self._coeff[1] != 0.:
                string += " + " + str(self._coeff[1]) + " * " + var_name
            for j, el in enumerate(self._coeff[2:], 2):
                if el != 0.:
                    string += " + " + str(el) + " * " + var_name + "**" + str(j)
        return string + " + " + self.remainder_term

    def var_inserted(self, index_new_var: int, unknown_name: Optional[str] = None) -> ComplexMultivarTaylor:
        """Method for the addition of a new variable. Output lives in another algebra, of higher dimension. All its
        terms associated with the new variable are zero and the other ones are identical to original expansion.
        Overwrites the parent implementation for performance.

        Args:
            index_new_var (int): index of new variable to be added.
            unknown_name (str): name of new variable.

        Returns:
            ComplexMultivarTaylor: Taylor expansion with an additional variable.

        """

        if unknown_name is None:
            unknown_name = default_unknown_name + str(2)
        if unknown_name in self.var_names:
            raise ValueError("Name of new variable already exists.")

        if index_new_var == 1:
            new_index_old_var = 0
            new_var_names = [self._var_names[0], unknown_name]
        else:
            new_index_old_var = 1
            new_var_names = [unknown_name, self._var_names[0]]

        new_expansion = ComplexMultivarTaylor(2, self._order, new_var_names)

        new_coeff = np.zeros(new_expansion.dim_alg, dtype=self._var_type)
        for exponent, index_coeff in new_expansion.get_mapping_monom().items():
            if exponent[index_new_var] == 0:
                new_coeff[index_coeff] = self._coeff[exponent[new_index_old_var]]

        new_expansion.coeff = new_coeff
        return new_expansion

    def var_eval(self, index_var: int, value: Scalar) -> "ComplexMultivarTaylor":
        if index_var != 0:
            raise IndexError("The inputted index does not correspond to any variable of the expansion.")
        return self.create_const_expansion(self.pointwise_eval(value))
