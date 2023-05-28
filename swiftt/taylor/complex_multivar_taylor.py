# complex_multivar_taylor.py: class implementing Taylor expansions of multiple complex variables
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

from typing import Union, List, Dict, Tuple, Iterable, Optional
import cmath
import numpy as np
from numba import njit
from swiftt.taylor.taylor_expans_abstract import TaylorExpansAbstract, landau_symbol, default_unknown_name
from swiftt.taylor.tables import algebra_dim, mapping_monom, flat_mul_table, mul_indices, square_indices, deriv_table
from swiftt.map_abstract import MapAbstract
from swiftt.taylor.tables import factorials

Scalar = Union[complex, float]


class ComplexMultivarTaylor(TaylorExpansAbstract):
    """Class for complex Taylor expansions i.e. with a real and an imaginary part for their coefficients and variables.

    """

    # intrinsic functions for complex
    _sqrt_cst = cmath.sqrt
    _log_cst, _exp_cst = cmath.log, cmath.exp
    _cos_cst, _sin_cst = cmath.cos, cmath.sin
    _cosh_cst, _sinh_cst = cmath.cosh, cmath.sinh
    _tan_cst, _tanh_cst = cmath.tan, cmath.tanh

    def __init__(self, n_var: int, order: int, var_names: List[str]) -> None:
        """Constructor for Taylor expansions of multiple complex variables. Result has no assigned coefficients.

        Args:
             n_var (int): number of variables in algebra.
             order (int): order of algebra.
             var_names (List[str]): name of variables in algebra.

        """

        TaylorExpansAbstract.__init__(self, n_var, order, np.complex128, var_names)

    @property
    def half_order(self) -> int:
        """Getter for the half-order of the algebra.

        Returns:
             int: half-order.

        """
        return self._order // 2

    @TaylorExpansAbstract.order.setter
    def order(self, new_order: int) -> None:
        if new_order < 0:
            raise ValueError("Order must be positive.")
        if new_order < self._order:
            truncated = self.truncated(new_order)
            self._order = new_order
            self._dim_alg = algebra_dim(self._n_var, self._order)
            self._coeff = truncated._coeff
        elif new_order > self._order:
            prolonged = self.prolong(new_order)
            self._order = new_order
            self._dim_alg = algebra_dim(self._n_var, self._order)
            self._coeff = prolonged._coeff

    @TaylorExpansAbstract.n_var.setter
    def n_var(self, n_var: int) -> None:
        if n_var == self._n_var:
            pass
        elif n_var <= 0:
            raise ValueError("Number of variables cannot be negative.")
        elif n_var == 1 or self._n_var == 1:
            raise ValueError("The setter for the number of variables should not be used to create a univariate "
                             "expansion. Create a instance of the appropriate class instead.")
        elif n_var < self._n_var:
            inter = self.copy()
            for __ in range(0, self._n_var - n_var):
                inter = inter.last_var_removed()
            self._n_var = n_var
            self._dim_alg = algebra_dim(self._n_var, self._order)
            self._coeff = inter.coeff
            self._var_names = inter.var_names
        else:
            # variables need to be added
            inter = self.copy()
            for __ in range(0, n_var - self._n_var):
                inter = inter.var_appended()
            self._n_var = n_var
            self._dim_alg = algebra_dim(self._n_var, self._order)
            self._coeff = inter.coeff
            self._var_names = inter.var_names

    def get_mapping_monom(self) -> Dict[Tuple[int, ...], int]:
        """Method returning the mapping between coefficients' numbers and monomials.

        Returns:
             Dict[Tuple[int, ...], int]: algebra's mapping.

        """
        return mapping_monom(self._n_var, self._order)

    def get_flat_table_mul(self) -> np.ndarray:
        """Method returning the flattened algebra's multiplication table.

        Returns:
             numpy.ndarray: flattened multiplication table.

        """
        return flat_mul_table(self._n_var, self._order)

    def get_indices_mul(self) -> np.ndarray:
        """Method returning the algebra's multiplication indices.

        Returns:
             numpy.ndarray: multiplication indices related to the multiplication table.

        """
        return mul_indices(self._n_var, self._order)

    def get_table_deriv(self) -> np.ndarray:
        """Method returning the algebra's differentiation table.

        Returns:
             numpy.ndarray: differentiation table.

        """
        return deriv_table(self._n_var, self._order)

    def get_square_indices(self) -> np.ndarray:
        """Method returning the coefficients' indices corresponding to monomials that are the square of an other
        monomial within the algebra.

        Returns:
             List[int]: so-called square indices.

        """
        return square_indices(self._n_var, self._order)

    def copy(self) -> "ComplexMultivarTaylor":
        """Method returning a copy of the Taylor expansion.

        Returns:
            ComplexMultivarTaylor: copy of the expansion.

        """
        return self.create_expansion_with_coeff(self._coeff)

    def __complex__(self) -> complex:
        """Method to cast Taylor expansion as a scalar (according to its constant part).

        Returns:
            complex: constant part of expansion seen as a scalar.

        """
        return complex(self.const)

    def get_var(self, var_index: int) -> "ComplexMultivarTaylor":
        """Method returning an expansion corresponding to the linear monomial for inputted variable index.

        Args:
            var_index (int): index of variable.

        Returns:
            ComplexMultivarTaylor: Taylor expansion whose coefficients are all zero except the one associated
               to linear term in input variable, which is one.

        """
        if var_index not in range(0, self.n_var):
            raise IndexError("Input variable index does not match any of expansion's")
        coeff = np.zeros(self.dim_alg)
        coeff[var_index + 1] = 1.
        return self.create_expansion_with_coeff(coeff)

    def create_expansion_with_coeff(self, coeff) -> "ComplexMultivarTaylor":
        """Method to create a new Taylor expansion (in same algebra than self) from given coefficients.

        Args:
            coeff (Iterable[complex]): coefficients of new expansion.

        Returns:
            ComplexMultivarTaylor: Taylor expansion corresponding to inputted coefficients.

        """
        expansion = self.__class__(self._n_var, self._order, self._var_names)
        expansion.coeff = coeff
        return expansion

    def create_const_expansion(self, const: Scalar) -> "ComplexMultivarTaylor":
        """Method to create a new Taylor expansion (in same algebra than self) from given constant coefficient.

        Args:
            const (Union[complex, float]): constant coefficient of new expansion.

        Returns:
            ComplexMultivarTaylor: Taylor expansion corresponding to inputted constant coefficient.

        """
        const_expans = self.create_null_expansion()
        const_expans.const = const
        return const_expans

    def create_null_expansion(self) -> "ComplexMultivarTaylor":
        """Method to create a new Taylor expansion (in same algebra than self) with null polynomial part.

        Returns:
            ComplexMultivarTaylor: Taylor expansion with same order and number of variables but null
               polynomial part.

        """
        return self.create_expansion_with_coeff(np.zeros(self._dim_alg, dtype=self._var_type))

    def create_monomial_expansion_from_exponent(self, indices: Iterable[int],
                                                factor: Scalar = 1.) -> "ComplexMultivarTaylor":
        """Method to create a new Taylor expansion (in same algebra than self) proportional to a given monomial.

         Args:
             indices (Iterable[int]): monomial's order for each variable.
             factor (Optional[Union[complex, float]]): scalar multiplying the monomial.

         Returns:
             ComplexMultivarTaylor: Taylor expansion proportional to inputted monomial.

        """
        coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        try:
            coeff[self.get_mapping_monom()[tuple(indices)]] = factor
        except KeyError:
            raise ValueError("The monomial is not in the same algebra than this expansion.")
        return self.create_expansion_with_coeff(coeff)

    def coeff_from_index(self, i: int) -> Scalar:
        return self._coeff[i]

    @property
    def coeff(self) -> np.ndarray:
        """Getter for coefficients of polynomial part.

        Returns:
            Iterable[Union[complex, float]: coefficients of Taylor expansion.

        """

        return np.array(self._coeff)

    @coeff.setter
    def coeff(self, coefficients: np.ndarray) -> None:
        """Setter for coefficients of polynomial part.

        Args:
            coefficients (Iterable[Union[complex, float]): new coefficients of Taylor expansion.

        """

        if len(coefficients) == self._dim_alg:
            self._coeff = np.array(coefficients, dtype=self._var_type)
        else:
            raise ValueError("Input coefficients have wrong number of elements (" + str(coefficients.shape) +
                             " instead of " + str(self._dim_alg) + ")")

    def is_const(self) -> bool:
        """Method returning True if the expansion's polynomial part is constant, False otherwise.

        Returns:
            bool: True if and only if all non-constant coefficients are zero.

        """
        non_zero = np.array(np.nonzero(self._coeff)[0], dtype=int)
        nb_non_zero_coeff = len(non_zero)
        if nb_non_zero_coeff == 0:
            # the polynomial part is null
            return True
        if nb_non_zero_coeff != 1:
            # expansion has at least two non-zero coefficients
            return False
        # expansion has one non-zero coefficient, is it the one from the constant part or not
        return self._coeff[0] != 0.

    def get_linear_part(self) -> "ComplexMultivarTaylor":
        """Method returning the linear part of the Taylor expansion.

        Returns:
            ComplexMultivarTaylor: linear part of Taylor expansion.

        """
        try:
            new_coeff = np.zeros(self._dim_alg, dtype=self._var_type)
            new_coeff[1:self._n_var + 1] = np.array(self._coeff[1:self._n_var + 1], dtype=self._var_type)
            return self.create_expansion_with_coeff(new_coeff)
        except IndexError:
            raise ValueError("There is no linear part in a zeroth-order expansion.")

    def get_nilpo_part(self) -> "ComplexMultivarTaylor":
        """Method returning the nilpotent part of the Taylor expansion.

        Returns:
            ComplexMultivarTaylor: nilpotent part of Taylor expansion.

        """
        nilpo = self.copy()
        nilpo.const = 0.
        return nilpo

    def get_high_order_part(self, order: int) -> "ComplexMultivarTaylor":
        """Method returning the high order part of the expansion in the same algebra. It is not a rigorous operation.

        Args:
            order (int): order (included) below which all contributions are removed.

        Returns:
            ComplexMultivarTaylor: high order part of the Taylor expansion.

        """
        if 1 <= order <= self._order:
            coeff = np.array(self._coeff, dtype=self._var_type)
            coeff[:algebra_dim(self._n_var, order - 1)] = 0.
            return self.create_expansion_with_coeff(coeff)

        raise ValueError("The inputted order exceeds the current order.")

    def get_low_order_part(self, order: int) -> "ComplexMultivarTaylor":
        """Method returning the low order part of the expansion in the same algebra. It is not a rigorous operation.

        Args:
            order (int): order (included) above which all contributions are removed.

        Returns:
            ComplexMultivarTaylor: low order part of the Taylor expansion.

        """
        if order == 0:
            return self.get_const_part()
        if order < self._order:
            coeff = np.array(self._coeff, dtype=self._var_type)
            coeff[algebra_dim(self._n_var, order):] = 0.
            return self.create_expansion_with_coeff(coeff)
        raise ValueError("The inputted order exceeds the current order.")

    def get_low_order_wrt_var(self, index_var: int, order: int) -> "ComplexMultivarTaylor":
        """Method returning an expansion where all the terms above a given order for a given variable have been removed.

        Args:
            index_var (int): index of variable.
            order (int): order (included) above which all contributions are removed.

        Returns:
            ComplexMultivarTaylor: Taylor expansion truncated at inputted order for inputted variable.

        """
        new_coeff = np.array(self._coeff)
        mapping = self.get_mapping_monom()
        indices_coeff = [mapping[exponent] for exponent in mapping if exponent[index_var] > order]
        new_coeff[indices_coeff] = 0.
        return self.create_expansion_with_coeff(new_coeff)

    def is_nilpotent(self) -> bool:
        """Method returning True if the expansion's polynomial part is nilpotent, False otherwise.

        Returns:
            bool: True if and only if the constant coefficient is zero.

        """
        return self.const == 0.

    @property
    def gradient(self) -> np.ndarray:
        """Method returning the evaluation of the first-order derivatives w.r.t. all variables of expansion.

        Returns:
            numpy.ndarray: first-order derivatives.

        """
        try:
            # the first-order derivatives are exactly the coefficients of the linear part
            return np.array(self._coeff[1:self._n_var + 1], dtype=self._var_type)
        except IndexError:
            raise ValueError("No gradient can be derived from a zeroth-order expansion.")

    @property
    def hessian(self) -> np.ndarray:
        """Method returning the evaluation of the second-order derivatives w.r.t. all variables of expansion.

        Returns:
            numpy.ndarray: second-order derivatives expressed in Hessian form.

        """

        derivs = self.get_all_partial_deriv_up_to(2)
        mapping = self.get_mapping_monom()

        hessian = np.empty((self._n_var, self._n_var), dtype=self._var_type)
        exponent = [0] * self._n_var
        for i in range(0, self._n_var):
            exponent[i] = 2
            hessian[i, i] = derivs[mapping[tuple(exponent)]]
            exponent[i] = 1
            for j in range(0, i):
                exponent[j] = 1
                hessian[i, j] = hessian[j, i] = derivs[mapping[tuple(exponent)]]
                exponent[j] = 0
            exponent[i] = 0

        return hessian

    def get_partial_deriv(self, exponents: Tuple[int, ...]) -> Scalar:
        output = self.coeff_from_index(self.get_mapping_monom()[exponents])
        return output * np.prod(factorials(self.order)[list(exponents)])

    def get_all_partial_deriv_up_to(self, order: int) -> np.ndarray:
        """Method returning the partial derivatives of the expansion up to a given order. In other words, it converts
        the polynomial coefficients by multiplying them with factorials.

        Args:
            order (int): order (included) up to which partial derivatives need to be outputted.

        Returns:
            numpy.ndarray: partial derivatives w.r.t. expansion's variables.

        """

        if order > self._order or order < 0:
            raise ValueError("Order of differentiation must be non-negative and not exceed order of algebra")

        output = np.array(self._coeff[:algebra_dim(self._n_var, order)], dtype=self.var_type)
        fac = factorials(order)
        for exponent, index_coeff in self.get_mapping_monom().items():
            output[index_coeff] *= np.prod(fac[list(exponent)])
            if index_coeff + 1 == len(output):
                # the next index is outside the scope of interest
                break
        return output

    def __mod__(self, other: float) -> "ComplexMultivarTaylor":
        """Modulo function when the left-hand side is a Taylor expansion.

        Args:
            other (float): modulo argument.

        Returns:
            ComplexMultivarTaylor: Taylor expansion's after modulo operation.

        """
        modulo = self.copy()
        modulo.const = modulo.const % other
        return modulo

    def __add__(self, other) -> "ComplexMultivarTaylor":
        """Method defining right-hand side addition. Works both with scalars and other expansions.

        Args:
            other (Union[ComplexMultivarTaylor, complex, float]): quantity to be added.

        Returns:
            ComplexMultivarTaylor: Taylor expansion summed with argument.

        """

        if isinstance(other, ComplexMultivarTaylor):

            if self.is_in_same_algebra(other):
                return self.create_expansion_with_coeff(self._coeff + other._coeff)

            raise ValueError("Expansions to be summed are not from the same algebra")

        # scalar case
        added = self.copy()
        added._coeff[0] += other
        return added

    def __sub__(self, other) -> "ComplexMultivarTaylor":
        """Method defining right-hand side subtraction. Works both with scalars and other expansions.

        Args:
            other (Union[ComplexMultivarTaylor, complex, float]): quantity to be subtracted.

        Returns:
            ComplexMultivarTaylor: Taylor expansion subtracted with argument.

        """

        if isinstance(other, ComplexMultivarTaylor):

            if self.is_in_same_algebra(other):
                return self.create_expansion_with_coeff(self._coeff - other._coeff)

            raise ValueError("Expansions to be subtracted are not from the same algebra")

        # scalar case
        subtracted = self.copy()
        subtracted._coeff[0] -= other
        return subtracted

    def __rsub__(self, other) -> "ComplexMultivarTaylor":
        """Method defining left-hand side subtraction. Works both with scalars and other expansions.

        Args:
            other (Union[ComplexMultivarTaylor, complex, float]): quantity to perform subtraction on.

        Returns:
            ComplexMultivarTaylor: argument subtracted with Taylor expansion.

        """

        if isinstance(other, ComplexMultivarTaylor):

            if self.is_in_same_algebra(other):
                return self.create_expansion_with_coeff(other._coeff - self._coeff)

            raise ValueError("Expansions to be subtracted are not from the same algebra")

        # scalar case
        subtracted = -self
        subtracted._coeff[0] += other
        return subtracted

    def __neg__(self) -> "ComplexMultivarTaylor":
        """Method defining negation (additive inverse). Overwrites parent implementation for performance.

        Returns:
            ComplexMultivarTaylor: opposite of object (from the point of view of addition).

        """
        return self.create_expansion_with_coeff(-self._coeff)

    def linearly_combine_with_another(self, alpha: Scalar, expansion: "ComplexMultivarTaylor",
                                      beta: Scalar) -> "ComplexMultivarTaylor":
        """Method multiplying with a scalar and then adding with a another expansion also multiplied by some scalar.
        Overwritten for speed purposes.

        Args:
            alpha (Union[complex, float]): multiplier for self.
            expansion (ComplexMultivarTaylor): expansion to be linearly combined with.
            beta (Union[complex, float]): multiplier for other expansion.

        Returns:
            ComplexMultivarTaylor: linear combination of self and arguments.

        """
        try:
            return self.create_expansion_with_coeff(alpha * self._coeff + beta * expansion._coeff)
        except ValueError as exc:
            raise exc

    def linearly_combine_with_many(self, alpha: Scalar, expansions: Iterable["ComplexMultivarTaylor"],
                                   betas: Iterable[Scalar]) -> "ComplexMultivarTaylor":
        """Method multiplying with a scalar and then adding an arbitrary number of expansions also multiplied by some
        scalar.

        Args:
            alpha (Union[complex, float]): multiplier for self.
            expansions (Iterable[ComplexMultivarTaylor]): expansions to be linearly combined with.
            betas (Iterable[Union[complex, float]]): multipliers for other expansions.

        Returns:
            ComplexMultivarTaylor: linear combination of self and arguments.

        """
        new_coeff = alpha * self._coeff
        for beta, expansion in zip(betas, expansions):
            new_coeff += beta * expansion._coeff
        return self.create_expansion_with_coeff(new_coeff)

    @staticmethod
    @njit(cache=True)
    def mul_multivar(coeff: np.ndarray, other_coeff: np.ndarray, square_ind: np.ndarray,
                     table_mul: np.ndarray, indices_mul: np.ndarray) -> np.ndarray:
        """Static method transforming two series of coefficients into the coefficients of the product of multivariate
        Taylor expansions. It emulates the polynomial product and the truncation at the same time.

        Args:
            coeff (numpy.ndarray): first set of coefficients.
            other_coeff (numpy.ndarray): second set of coefficients.
            square_ind (numpy.ndarray): precomputed indices corresponding to monomials which are the square
                of another monomial in the algebra.
            table_mul (numpy.ndarray): flattened algebra's multiplication table.
            indices_mul (numpy.ndarray): algebra's multiplication indices.

        Returns:
            numpy.ndarray: coefficient corresponding to product.

        """

        multiplied_coeff = coeff[0] * other_coeff + other_coeff[0] * coeff
        dim_half_order = len(square_ind)
        symmetric_terms = coeff[:dim_half_order] * other_coeff[:dim_half_order]
        multiplied_coeff[0] = 0.
        multiplied_coeff[square_ind] += symmetric_terms
        slices = indices_mul[2:] - indices_mul[1:-1]

        for i, (slice_index, el1, el2) in enumerate(zip(slices, coeff[2:], other_coeff[2:]), 2):
            multiplied_coeff[table_mul[indices_mul[i - 1] + 1:indices_mul[i]]] += el1 * other_coeff[1:slice_index] \
                                                                                  + el2 * coeff[1:slice_index]

        return multiplied_coeff

    def __mul__(self, other) -> "ComplexMultivarTaylor":
        """Method defining right-hand side multiplication. It works for the external multiplication i.e. with scalars
        and for the internal one, that is between expansions.

        Args:
            other (Union[ComplexMultivarTaylor, complex, float]): quantity to be multiplied with.

        Returns:
            ComplexMultivarTaylor: multiplied objects.

        """

        if isinstance(other, ComplexMultivarTaylor):

            try:
                multiplied_coeff = self.mul_multivar(self._coeff, other._coeff, self.get_square_indices(),
                                                     self.get_flat_table_mul(), self.get_indices_mul())
                return self.create_expansion_with_coeff(multiplied_coeff)

            except ValueError:
                raise ValueError("Expansions to be multiplied are not from the same algebra")

        if isinstance(other, (MapAbstract, np.ndarray)):
            return other * self

        # scalar case
        return self.create_expansion_with_coeff(other * self._coeff)

    @staticmethod
    @njit(cache=True)
    def pow2_multivar(coeff: np.ndarray, square_ind: np.ndarray,
                      table_mul: np.ndarray, indices_mul: np.ndarray) -> np.ndarray:
        """Static method transforming coefficients into the coefficients of the square of a multivariate
        Taylor expansion. It emulates the polynomial product and the truncation at the same time.

        Args:
            coeff (numpy.ndarray): set of coefficients.
            square_ind (numpy.ndarray): precomputed indices corresponding to monomials which are the square
                of another monomial in the algebra.
            table_mul (np.ndarray): flattened algebra's multiplication table.
            indices_mul (numpy.ndarray): algebra's multiplication indices.

        Returns:
            numpy.ndarray: coefficients corresponding to square.

        """
        twice_coeff = 2. * coeff
        pow2_coeff = coeff[0] * twice_coeff
        squared_terms = coeff[:len(square_ind)] ** 2
        pow2_coeff[0] = 0.
        pow2_coeff[square_ind] += squared_terms
        slices = indices_mul[2:] - indices_mul[1:-1]

        for i, (slice_index, el) in enumerate(zip(slices, coeff[2:]), 2):
            pow2_coeff[table_mul[indices_mul[i - 1] + 1:indices_mul[i]]] += twice_coeff[1:slice_index] * el

        return pow2_coeff

    def pow2(self) -> "ComplexMultivarTaylor":
        """Method to raise Taylor expansion to the power 2 i.e. compute its multiplicative square.

        Returns:
            ComplexMultivarTaylor: Taylor expansion raised to power 2.

        """
        coeff = self.pow2_multivar(self._coeff, self.get_square_indices(), self.get_flat_table_mul(),
                                   self.get_indices_mul())
        return self.create_expansion_with_coeff(coeff)

    @staticmethod
    @njit(cache=True)
    def reciprocal_multivar(coeff: np.ndarray, square_ind: np.ndarray,
                            table_mul: np.ndarray, indices_mul: np.ndarray) -> np.ndarray:
        """Static method transforming coefficients into the coefficients of the reciprocal of a multivariate
        Taylor expansion. Uses recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): set of coefficients.
            square_ind (numpy.ndarray): precomputed indices corresponding to monomials which are the square
                of another monomial in the algebra.
            table_mul (np.ndarray): flattened algebra's multiplication table.
            indices_mul (numpy.ndarray): algebra's multiplication indices.

        Returns:
            numpy.ndarray: coefficients corresponding to reciprocal.

        """
        new_coeff = np.zeros_like(coeff)
        new_coeff[0] = 1. / coeff[0]
        new_coeff[1:] = -coeff[1:] * (new_coeff[0] ** 2)
        new_coeff[square_ind[1]] -= coeff[1] * new_coeff[1] * new_coeff[0]
        slices = indices_mul[2:] - indices_mul[1:-1]
        for i, (slice_index, el) in enumerate(zip(slices, coeff[2:]), 2):
            if i < len(square_ind):
                new_coeff[square_ind[i]] -= el * new_coeff[i] * new_coeff[0]
            new_coeff[table_mul[indices_mul[i - 1] + 1:indices_mul[i]]] -= (new_coeff[i] * coeff[1:slice_index] +
                                                                            el * new_coeff[1:slice_index]) * new_coeff[0]
        return new_coeff

    def reciprocal(self) -> "ComplexMultivarTaylor":
        new_coeff = self.reciprocal_multivar(self._coeff, self.get_square_indices(),
                                             self.get_flat_table_mul(), self.get_indices_mul())
        return self.create_expansion_with_coeff(new_coeff)

    @staticmethod
    @njit(cache=True)
    def division_multivar(coeff: np.ndarray, other_coeff: np.ndarray, square_ind: np.ndarray,
                          table_mul: np.ndarray, indices_mul: np.ndarray) -> np.ndarray:
        """Static method transforming coefficients of two multivariate Taylor expansions into the coefficients of their
        ratio. Uses recursive formula from Neidinger 2013.

        Args:
            coeff (numpy.ndarray): set of coefficients for numerator.
            other_coeff (numpy.ndarray): set of coefficients for denominator.
            square_ind (numpy.ndarray): precomputed indices corresponding to monomials which are the square
                of another monomial in the algebra.
            table_mul (np.ndarray): flattened algebra's multiplication table.
            indices_mul (numpy.ndarray): algebra's multiplication indices.

        Returns:
            numpy.ndarray: coefficients corresponding to division.

        """
        new_coeff = coeff / other_coeff[0]
        new_coeff[1:] -= other_coeff[1:] * new_coeff[0] / other_coeff[0]
        new_coeff[square_ind[1]] -= other_coeff[1] * new_coeff[1] / other_coeff[0]
        slices = indices_mul[2:] - indices_mul[1:-1]
        for i, (slice_index, el2) in enumerate(zip(slices, other_coeff[2:]), 2):
            if i < len(square_ind):
                new_coeff[square_ind[i]] -= el2 * new_coeff[i] / other_coeff[0]
            new_coeff[table_mul[indices_mul[i - 1] + 1:indices_mul[i]]] -= (new_coeff[i] * other_coeff[1:slice_index] +
                                                                            el2 * new_coeff[1:slice_index]) /\
                                                                           other_coeff[0]
        return new_coeff

    def __truediv__(self, other: Union[Scalar, "ComplexMultivarTaylor"]) -> "ComplexMultivarTaylor":
        if isinstance(other, self.__class__):
            ratio_coeff = self.division_multivar(self._coeff, other._coeff, self.get_square_indices(),
                                                 self.get_flat_table_mul(), self.get_indices_mul())
            return self.create_expansion_with_coeff(ratio_coeff)

        # scalar case
        return self * (1. / other)

    def __rtruediv__(self, other: Union[Scalar, "ComplexMultivarTaylor"]) -> "ComplexMultivarTaylor":
        if isinstance(other, self.__class__):
            ratio_coeff = self.division_multivar(other._coeff, self._coeff, self.get_square_indices(),
                                                 self.get_flat_table_mul(), self.get_indices_mul())
            return self.create_expansion_with_coeff(ratio_coeff)

        # scalar case
        return other * self.reciprocal()

    @staticmethod
    @njit(cache=True)
    def sqrt_multivar(preprocessed_coeff: np.ndarray, square_ind: np.ndarray,
                      table_mul: np.ndarray, indices_mul: np.ndarray) -> np.ndarray:
        """Static method transforming coefficients into the coefficients of the square root of a multivariate
        Taylor expansion. Uses recursive formula from Neidinger 2013.

        Args:
            preprocessed_coeff (numpy.ndarray): coefficients with some pre-computations done.
            square_ind (numpy.ndarray): precomputed indices corresponding to monomials which are the square
                of another monomial in the algebra.
            table_mul (np.ndarray): flattened algebra's multiplication table.
            indices_mul (numpy.ndarray): algebra's multiplication indices.

        Returns:
            numpy.ndarray: coefficients corresponding to square root.

        """
        slices = indices_mul[2:] - indices_mul[1:-1]
        factor = 1. / (2. * preprocessed_coeff[0])
        preprocessed_coeff[square_ind[1]] -= preprocessed_coeff[1]**2 * factor
        for i, slice_index in enumerate(slices, 2):
            if i < len(square_ind):
                preprocessed_coeff[square_ind[i]] -= preprocessed_coeff[i]**2 * factor
            preprocessed_coeff[table_mul[indices_mul[i - 1] + 1:indices_mul[i]]] -= preprocessed_coeff[i] * \
                                                                                    preprocessed_coeff[1:slice_index] \
                                                                                    / preprocessed_coeff[0]
        return preprocessed_coeff

    def sqrt(self) -> "ComplexMultivarTaylor":
        sqrt_cst = self._sqrt_cst(self._coeff[0])
        preprocessed_coeff = self._coeff / (2. * sqrt_cst)
        preprocessed_coeff[0] = sqrt_cst
        new_coeff = self.sqrt_multivar(preprocessed_coeff, self.get_square_indices(),
                                       self.get_flat_table_mul(), self.get_indices_mul())
        return self.create_expansion_with_coeff(new_coeff)

    def rigorous_integ_once_wrt_var(self, index_var: int) -> "ComplexMultivarTaylor":
        """Method performing integration with respect to a given unknown variable. The integration constant is zero.
        This transformation is rigorous as the order of the expansion is increased by one. In other words, the output
        lives in another algebra, of higher dimension.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            ComplexMultivarTaylor: integrated Taylor expansion w.r.t. input variable number.

        """
        new_expansion = self.__class__(self._n_var, self._order + 1, self._var_names)
        new_coeff = np.zeros(new_expansion.dim_alg, dtype=self._var_type)
        inv_exponents_plus_one = 1. / np.arange(1., self._order + 2)
        new_mapping = new_expansion.get_mapping_monom()
        for exponent, index_coeff in self.get_mapping_monom().items():
            new_tuple = exponent[:index_var] + (exponent[index_var] + 1,) + exponent[index_var + 1:]
            new_coeff[new_mapping[new_tuple]] = self._coeff[index_coeff] * inv_exponents_plus_one[exponent[index_var]]
        new_expansion.coeff = new_coeff
        return new_expansion

    def integ_once_wrt_var(self, index_var: int) -> "ComplexMultivarTaylor":
        """Method performing integration with respect to a given unknown variable while remaining in the same
        algebra. The integration constant is zero. This transformation is not rigorous in the sense that the order of
        the expansion should be increased by one.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            ComplexMultivarTaylor: integrated Taylor expansion w.r.t. input variable number.

        """
        if self.is_trivial():
            return self.create_null_expansion()

        # order of at least one
        coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        nb = algebra_dim(self._n_var, self._order - 1)
        weights = np.array(list(self.get_mapping_monom().keys()))[:nb, index_var] + 1.
        coeff[self.get_table_deriv()[:nb, index_var]] = self._coeff[:nb] / weights
        return self.create_expansion_with_coeff(coeff)

    def deriv_once_wrt_var(self, index_var: int) -> "ComplexMultivarTaylor":
        """Method performing differentiation with respect to a given unknown variable while remaining in the same
        algebra. This transformation is not rigorous in the sense that the order of the expansion should be decreased by
        one.

        Args:
            index_var (int): variable number w.r.t. which differentiation needs to be performed.

        Returns:
            ComplexMultivarTaylor: differentiated Taylor expansion w.r.t. input variable number.

        """
        if self.is_trivial():
            return self.create_null_expansion()

        # order of at least one
        coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        nb = algebra_dim(self._n_var, self._order - 1)
        links = self.get_table_deriv()[:nb, index_var]
        integers = np.array(list(self.get_mapping_monom().keys()))[links, index_var]
        coeff[:nb] = self._coeff[links] * integers
        return self.create_expansion_with_coeff(coeff)

    def compose(self, other) -> "ComplexMultivarTaylor":
        """Method performing composition with inputted Taylor map (must have the same order and as many elements as
        self has variables). The result is a single expansion and has the same variables than the input.

        Args:
            other (ComplexTaylorMap): Taylor map to be composed on the right-hand side.

        Returns:
            ComplexMultivarTaylor: composed expansion.

        """
        from swiftt.taylor.taylor_map import ComplexTaylorMap
        if not isinstance(other, ComplexTaylorMap):
            raise ValueError("Multivariate expansions can only be composed on the right by Taylor maps")

        if other.order != self._order:
            raise ValueError("Expansions to be composed must have same order")

        if not other.is_nilpotent():
            raise ValueError("Right-hand-side map for composition must be nilpotent")

        if len(other) != self._n_var:
            raise ValueError("Right-hand-side map does not have as many elements as left-hand side has"
                             "variables")

        if self.is_trivial():
            return other[0].create_const_expansion(self.const)

        # order of at least one
        mapping = self.get_mapping_monom()
        powers = list(mapping.keys())
        already_computed = dict(zip(powers[1:self.n_var + 1], other))

        # define recursive function computing required products of expansions with memoization
        def lazy(power: Tuple[int, ...]) -> ComplexMultivarTaylor:
            k = new_tuple = None
            for index_var, power_var in enumerate(power):
                if power_var > 0:
                    k = index_var
                    new_tuple = power[:index_var] + (power_var - 1,) + power[index_var + 1:]
                    try:
                        already_computed[power] = already_computed[new_tuple] * other[index_var]
                        # this product could be obtained with an already computed quantity, so go to return
                        break
                    except KeyError:
                        # keep trying with other variables
                        pass
            else:  # no break
                already_computed[power] = lazy(new_tuple) * other[k]  # recursive call
            return already_computed[power]

        rhs_coeff = np.zeros((self._dim_alg, other[0].dim_alg), dtype=self._var_type)
        rhs_coeff[0, 0] = 1.
        rhs_coeff[1:self.n_var + 1, :] = other.coeff
        # this implementation is optimized for a sparse expansion (many vanishing coeff) on the left-hand side
        for j in (np.nonzero(self._coeff[self._n_var + 1:])[0] + self._n_var + 1):
            rhs_coeff[j, :] = lazy(powers[j]).coeff

        return other[0].create_expansion_with_coeff(self._coeff.dot(rhs_coeff))

    def pointwise_eval(self, x: np.ndarray) -> np.ndarray:
        """Method for the evaluation of the Taylor expansion on a given point.

        Args:
            x (numpy.ndarray): point of evaluation.

        Returns:
            numpy.ndarray: Taylor expansion evaluated at given point.

        """

        if self.is_trivial():
            return self.const

        if len(x) == self._n_var:

            mapping = self.get_mapping_monom()
            products = np.ones(self._dim_alg, dtype=self._var_type)
            for exponent, index_coeff in mapping.items():
                for index_var, power_var in enumerate(exponent):
                    if power_var > 0:
                        new_tuple = exponent[:index_var] + (power_var - 1,) + exponent[index_var + 1:]
                        products[index_coeff] = x[index_var] * products[mapping[new_tuple]]
                        break

            return self._coeff.dot(products)

        raise ValueError("Inconsistent point for evaluation")

    def massive_eval(self, Xs: np.ndarray) -> np.ndarray:
        """Method for the evaluation of the Taylor expansion on a range of points (vectorized evaluation).

        Args:
            Xs (numpy.ndarray): points of evaluation.

        Returns:
            numpy.ndarray: Taylor expansion evaluated at given points.

        """

        if Xs.shape[1] != self._n_var:
            raise IndexError("The number of columns of the input should equal the number of variables in the "
                             "expansion.")

        mapping = self.get_mapping_monom()
        products = np.ones((self._dim_alg, Xs.shape[0]), dtype=self._var_type)
        for exponent, index_coeff in mapping.items():
            for index_var, power_var in enumerate(exponent):
                if power_var > 0:
                    new_tuple = exponent[:index_var] + (power_var - 1,) + exponent[index_var + 1:]
                    products[index_coeff, :] = Xs[:, index_var] * products[mapping[new_tuple], :]

        return self._coeff.dot(products)

    def __call__(self, *args, **kwargs):
        """Method for calling the Taylor expansion. Wraps several possibilities: evaluation and composition with a map.

        Returns:
            Union[ComplexMultivarTaylor, numpy.ndarray]: Taylor expansion called on input.

        """
        return self.compose(args[0]) if isinstance(args[0][0], ComplexMultivarTaylor) else self.pointwise_eval(args[0])

    @property
    def remainder_term(self) -> str:
        return TaylorExpansAbstract.landau_multivar(self._order, self._var_names)

    @property
    def const(self) -> Union[float, complex]:
        """
        Getter for the so-called constant coefficient i.e. associated to the zeroth-order contribution.

        Returns:
            Union[float, complex]: constant.

        """
        return self._coeff[0]

    @const.setter
    def const(self, cst: Union[float, complex]) -> None:
        """
        Setter for the so-called constant coefficient i.e. associated to the zeroth-order contribution.

        Args:
            cst (Union[float, complex]): new constant.

        """
        self._coeff[0] = cst

    def __str__(self) -> str:
        """Method to cast complex Taylor expansion as a string.

        Returns:
            str: string representing the Taylor expansion.

        """
        string = str(self.const)
        var_names = self.var_names
        if not self.is_trivial():
            for exponent, index_coeff in self.get_mapping_monom().items():
                if self._coeff[index_coeff] != 0. and index_coeff != 0:
                    string += " + " + str(self._coeff[index_coeff])
                    for index_var, power_var in enumerate(exponent):
                        if power_var != 0:
                            string += " * " + var_names[index_var]
                            if power_var > 1:
                                string += "**" + str(power_var)
        return string + " + " + self.remainder_term

    def divided_by_var(self, index_var: int) -> "ComplexMultivarTaylor":
        """Method returning the Taylor expansion divided by the input variable. This is not a rigorous operation as the
        order should be decreased by one.

        Args:
            index_var (str): index of variable to divide with.

        Returns:
            ComplexMultivarTaylor: Taylor expansion divided by variable.

        """

        if self.const != 0.:
            raise ValueError("Cannot divide by variable if constant term is not zero.")

        coeff = np.zeros(self._dim_alg, dtype=self._var_type)

        mapping = self.get_mapping_monom()
        for exponent, index_coeff in mapping.items():
            if self._coeff[index_coeff] != 0.:
                if exponent[index_var] == 0:
                    raise ValueError("Cannot divide by variable if there are non-zero terms without this variable.")

                new_exponent = np.array(exponent, dtype=int)
                new_exponent[index_var] -= 1
                coeff[mapping[tuple(new_exponent)]] = self._coeff[index_coeff]

        return self.create_expansion_with_coeff(coeff)

    def contrib_removed(self, indices_var: List[int]) -> "ComplexMultivarTaylor":
        """Method returning a Taylor expansion where all coefficients associated to input variables' indices are set to
        zero.

        Args:
            indices_var (List[int]): indices of variables whose contribution is to be removed.

        Returns:
            ComplexMultivarTaylor: Taylor expansion with removed contributions.

        """

        coeff = np.array(self._coeff)
        try:
            exponents = np.array(list(self.get_mapping_monom().keys()))[:, indices_var]
        except TypeError:
            raise ValueError("At least one inputted variable index is not an integer")
        except IndexError:
            raise ValueError("At least one inputted variable index does not exist in this algebra")
        for i in range(0, len(indices_var)):
            coeff[exponents[:, i] != 0] = 0.
        return self.create_expansion_with_coeff(coeff)

    def var_eval(self, index_var: int, value: Scalar) -> "ComplexMultivarTaylor":
        """Method returning a Taylor expansion where a variable has been replaced by a fixed scalar value. In other
        words, it is a partial evaluation of the polynomial part. It is not rigorous as terms of higher order hidden in
        the remainder would need to be considered in this operation.

        Args:
            index_var (int): index of variable to be evaluated.
            value (Union[complex, float]): value to replace given variable.

        Returns:
            ComplexMultivarTaylor: Taylor expansion with removed dependency.

        """
        if index_var >= self._n_var or index_var < 0:
            raise IndexError("The inputted index does not correspond to any variable of the expansion.")
        powers = np.cumprod(np.full(self._order, value, dtype=self._var_type))
        new_coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        mapping = self.get_mapping_monom()
        tuple_0 = (0,)
        for exponent, index_coeff in mapping.items():
            if exponent[index_var] == 0:
                new_coeff[index_coeff] += self._coeff[index_coeff]
            else:
                new_exponent = exponent[:index_var] + tuple_0 + exponent[index_var + 1:]
                new_coeff[mapping[new_exponent]] += self._coeff[index_coeff] * powers[exponent[index_var] - 1]
        return self.create_expansion_with_coeff(new_coeff)

    def truncated(self, new_order: int) -> "ComplexMultivarTaylor":
        """Method for the truncation at a given order. Output lives in another algebra, of lower dimension.

        Args:
            new_order (int): order of algebra in which to truncate the Taylor expansion.

        Returns:
            ComplexMultivarTaylor: Taylor expansion truncated at input order.

        """
        if new_order < 0 or new_order >= self._order or self.is_trivial():
            raise ValueError("Input order must be an integer between zero and order of initial algebra")

        truncated = self.__class__(self._n_var, new_order, self._var_names)
        truncated.coeff = self._coeff[:truncated.dim_alg]
        return truncated

    def prolong(self, new_order: int) -> "ComplexMultivarTaylor":
        """Method for the prolongation (opposite of truncation) at a given order. Output lives in another algebra, of
        higher dimension. It is not a rigorous operation as the possible contributions within the former remainder are
        ignored.

        Args:
            new_order (int): order of algebra in which to prolong the Taylor expansion.

        Returns:
            ComplexMultivarTaylor: Taylor expansion prolonged at input order.

        """
        if new_order <= self._order:
            raise ValueError("Input order must be an integer greater than order of initial algebra")

        prolonged = self.__class__(self._n_var, new_order, self._var_names)
        new_coeff = np.zeros(prolonged.dim_alg, dtype=self._var_type)
        new_coeff[:self._dim_alg] = np.array(self._coeff, dtype=self._var_type)
        prolonged.coeff = new_coeff
        return prolonged

    def prolong_one_order(self) -> "ComplexMultivarTaylor":
        """Method for the prolongation (opposite of truncation) of one order. Output lives in another algebra, of
        higher dimension. It is not a rigorous operation as the possible contributions within the former remainder are
        ignored.

        Returns:
            ComplexMultivarTaylor: Taylor expansion prolonged by one order.

        """
        prolonged = self.__class__(self._n_var, self._order + 1, self._var_names)
        new_coeff = np.zeros(prolonged.dim_alg, dtype=self._var_type)
        new_coeff[:self._dim_alg] = np.array(self._coeff, dtype=self._var_type)
        prolonged.coeff = new_coeff
        return prolonged

    def var_inserted(self, index_new_var: int, unknown_name: Optional[str] = None) -> "ComplexMultivarTaylor":
        """Method for the addition of a new variable. Output lives in another algebra, of higher dimension. All its
        terms associated with the new variable are zero and the other ones are identical to original expansion.

        Args:
            index_new_var (int): index of new variable to be added.
            unknown_name (str): name of new variable.

        Returns:
            ComplexMultivarTaylor: Taylor expansion with an additional variable.

        """

        if unknown_name is None:
            unknown_name = default_unknown_name + str(self._n_var + 1)
        if unknown_name in self.var_names:
            raise ValueError("Proposed name of new variable already exists.")

        new_var_names = list(self._var_names)
        new_var_names.append(unknown_name)
        new_expansion = self.__class__(self._n_var + 1, self._order, new_var_names)

        new_coeff = np.zeros(new_expansion.dim_alg, dtype=self._var_type)
        if not self.is_trivial():
            for new_exponent, new_index_coeff in new_expansion.get_mapping_monom().items():
                if new_exponent[index_new_var] == 0:
                    new_tuple = new_exponent[:index_new_var] + new_exponent[index_new_var + 1:]
                    new_coeff[new_index_coeff] = self._coeff[self.get_mapping_monom()[new_tuple]]
        else:
            new_coeff[0] = self.const

        new_expansion.coeff = new_coeff
        return new_expansion

    def var_removed(self, index_var: int) -> "ComplexMultivarTaylor":
        """Method for the removal of a variable. Output lives in another algebra, of smaller dimension. All its
        terms associated with the old variables only are identical to original expansion.

        Args:
            index_var (int): index of variable to be removed.

        Returns:
            ComplexMultivarTaylor: Taylor expansion with a variable removed.

        """
        if self._n_var == 1:
            raise ValueError("A univariate expansion cannot have variables removed.")
        if index_var >= self._n_var or index_var < 0:
            raise IndexError("The inputted index does not correspond to any variable of the expansion.")

        new_var_names = list(self.var_names)
        del new_var_names[index_var]

        if self._n_var == 2:
            # TODO: improve the treatment of this case
            if self._var_type == np.complex128:
                from swiftt.taylor.complex_univar_taylor import ComplexUnivarTaylor
                new_expansion = ComplexUnivarTaylor(self._order, new_var_names)
            else:
                from swiftt.taylor.real_univar_taylor import RealUnivarTaylor
                new_expansion = RealUnivarTaylor(self._order, new_var_names)

            coeff = np.zeros(new_expansion.dim_alg, dtype=self._var_type)
            if not self.is_trivial():
                for old_exponent, index_old_coeff in self.get_mapping_monom().items():
                    if old_exponent[index_var] == 0:
                        coeff[sum(old_exponent)] = self._coeff[index_old_coeff]
            else:
                coeff[0] = self.const

        else:
            new_expansion = self.__class__(self._n_var - 1, self._order, new_var_names)
            coeff = np.zeros(new_expansion.dim_alg, dtype=self._var_type)
            if not self.is_trivial():
                new_mapping = new_expansion.get_mapping_monom()
                for old_exponent, index_old_coeff in self.get_mapping_monom().items():
                    if old_exponent[index_var] == 0:
                        new_tuple = old_exponent[:index_var] + old_exponent[index_var + 1:]
                        coeff[new_mapping[new_tuple]] = self._coeff[index_old_coeff]
            else:
                coeff[0] = self.const

        new_expansion.coeff = coeff
        return new_expansion

    def create_expansion_from_smaller_algebra(self, expansion: "ComplexMultivarTaylor") -> "ComplexMultivarTaylor":
        """Method to create a Taylor expansion in same algebra than self, from an expansion in another algebra with
        same order but less variables. Names of intersecting variables must be identical otherwise the function does not
        work.

        Args:
            expansion (ComplexMultivarTaylor): Taylor expansion to extend in current algebra.

        Returns:
            ComplexMultivarTaylor: Taylor expansion whose polynomial coefficients are all zero except the
                ones related only to the variables of the input that are then identical to them.

        """
        if expansion.order != self._order:
            raise ValueError("The inputted expansion has a different order.")
        if expansion.n_var >= self._n_var:
            raise ValueError("The inputted expansion is not from an algebra with less variables.")

        expansion_coeff = expansion.coeff
        new_coeff = np.zeros(self._dim_alg, dtype=self._var_type)
        coeff_old_algebra = np.zeros(expansion.dim_alg, dtype=self._var_type)
        if expansion.n_var == 1:
            # the monomials-coefficients mapping is trivial in the univariate case (hence no dedicated function)
            old_mapping = {(j,): j for j in range(0, expansion.order + 1)}
        else:
            old_mapping = expansion.get_mapping_monom()

        for old_exponent, old_index_var in old_mapping.items():
            coeff_old_algebra[:old_index_var] = 0.
            coeff_old_algebra[old_index_var] = 1.
            old_monomial = expansion.create_expansion_with_coeff(coeff_old_algebra)
            str_old_monomial = str(old_monomial).split(landau_symbol)[0]
            coeff_new_algebra = np.zeros(self._dim_alg, dtype=self._var_type)
            for new_exponent, new_index_var in self.get_mapping_monom().items():
                coeff_new_algebra[:new_index_var] = 0.
                if sum(new_exponent) == sum(old_exponent):
                    coeff_new_algebra[new_index_var] = 1.
                    str_new_monomial = str(self.create_expansion_with_coeff(coeff_new_algebra)).split(landau_symbol)[0]
                    if str_new_monomial == str_old_monomial:
                        new_coeff[new_index_var] = expansion_coeff[old_index_var]
                        break
            else:  # no break
                raise ValueError
        return self.create_expansion_with_coeff(new_coeff)

    @property
    def effective_order(self) -> int:
        """Method returning the effective order of the expansion, that is the highest order with non-zero coefficients.

        Returns:
            int: effective order of Taylor expansion.

        """
        for exponent, el in zip(reversed(list(self.get_mapping_monom().keys())), self._coeff[::-1]):
            if el != 0.:
                return sum(exponent)
        return 0

    @property
    def total_depth(self) -> int:
        """Method returning the total depth of the Taylor expansion, that is the lowest order with non-zero coefficient.

        Returns:
            int: total depth of Taylor expansion.

        """
        for exponents, index_coeff in self.get_mapping_monom().items():
            if self._coeff[index_coeff] != 0.:
                return sum(exponents)
        return self._order + 1

    @property
    def norm(self) -> float:
        """Method returning the norm of the Taylor expansion as defined by M. Berz. This quantity depends on the order.

        Returns:
            float: expansion's norm.

        """
        return np.max(np.abs(self._coeff)) * self._dim_alg

    def __eq__(self, other: Union["ComplexMultivarTaylor", complex, float]) -> bool:
        """Method enabling equality comparison with other Taylor expansions and scalars.

        Args:
            other (Union[ComplexMultivarTaylor, complex, float]): quantity to test equality on.

        Returns:
            bool: if the input is a Taylor expansion, returns True if it is in the same algebra and has the
                same coefficients. If the input is a scalar, checks if it equals the constant coefficients and if
                all the others are zero.

        """
        if isinstance(other, ComplexMultivarTaylor):

            if not self.is_in_same_algebra(other):
                return False

            # if expansions have same order and number of variables, then compare their coefficients
            return np.array_equal(self._coeff, other._coeff)

        # scalar case
        return self == self.create_const_expansion(other)
