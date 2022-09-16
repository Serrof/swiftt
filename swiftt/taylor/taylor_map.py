# taylor_map.py: range of classes implementing maps of Taylor expansion
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

from typing import List, Optional, Iterable, Union
import numpy as np
from numba import njit
from swiftt.taylor.complex_multivar_taylor import ComplexMultivarTaylor, Scalar
from swiftt.taylor.taylor_expans_abstract import TaylorExpansAbstract, landau_symbol, default_inverse_name
from swiftt.taylor.tables import algebra_dim
from swiftt.interval import Interval
from swiftt.map_abstract import MapAbstract


class ComplexTaylorMap(TaylorExpansAbstract, MapAbstract):
    """Class for so-called Taylor map i.e. vectors of Taylor expansions in same algebra (same order and
        number of variables).

    Attributes:
        _coeff (numpy.ndarray): 2-D array grouping all the polynomial coefficients of the various map's
            elements (columns are along the algebra's dimension). Allows for faster manipulation.
        _var_type (type): type of coefficients (and implicitly the variables).


    """

    # intrinsic functions for arrays
    _sqrt_cst = np.sqrt
    _log_cst, _exp_cst = np.log, np.exp
    _cos_cst, _sin_cst = np.cos, np.sin
    _cosh_cst, _sinh_cst = np.cosh, np.sinh

    def __init__(self, expansions) -> None:
        """Constructor for Taylor maps.

        Args:
            expansions (List[ComplexMultivarTaylor}): Taylor expansions forming the map. Order is preserved.

        """
        MapAbstract.__init__(self, expansions)
        self._n_var = expansions[0].n_var
        self._order = expansions[0].order
        self._dim_alg = expansions[0].dim_alg
        self._var_type = expansions[0].var_type

        self._coeff = np.empty((self._len, self._dim_alg), dtype=self._var_type)
        for i, expansion in enumerate(expansions):
            self._coeff[i, :] = expansion.coeff

    @property
    def remainder_term(self) -> str:
        return self[0].remainder_term

    @property
    def const(self) -> np.ndarray:
        """
        Getter for the so-called constant coefficients i.e. associated to the zeroth-order contribution.

        Returns:
            Union[float, complex]: constants.

        """
        return np.array(self._coeff[:, 0])

    @const.setter
    def const(self, cst: Union[complex, float, np.ndarray]) -> None:
        """
        Setter for the so-called constant coefficients i.e. associated to the zeroth-order contribution.
        If a scalar is given, it replaces all the constant coefficients.

        Args:
            Union[float, complex]: new constants.

        """
        if isinstance(cst, (complex, float)):
            self.const = np.full(self._len, cst)
        elif len(cst) != self._len:
            raise ValueError("Inconsistent number of constants given for this map (" +
                             str(len(cst)) + " instead of " + str(self._len) + ")")
        else:
            for i, el in enumerate(cst):
                self[i].const = el
            self._coeff[:, 0] = np.array(cst, dtype=self._var_type)

    @TaylorExpansAbstract.order.setter
    def order(self, new_order: int) -> None:
        if new_order != self._order:
            for i in range(0, self._len):
                self[i].order = new_order
            self._order = new_order
            self._dim_alg = self[0].dim_alg

    @TaylorExpansAbstract.n_var.setter
    def n_var(self, n_var: int) -> None:
        if n_var != self._n_var:
            for i in range(0, self._len):
                self[i].n_var = n_var
            self._n_var = n_var
            self._dim_alg = self[0].dim_alg

    @property
    def coeff(self) -> np.ndarray:
        return np.array(self._coeff)
    
    @coeff.setter
    def coeff(self, coefficients: np.ndarray) -> None:
        if coefficients.shape == (self._len, self._dim_alg):
            self._coeff = np.array(coefficients, dtype=self._var_type)
        else:
            raise ValueError("The given coefficients do not match the size of the map.")

    def __setitem__(self, item: int, expansion: ComplexMultivarTaylor) -> None:
        if not self.is_in_same_algebra(expansion):
            raise ValueError("Input expansion is not in the same algebra than the Taylor map.")
        self._coeff[item, :] = expansion.coeff
        MapAbstract.__setitem__(self, item, expansion)

    def __neg__(self) -> "ComplexTaylorMap":
        """Method defining negation (additive inverse). Overwrites parent implementation for performance.

        Returns:
            ComplexTaylorMap: opposite of object (from the point of view of addition).

        """
        return self.create_map_with_coeff(-self._coeff)

    def create_map_with_coeff(self, new_coeff: np.ndarray) -> "ComplexTaylorMap":
        """Method to create a new Taylor map (in same algebra and with same size than self) from given coefficients.

        Args:
            new_coeff (numpy.ndarray): coefficients of new map. Rows are for each map's elements and columns
               for monomials.

        Returns:
            ComplexTaylorMap: Taylor map corresponding to inputted coefficients.

        """
        return self.__class__([self[0].create_expansion_with_coeff(new_coeff[i, :]) for i in range(0, self._len)])

    def create_const_expansion(self, const: Iterable[Scalar]) -> "ComplexTaylorMap":
        """Method to create a Taylor map (in same algebra and with same size than self) whose polynomial part is
        constant.

        Args:
            const (Iterable[Scalar]): constants for each element.

        Returns:
            ComplexTaylorMap: Taylor map corresponding to inputted coefficients.

        """
        new_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        new_coeff[:, 0] = np.array(const)
        return self.create_map_with_coeff(new_coeff)

    def create_null_map(self) -> "ComplexTaylorMap":
        """Method to create a Taylor map (in same algebra and with same size than self) whose polynomial part is
        zero.

        Returns:
            ComplexTaylorMap: null Taylor map of self size and algebra.

        """
        new_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        return self.create_map_with_coeff(new_coeff)

    @TaylorExpansAbstract.var_names.setter
    def var_names(self, var_names: List[str]) -> None:
        """Setter for the name of the variables. Overwrites parent implementation to extend the new names to all
        elements of the map.

        Args:
            List[str]: name of the variables.

        """
        for i in range(0, self._len):
            self[i].var_names = var_names

    def is_univar(self) -> bool:
        """Method returning True if the map is made of univariate Taylor expansions, False otherwise.

        Returns:
            (bool): True if the map is univariate.

        """
        return self._n_var == 1

    def is_square_sized(self) -> bool:
        """Method returning True if the map has as many elements than variables, False otherwise.

        Returns:
            (bool): True if the map is square (as many variables as components).

        """
        return self._n_var == self._len

    def is_const(self) -> bool:
        """Method returning True if the polynomial part of all the map's elements are constant, False otherwise.

        Returns:
            (bool): True if and only if all non-constant coefficients are zero.

        """
        for expansion in self:
            if not expansion.is_const():
                return False
        return True

    def is_nilpotent(self) -> bool:
        """Method returning True if all the map's elements are nilpotent, False otherwise.

        Returns:
            (bool): True if the map is nilpotent.

        """
        for i in range(0, self._len):
            if self._coeff[i, 0] != 0.:
                return False
        return True

    def get_nilpo_part(self) -> "ComplexTaylorMap":
        """Method returning the nilpotent part of the Taylor map.

        Returns:
            ComplexTaylorMap: nilpotent part of Taylor map.

        """
        new_coeff = np.array(self._coeff, dtype=self._var_type)
        new_coeff[:, 0] = 0.
        return self.create_map_with_coeff(new_coeff)

    def get_linear_part(self) -> "ComplexTaylorMap":
        """Method returning the linear part of the Taylor map.

        Returns:
            ComplexTaylorMap: linear part of Taylor map.

        """
        try:
            n_var_plus_one = self._n_var + 1
            new_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
            new_coeff[:, :n_var_plus_one] += self._coeff[:, :n_var_plus_one]
            return self.create_map_with_coeff(new_coeff)
        except IndexError:
            raise ValueError("There is no linear part in a zeroth-order expansion.")

    def get_high_order_part(self, order: int) -> "ComplexTaylorMap":
        """Method returning the high order part of the map in the same algebra. It is not a rigorous operation.

        Args:
            order (int): order (included) below which all contributions are removed.

        Returns:
            ComplexTaylorMap: high order part of the Taylor map.

        """
        if 1 <= order <= self._order:
            map_coeff = np.array(self._coeff, dtype=self._var_type)
            map_coeff[:, :algebra_dim(self._n_var, order - 1)] = 0.
            return self.create_map_with_coeff(map_coeff)
        raise ValueError("The inputted order exceeds the current order.")

    def get_low_order_part(self, order: int) -> "ComplexTaylorMap":
        """Method returning the low order part of the map in the same algebra. It is not a rigorous operation.

        Args:
            order (int): order (included) above which all contributions are removed.

        Returns:
            ComplexTaylorMap: low order part of the Taylor map.

        """
        if order == 0:
            return self.get_const_part()
        if order < self._order:
            map_coeff = np.array(self._coeff, dtype=self._var_type)
            map_coeff[:, algebra_dim(self._n_var, order):] = 0.
            return self.create_map_with_coeff(map_coeff)
        raise ValueError("The inputted order exceeds the current order.")

    def get_low_order_wrt_var(self, index_var: int, order: int) -> "ComplexTaylorMap":
        new_coeff = self.coeff
        mapping = self[0].get_mapping_monom()
        indices_coeff = [mapping[exponent] for exponent in mapping if exponent[index_var] > order]
        new_coeff[:, indices_coeff] = 0.
        return self.create_map_with_coeff(new_coeff)

    def linearly_combine_with_another(self, alpha: Scalar, expansion: "ComplexTaylorMap",
                                      beta: Scalar) -> "ComplexTaylorMap":
        """Method multiplying with a scalar and then adding with a another expansion also multiplied by some scalar.
        Overwritten for speed purposes.

        Args:
            alpha (Union[complex, float]): multiplier for self.
            expansion (ComplexMultivarTaylor): expansion to be linearly combined with.
            beta (Union[complex, float]): multiplier for other expansion.

        Returns:
            (ComplexMultivarTaylor): linear combination of self and arguments.

        """
        return self.create_map_with_coeff(alpha * self._coeff + beta * expansion.coeff)

    def deriv_once_wrt_var(self, index_var: int) -> "ComplexTaylorMap":
        """Method performing differentiation with respect to a given unknown variable while remaining in the same
        algebra. This transformation is not rigorous in the sense that the order of the map should be decreased by
        one.

        Args:
            index_var (int): variable number w.r.t. which differentiation needs to be performed.

        Returns:
            ComplexTaylorMap: differentiated Taylor map w.r.t. input variable number.

        """
        if self.is_trivial():
            return self.create_null_map()

        # order of at least one
        new_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        if self.is_univar():
            new_coeff[:, :-1] = self._coeff[:, 1:] * np.arange(1, self.dim_alg)
        else:
            # multivariate case
            nb = algebra_dim(self._n_var, self._order - 1)
            links = self[0].get_table_deriv()[:nb, index_var]
            integers = np.array(list(self[0].get_mapping_monom().keys()))[links, index_var]
            new_coeff[:, :nb] = self._coeff[:, links] * integers
        return self.create_map_with_coeff(new_coeff)

    def integ_once_wrt_var(self, index_var: int) -> "ComplexTaylorMap":
        """Method performing integration with respect to a given unknown variable while remaining in the same
        algebra. The integrations constant are zero. This transformation is not rigorous in the sense that the order of
        the map should be increased by one.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            ComplexTaylorMap: integrated Taylor map w.r.t. input variable number.

        """
        new_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        if self.is_univar():
            new_coeff[:, 1:] = self._coeff[:, :-1] / np.arange(1, self.dim_alg)
        else:
            # multivariate case
            nb = algebra_dim(self._n_var, self._order - 1)
            weights = np.array(list(self[0].get_mapping_monom().keys()))[:nb, index_var] + 1.
            new_coeff[:, self[0].get_table_deriv()[:nb, index_var]] = self._coeff[:, :nb] / weights
        return self.create_map_with_coeff(new_coeff)

    def rigorous_integ_once_wrt_var(self, index_var: int) -> "ComplexTaylorMap":
        """Method performing integration with respect to a given unknown variable. The integration constants are zero.
        This transformation is rigorous as the order of the map is increased by one. In other words, the output
        lives in another algebra, of higher dimension.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            ComplexTaylorMap: integrated Taylor map w.r.t. input variable number.

        """

        new_coeff = np.zeros((self._len, algebra_dim(self._n_var, self._order + 1)),
                                 dtype=self._var_type)
        if self.is_univar():
            new_expansions = [self[0].__class__(self._order + 1, self[0].var_names)]
            new_coeff[:, 1:] = self._coeff / np.arange(1, self.order + 2)
        else:
            # multivariate case
            new_expansions = [self[0].__class__(self._n_var, self._order + 1, self[0].var_names)]
            inv_exponents_plus_one = 1. / np.arange(1., self._order + 2)
            new_mapping = new_expansions[0].get_mapping_monom()
            for exponent, index_coeff in self[0].get_mapping_monom().items():
                new_tuple = exponent[:index_var] + (exponent[index_var] + 1,) + exponent[index_var + 1:]
                new_coeff[:, new_mapping[new_tuple]] = self._coeff[:, index_coeff] * \
                                                           inv_exponents_plus_one[exponent[index_var]]

        new_expansions[0].coeff = new_coeff[0, :]
        for i in range(1, self._len):
            new_expansions.append(new_expansions[0].create_expansion_with_coeff(new_coeff[i, :]))

        return self.__class__(new_expansions)

    def truncated(self, new_order: int) -> "ComplexTaylorMap":
        """Method for the truncation at a given order. Output lives in another algebra, of lower dimension.

        Args:
            new_order (int): order of algebra in which to truncate the Taylor map.

        Returns:
            ComplexTaylorMap: Taylor map truncated at input order.

        """
        truncated = [self[0].truncated(new_order)]
        new_dim = truncated[0].dim_alg
        for i in range(1, self._len):
            truncated.append(truncated[0].create_expansion_with_coeff(self._coeff[i, :new_dim]))
        return self.__class__(truncated)

    def prolong(self, new_order: int) -> "ComplexTaylorMap":
        """Method for the prolongation (opposite of truncation) at a given order. Output lives in another algebra, of
        higher dimension. It is not a rigorous operation as the possible contributions within the former remainder are
        ignored.

        Args:
            new_order (int): order of algebra in which to prolong the Taylor map.

        Returns:
            ComplexTaylorMap: Taylor expansion prolonged at input order.

        """
        prolonged = [self[0].prolong(new_order)]
        new_coeff = np.zeros(prolonged[0].dim_alg)
        for i in range(1, self._len):
            new_coeff[:self.dim_alg] = np.array(self._coeff[i, :], dtype=self._var_type)
            prolonged.append(prolonged[0].create_expansion_with_coeff(new_coeff))
        return self.__class__(prolonged)

    def __add__(self, other) -> "ComplexTaylorMap":
        """Method defining right-hand side addition. Works both with scalars and expansions (element-wise) as well as
        with other maps.

        Args:
            other (Union[ComplexTaylorMap, ComplexMultivarTaylor, complex, float]): quantity to be added.

        Returns:
            ComplexTaylorMap: Taylor map summed with argument.

        """
        if isinstance(other, ComplexTaylorMap):
            map_coeff = self._coeff + other._coeff
        elif isinstance(other, ComplexMultivarTaylor):
            map_coeff = self._coeff + other.coeff
        else:
            # scalar case
            map_coeff = np.array(self._coeff, dtype=self._var_type)
            map_coeff[:, 0] += other
        return self.create_map_with_coeff(map_coeff)

    def __sub__(self, other) -> "ComplexTaylorMap":
        """Method defining right-hand side subtraction. Works both with scalars and expansions (element-wise) as well as
        with other maps.

        Args:
            other (Union[ComplexTaylorMap, ComplexMultivarTaylor, complex, float]): quantity to be subtracted.

        Returns:
            ComplexTaylorMap: Taylor map subtracted with argument.

        """
        if isinstance(other, ComplexTaylorMap):
            map_coeff = self._coeff - other._coeff
        elif isinstance(other, ComplexMultivarTaylor):
            map_coeff = self._coeff - other.coeff
        else:
            # scalar case
            map_coeff = np.array(self._coeff, dtype=self._var_type)
            map_coeff[:, 0] -= other
        return self.create_map_with_coeff(map_coeff)

    def __rsub__(self, other) -> "ComplexTaylorMap":
        """Method defining left-hand side addition. Works both with scalars and expansions (element-wise) as well as
        with other maps.

        Args:
            other (Union[ComplexTaylorMap, ComplexMultivarTaylor, complex, float]): quantity to perfect subtraction on.

        Returns:
            ComplexTaylorMap: argument subtracted with Taylor map.

        """
        if isinstance(other, ComplexTaylorMap):
            map_coeff = -self._coeff + other._coeff
        elif isinstance(other, ComplexMultivarTaylor):
            map_coeff = -self._coeff + other.coeff
        else:
            # scalar case
            map_coeff = -self._coeff
            map_coeff[:, 0] += other
        return self.create_map_with_coeff(map_coeff)

    def __eq__(self, other) -> bool:
        """Method enabling comparison with other Taylor maps as well as vectors.

        Returns:
            (bool): True if the elements of the argument are all equal (pair-wise) with the ones of this map.

        """
        if self._len != len(other):
            return False

        # objects have same number of elements, so now compare them pair-wise
        for expans1, expans2 in zip(self, other):
            if expans1 != expans2:
                return False
        return True

    def pointwise_eval(self, x: np.ndarray) -> np.ndarray:
        """Method for the evaluation of the Taylor map on a given point.

        Args:
            x (numpy.ndarray): point of evaluation.

        Returns:
            (numpy.ndarray): Taylor map evaluated at given point.

        """
        if self.is_trivial():
            return self.const

        if self.is_univar():
            # Horner's scheme
            output = self._coeff[:, -1] * x + self._coeff[:, -2]
            for i in range(self.dim_alg - 3, -1, -1):
                output = output * x + self._coeff[:, i]
            return output

        # multivariate case with order at least one
        mapping = self[0].get_mapping_monom()
        products = np.ones(self.dim_alg, dtype=self._var_type)
        for exponent, index_coeff in mapping.items():
            for index_var, power_var in enumerate(exponent):
                if power_var > 0:
                    new_tuple = exponent[:index_var] + (power_var - 1,) + exponent[index_var + 1:]
                    products[index_coeff] = x[index_var] * products[mapping[new_tuple]]
                    break
        return self._coeff.dot(products)

    def massive_eval(self, Xs: np.ndarray) -> np.ndarray:
        """Method for the evaluation of the Taylor map on a range of points (vectorized evaluation).

        Args:
            Xs (numpy.ndarray): points of evaluation.

        Returns:
            (numpy.ndarray): Taylor map evaluated at given points.

        """
        if self.is_univar():
            return np.transpose(np.array([expansion.massive_eval(Xs) for expansion in self], dtype=self._var_type))
        if Xs.shape[1] != self.n_var and len(Xs.shape) == 2:
            raise IndexError("The number of columns of the input should equal the number of variables in the "
                             "expansion.")
        if self.is_trivial():
            return np.array([self.const] * Xs.shape[0], dtype=self._var_type)

        # multivariate case of order at least two
        products = np.ones((self.dim_alg, Xs.shape[0]), dtype=self._var_type)
        mapping = self[0].get_mapping_monom()
        for exponent, index_coeff in mapping.items():
            for index_var, power_var in enumerate(exponent):
                if power_var > 0:
                    new_tuple = exponent[:index_var] + (power_var - 1,) + exponent[index_var + 1:]
                    products[index_coeff, :] = Xs[:, index_var] * products[mapping[new_tuple], :]
                    break
        return np.transpose(self._coeff @ products)

    def __call__(self, *args, **kwargs):
        """Method for calling the Taylor expansion. Wraps several possibilities: evaluation and composition with a map.

        Returns:
            (Union[ComplexTaylorMap, numpy.ndarray]): Taylor map called on input.

        """
        other = args[0]
        return self.compose(other) if isinstance(other, self.__class__) else self.pointwise_eval(other)

    def compose(self, other: "ComplexTaylorMap") -> "ComplexTaylorMap":
        """Method performing composition with inputted Taylor map (must have the same order and as many elements as
        self has variables). The result is another map and has the same variables than the input.

        Args:
            other (ComplexTaylorMap): Taylor map to be composed on the right-hand side.

        Returns:
            ComplexTaylorMap: composed map.

        """

        if other.order != self._order or len(other) != self._n_var:
            raise ValueError("Inconsistent maps for composition")
        if not other.is_nilpotent():
            raise ValueError("Right-hand-side map must be nilpotent")
        if self.is_univar() and len(other) == 1:
            return self.__class__([self[0].compose(other[0])])
        if self.is_trivial():
            return other.create_const_expansion(self.const)

        # order of at least one
        rhs_coeff = np.zeros((self.dim_alg, other[0].dim_alg), dtype=self._var_type)
        products = [other[0].create_const_expansion(1.)]
        mapping = self[0].get_mapping_monom()
        for exponent, index_coeff in mapping.items():
            for index_var, power_var in enumerate(exponent):
                if power_var > 0:
                    new_tuple = exponent[:index_var] + (power_var - 1,) + exponent[index_var + 1:]
                    products.append(other[index_var] * products[mapping[new_tuple]])
                    rhs_coeff[index_coeff, :] = products[index_coeff].coeff
                    break
            else:  # no break
                rhs_coeff[0, 0] = 1.

        new_coeff = self._coeff @ rhs_coeff
        return self.__class__([other[0].create_expansion_with_coeff(new_coeff[i, :]) for i in range(0, self._len)])

    def _compo_inverse_lin_part(self) -> "ComplexTaylorMap":
        """Method returning the inverse of the linear part of the Taylor map from the point of view of composition. It
        boils down to inverting the square matrix made of all the corresponding coefficients.

        Returns:
            ComplexTaylorMap: inverse of linear part of Taylor expansion.

        """
        lin_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        lin_coeff[:, 1:self._len+1] = np.linalg.inv(self.jacobian)
        return self.create_map_with_coeff(lin_coeff)

    def compo_inverse(self, names_inverse: Optional[List[str]] = None) -> "ComplexTaylorMap":
        """Method returning the inverse of the Taylor map from the point of view of composition, i.e.
        so that composed with self it gives the identity map (X1, X2, ...). The computation is done iteratively,
        starting from the inversion of the first-order truncated map. See the work of M. Berz.

        Args:
            names_inverse (List[str]): name of inverse variables.

        Returns:
            ComplexTaylorMap: composition-inverse of Taylor expansion.

        """

        if not self.is_square_sized():
            raise ValueError("Right-hand-side size must equal left-hand-side\'s number of variables")
        if not self.is_nilpotent():
            raise ValueError("Map to be inverted cannot have non-zero constant terms")

        if self._len == 1:
            return self.__class__([self[0].compo_inverse(names_inverse)])

        if names_inverse is None:
            names_inverse = TaylorExpansAbstract.\
                get_default_var_names(self._n_var, default_inverse_name)

        inv_lin_map = self._compo_inverse_lin_part()
        inv_lin_map.var_names = names_inverse

        if self._order == 1:
            return inv_lin_map

        # order is at least two
        inter = -self.get_nonlin_part()
        ident = self.create_id_map()
        inverted_map = inv_lin_map.compose(inter.compose(inv_lin_map) + ident)
        for __ in range(2, self._order):
            inverted_map = inv_lin_map.compose(inter.compose(inverted_map) + ident)
        return inverted_map

    def create_id_map(self) -> "ComplexTaylorMap":
        """Method returning the so-called identity map of the algebra. If the variables' names are x1, ... xN, the i-th
        element of the map is xi + remainder.

        Returns:
            ComplexTaylorMap: identity map for this order and number of variables.

        """

        if not self.is_square_sized():
            raise ValueError("This map is not squared thus an identity map cannot be derived")

        id_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        id_coeff[:, 1:self._len+1] = np.eye(self._len)
        return self.create_map_with_coeff(id_coeff)

    @property
    def jacobian(self) -> np.ndarray:
        """Method returning the evaluation of the first-order derivatives w.r.t. all variables of all map's components.

        Returns:
            (numpy.ndarray): first-order derivatives in 2-D array form.

        """
        return np.array(self._coeff[:self.n_var, 1:self.n_var+1], dtype=self._var_type)

    def dot(self, other) -> ComplexMultivarTaylor:
        if not isinstance(other, self.__class__):
            new_coeff = other.dot(self._coeff)
            return self[0].create_expansion_with_coeff(new_coeff)
        return MapAbstract.dot(self, other)

    def divided_by_var(self, index_var: int) -> "ComplexTaylorMap":
        """Method returning the Taylor map divided by the input variable. This is not a rigorous operation as the
        order should be decreased by one.

        Args:
            index_var (str): index of variable to divide with.

        Returns:
            ComplexTaylorMap: Taylor expansion divided by variable.

        """

        return self.__class__([expansion.divided_by_var(index_var) for expansion in self])

    def var_eval(self, index_var: int, value: Scalar) -> "ComplexTaylorMap":
        """Method returning a Taylor map where a variable has been replaced by a fixed scalar value. In other
        words, it is a partial evaluation of the polynomial part. It is not rigorous as terms of higher order hidden in
        the remainder would need to be considered in this operation.

        Args:
            index_var (int): index of variable to be evaluated.
            value (Union[complex, float]): value to replace given variable.

        Returns:
            ComplexTaylorMap: Taylor map with removed dependency.

        """
        powers = np.cumprod(value * np.ones(self._order, dtype=self._var_type))
        new_coeff = np.zeros((self._len, self.dim_alg), dtype=self._var_type)
        mapping = self[0].get_mapping_monom()
        tuple_0 = (0,)
        for exponent, index_coeff in mapping.items():
            if exponent[index_var] == 0:
                new_coeff[:, index_coeff] += self._coeff[:, index_coeff]
            else:
                new_exponent = exponent[:index_var] + tuple_0 + exponent[index_var + 1:]
                new_coeff[:, mapping[new_exponent]] += self._coeff[:, index_coeff] * powers[exponent[index_var] - 1]
        return self.create_map_with_coeff(new_coeff)

    def contrib_removed(self, indices_var: List[int]) -> "ComplexTaylorMap":
        """Method returning a Taylor map where all coefficients associated to input variables' indices are set to
        zero.

        Args:
            indices_var (Iterable[int]): indices of variables whose contribution is to be removed.

        Returns:
            ComplexTaylorMap: Taylor map with removed contributions.

        """
        new_coeff = np.array(self._coeff, dtype=self._var_type)
        try:
            exponents = np.array(list(self[0].get_mapping_monom().keys()))[:, indices_var]
        except TypeError:
            raise ValueError("At least one inputted variable index is not an integer")
        except IndexError:
            raise ValueError("At least one inputted variable index does not exist in this algebra")
        for i in range(0, len(indices_var)):
            new_coeff[:, exponents[:, i] != 0] = 0.
        return self.create_map_with_coeff(new_coeff)

    def var_removed(self, index_var: int) -> "ComplexTaylorMap":
        """Method for the removal of a variable. Output lives in another algebra, of smaller dimension. All its
        terms associated with the old variables only are identical to original expansion.

        Args:
            index_var (int): index of variable to be removed.

        Returns:
            ComplexTaylorMap: Taylor map with a variable removed.

        """
        return self.__class__([expansion.var_removed(index_var) for expansion in self])

    def var_inserted(self, index_new_var: int, unknown_name: Optional[str] = None) -> "ComplexTaylorMap":
        """Method for the addition of a new variable. Output lives in another algebra, of higher dimension. All its
        terms associated with the new variable are zero and the other ones are identical to original expansion.

        Args:
            index_new_var (int): index of new variable to be added.
            unknown_name (str): name of new variable.

        Returns:
            ComplexTaylorMap: Taylor expansion with an additional variable.

        """
        return self.__class__([expansion.var_inserted(index_new_var, unknown_name) for expansion in self])

    def pow2(self) -> "ComplexTaylorMap":
        """Method to raise Taylor map to the power 2 i.e. compute its multiplicative square.

        Returns:
            ComplexTaylorMap: Taylor map raised to power 2.

        """
        return self.__class__([expansion.pow2() for expansion in self])

    @staticmethod
    @njit(cache=True)
    def mul_map_expans(coeff_map: np.ndarray, other_coeff: np.ndarray, square_indices: np.ndarray,
                       table_mul: np.ndarray, indices_mul: np.ndarray) -> np.ndarray:
        """Static method transforming series of coefficients of a map and a single Taylor expansion into the
        coefficients of the map from their multiplicative product. Method is static for just-in-time compiling
        with Numba.

        Args:
            coeff_map (numpy.ndarray): coefficients from the Taylor map.
            other_coeff (numpy.ndarray): coefficients of the Taylor expansion.
            square_indices (numpy.ndarray): precomputed indices corresponding to monomials which are the square
                of another monomial in the algebra.
            table_mul (numpy.ndarray): flattened algebra's multiplication table.
            indices_mul (numpy.ndarray): algebra's multiplication indices.

        Returns:
            (numpy.ndarray): coefficient corresponding to product.

        """
        multiplied_coeff = np.outer(coeff_map[:, 0], other_coeff) + other_coeff[0] * coeff_map
        dim_half_order = len(square_indices)
        symmetric_terms = coeff_map[:, :dim_half_order] * other_coeff[:dim_half_order]
        multiplied_coeff[:, 0] = 0.
        multiplied_coeff[:, square_indices] += symmetric_terms
        slices = indices_mul[2:] - indices_mul[1:-1]

        for i, (slice_index, el) in enumerate(zip(slices, other_coeff[2:]), 2):
            multiplied_coeff[:, table_mul[indices_mul[i - 1] + 1:indices_mul[i]]] += np.outer(coeff_map[:, i], other_coeff[1:slice_index]) \
                                                                                     + el * coeff_map[:, 1:slice_index]

        return multiplied_coeff

    def __mul__(self, other) -> "ComplexTaylorMap":
        """Method defining right-hand side multiplication. It works for the external multiplication i.e. with scalars
        and for the internal one with an expansion (element-wise too).

        Args:
            other (Union[ComplexMultivarTaylor, Iterable[ComplexMultivarTaylor, complex, float], complex, float]):
                quantity to be multiplied with.

        Returns:
            ComplexTaylorMap: multiplied objects.

        """

        if isinstance(other, ComplexMultivarTaylor):
            multiplied_coeff = self.mul_map_expans(self._coeff, other.coeff, self[0].get_square_indices(),
                                                   self[0].get_flat_table_mul(), self[0].get_indices_mul())
            return self.create_map_with_coeff(multiplied_coeff)
        if isinstance(other, (complex, float)):
            return self.create_map_with_coeff(self._coeff * other)
        try:
            if len(other) == self._len:  # component-wise multiplication

                if isinstance(other[0], ComplexMultivarTaylor):
                    return self.__class__([el1 * el2 for el1, el2 in zip(self, other)])

                return self.create_map_with_coeff(np.einsum("ji,j->ji", self._coeff, other))
        except TypeError:
            pass

        raise ValueError

    def create_map_from_smaller_algebra(self, other: "ComplexTaylorMap") -> "ComplexTaylorMap":
        """Method to create a Taylor map in same algebra than self, from a map with same size but in another algebra
        with same order but less variables. Names of intersecting variables must be identical otherwise the function
        does not work.

        Args:
            other (ComplexTaylorMap): Taylor expansion to extend in current algebra.

        Returns:
            ComplexTaylorMap: Taylor map whose polynomial coefficients are all zero except the ones related only to
                the variables of the input that are then identical to them.

        """

        if other.order != self._order or other.n_var >= self._n_var:
            raise ValueError("The inputted map has a different order.")

        expansion_coeff = other.coeff
        new_coeff = np.zeros((len(other), self._dim_alg), dtype=self._var_type)
        coeff_old_algebra = np.zeros(other.dim_alg, dtype=self._var_type)
        if other[0].n_var == 1:
            # the monomials-coefficients mapping is trivial in the univariate case (hence no dedicated function)
            old_mapping = {(j,): j for j in range(0, other[0].order + 1)}
        else:
            # multivariate case
            old_mapping = other[0].get_mapping_monom()

        for old_exponent, old_index_var in old_mapping.items():
            coeff_old_algebra[:old_index_var] = 0.
            coeff_old_algebra[old_index_var] = 1.
            old_monomial = other[0].create_expansion_with_coeff(coeff_old_algebra)
            str_old_monomial = str(old_monomial).split(landau_symbol)[0]
            coeff_new_algebra = np.zeros(self._dim_alg, dtype=self._var_type)
            for new_exponent, new_index_var in self[0].get_mapping_monom().items():
                coeff_new_algebra[:new_index_var] = 0.
                if sum(new_exponent) == sum(old_exponent):
                    coeff_new_algebra[new_index_var] = 1.
                    str_new_monomial = str(self[0].create_expansion_with_coeff(coeff_new_algebra)).split(landau_symbol)[0]
                    if str_new_monomial == str_old_monomial:
                        new_coeff[:, new_index_var] = expansion_coeff[:, old_index_var]
                        break
            else:  # no break
                raise ValueError

        return self.__class__([self[0].create_expansion_with_coeff(new_coeff[i, :]) for i in range(0, len(other))])

    def tan(self) -> "ComplexTaylorMap":
        return ComplexTaylorMap(np.tan(self))

    def tanh(self) -> "ComplexTaylorMap":
        return ComplexTaylorMap(np.tanh(self))


class RealTaylorMap(ComplexTaylorMap):
    """Class for Taylor maps of real variable(s).

    """

    def bounder(self, domains: Iterable[Interval]) -> List[Interval]:
        """Method to evaluate the polynomial part of the Taylor map on a Cartesian product of segments via
        interval arithmetic.

        Args:
            domains (Iterable[Interval]): input interval(s) for expansion's variables.

        Returns:
            List[Interval]: image of inputted interval(s) through Taylor map.

        """
        if self.is_trivial():
            return [Interval.singleton(self._coeff[i, 0]) for i in range(0, self._len)]

        # order of at least one
        if self.is_univar():
            array_intervals = np.array(domains, dtype=Interval)
            # Horner's evaluation is not performed on purpose with intervals
            output = self.const + self._coeff[:, 1] * array_intervals
            for i in range(2, self._order + 1):
                output += self._coeff[:, i] * array_intervals**i
        else:
            # multivariate case
            output = np.array([Interval.singleton(0.)] * self._len, dtype=Interval)
            powers = []
            for domain in domains:
                powers_xi = [1., domain]
                for j in range(2, self._order + 1):
                    powers_xi.append(domain ** j)
                powers.append(powers_xi)

            for exponent, index_coeff in self[0].get_mapping_monom().items():
                product = powers[0][exponent[0]]
                for index_var, power_var in enumerate(exponent[1:], 1):
                    product *= powers[index_var][power_var]
                output += self._coeff[:, index_coeff] * product

        return list(output)

    def cbrt(self) -> "RealTaylorMap":
        return RealTaylorMap(np.cbrt(self))

    def arcsin(self) -> "RealTaylorMap":
        return RealTaylorMap(np.arcsin(self))

    def arccos(self) -> "RealTaylorMap":
        return RealTaylorMap(np.arccos(self))

    def arctan(self) -> "RealTaylorMap":
        return RealTaylorMap(np.arctan(self))

    def arcsinh(self) -> "RealTaylorMap":
        return RealTaylorMap(np.arcsinh(self))

    def arccosh(self) -> "RealTaylorMap":
        return RealTaylorMap(np.arccosh(self))

    def arctanh(self) -> "RealTaylorMap":
        return RealTaylorMap(np.arctanh(self))
