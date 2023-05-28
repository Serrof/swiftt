# taylor_expans_abstract.py: abstract class for Taylor expansions
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

from functools import lru_cache
from abc import ABCMeta, abstractmethod
from typing import List, Union, Iterable, Tuple, Callable, Optional
import numpy as np
from swiftt.algebraic_abstract import AlgebraicAbstract
from swiftt.taylor.tables import algebra_dim

TaylorOrScalar = Union["TaylorExpansAbstract", complex, float]
Scalar1OrND = Union[complex, float, Iterable[complex], Iterable[float]]

default_unknown_name = "x"
default_inverse_name = "y"
landau_symbol = "o"


class TaylorExpansAbstract(AlgebraicAbstract, metaclass=ABCMeta):
    """Abstract class for all objects of a Taylor differential algebra.

    Attributes:
        _n_var (int): number of variable(s) in algebra.
        _order (int): order of algebra.
        _dim_alg (int): dimension of algebra i.e. number of monomials in the polynomial part of an expansion.
        _var_type (type): type of variable(s).
        _var_names (List[str]): name of variable(s). They are here only for printing purposes: their consistency is
            *not* checked when performing operations on Taylor expansions, for example "x+o(x)"+"y+o(y)"="2x+o(x)".

    """

    # pre-declared intrinsic functions for scalars
    _exp_cst: Callable
    _log_cst: Callable
    _sqrt_cst: Callable
    _cos_cst: Callable
    _sin_cst: Callable
    _tan_cst: Callable
    _cosh_cst: Callable
    _sinh_cst: Callable
    _tanh_cst: Callable

    def __init__(self, n_var: int, order: int, var_type: type, var_names: List[str]) -> None:
        """Constructor for TaylorExpansAbstract class.

        Args:
             n_var (int): number of variable(s) in algebra.
             order (int): order of algebra.
             var_type (type): type of variable(s).
             var_names (List[str]): name of variable(s).

        """
        self._n_var = n_var
        self._order = order
        self._var_names = var_names
        self._var_type = var_type
        self._dim_alg = algebra_dim(n_var, order)
        self._coeff: np.ndarray = None

    @property
    def dim_alg(self) -> int:
        """Getter for dimension of algebra.

        Returns:
            int: dimension of algebra.

        """
        return self._dim_alg

    @property
    def order(self) -> int:
        """Getter for order of algebra.

        Returns:
            int: order of algebra.

        """
        return self._order

    @property
    def n_var(self) -> int:
        """Getter for number of variables in algebra.

        Returns:
            int: number of variables.

        """
        return self._n_var

    @property
    def var_type(self) -> type:
        """
        Getter for the type of variables.

        Returns:
            type: type of variables

        """
        return self._var_type

    @property
    def var_names(self) -> List[str]:
        """Getter for names of variables in algebra. They are implemented in a lazy fashion so if not defined before,
        the names are arbitrary.

        Returns:
            List[str]: names of variables.

        """
        if self._var_names is None:
            # if not provided by user upon initialization, call for default names
            self._var_names = TaylorExpansAbstract.get_default_var_names(self._n_var)
        if len(self._var_names) != self._n_var:
            raise ValueError("The stored names of variables do not match the number of variables.")
        if len(set(self._var_names)) != self._n_var:
            raise ValueError("At least two variables have exactly the same name.")
        return list(self._var_names)

    @var_names.setter
    def var_names(self, names: List[str]) -> None:
        """Setter for names of variables in algebra. It checks the number of variables so is not used in the constructor
        in order to preserve computational performance.

        Args:
            (List[str]): names of variables.

        """
        if len(names) != self._n_var:
            raise ValueError("The number of names does not match the number of variables.")
        if len(set(names)) != self._n_var:
            raise ValueError("The given names of the variables are not unique.")
        self._var_names = list(names)

    @property
    @abstractmethod
    def remainder_term(self) -> str:
        """Abstract getter for the remainder (for printing purposes).

        Returns:
            str: symbolic expression for the remainder.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def coeff(self) -> np.ndarray:
        """Getter for coefficients of polynomial part.

        Returns:
            np.ndarray: coefficients of Taylor expansion.

        """
        raise NotImplementedError

    @coeff.setter
    @abstractmethod
    def coeff(self, coefficients: np.ndarray) -> None:
        """Setter for coefficients of polynomial part.

        Args:
            coefficients (np.ndarray): new coefficients of Taylor expansion.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def const(self):
        """
        Getter for the so-called constant coefficient(s) i.e. associated to the zeroth-order contribution and to be
        implemented by all inheritors.

        """
        raise NotImplementedError

    @const.setter
    @abstractmethod
    def const(self, cst) -> None:
        """
        Setter for the so-called constant coefficient(s) i.e. associated to the zeroth-order contribution and to be
        implemented by all inheritors.

        """
        raise NotImplementedError

    def is_in_same_algebra(self, other: "TaylorExpansAbstract") -> bool:
        """Checks if input has same order and number of variables.

        Args:
             other (TaylorExpansAbstract): Taylor expansion used to compare algebras.

        Returns:
            bool: True if and only if the two objects have the same order and number of variables.

        """
        return self._order == other.order and self._n_var == other.n_var

    def is_trivial(self) -> bool:
        """Function to call to know if expansion is in an algebra of order zero.

        Returns:
            bool: true if expansion is at order zero, false otherwise.

        """
        return self._order == 0

    def get_const_part(self) -> "TaylorExpansAbstract":
        """Abstract method returning the constant part (zeroth-order) and to be implemented by all inheritors.

        Returns:
            TaylorExpansAbstract: constant part of Taylor expansion.

        """
        return self.create_const_expansion(self.const)

    @abstractmethod
    def get_linear_part(self) -> "TaylorExpansAbstract":
        """Abstract method returning the linear part and to be implemented by all inheritors.

        Returns:
            TaylorExpansAbstract: linear part of Taylor expansion.

        """
        raise NotImplementedError

    def get_affine_part(self) -> "TaylorExpansAbstract":
        """Method returning the affine part.

        Returns:
            TaylorExpansAbstract: affine part of Taylor expansion.

        """
        try:
            return self.get_low_order_part(1)
        except ValueError:
            raise ValueError("There is no affine part in a zeroth-order expansion.")

    def get_nonlin_part(self) -> "TaylorExpansAbstract":
        """Method returning the non-linear part.

        Returns:
            TaylorExpansAbstract: non-linear part of Taylor expansion.

        """
        try:
            return self.get_high_order_part(2)
        except ValueError:
            raise ValueError("There is no non-linear part in an expansion below order two.")

    @abstractmethod
    def get_nilpo_part(self) -> "TaylorExpansAbstract":
        """Abstract method returning the nilpotent part and to be implemented by all inheritors. The nilpotent part (of
        the polynomial part in a Taylor expansion) is equal to the original expansion, except for the constant term that
        is set to zero.

        Returns:
            TaylorExpansAbstract: nilpotent part of Taylor expansion i.e. without the zeroth-order contribution.

        """
        raise NotImplementedError

    @abstractmethod
    def create_const_expansion(self, const: Scalar1OrND) -> "TaylorExpansAbstract":
        raise NotImplementedError

    @abstractmethod
    def is_nilpotent(self) -> bool:
        """Abstract method returning true if the Taylor expansions is nilpotent and to be implemented by all inheritors.
        A nilpotent expansion has a vanishing polynomial part if raised to a power greater than the order.

        Returns:
            bool: true if constant coefficient(s) is (are) zero.

        """
        raise NotImplementedError

    def __pow__(self, alpha: Union[int, float], modulo: Optional[float] = None) -> "TaylorExpansAbstract":
        """Method raising Taylor expansion objects to power. For integer, it wraps with the multiplication and squaring.
        For non-integer exponent, it is seen as a composition with x -> x ** alpha

        Args:
            alpha (Union[int, float]): power exponent.
            modulo (None): not used for Taylor expansions.

        Returns:
            TaylorExpansAbstract: Taylor expansion object to input power.

        """
        if self.is_trivial():
            return self.create_const_expansion(self.const ** alpha)

        # order of at least one
        is_int = isinstance(alpha, int)
        if is_int:
            if alpha < 0.:
                return self._pown(-alpha).reciprocal()
            if alpha > 1:
                return self._pown(alpha)
            if alpha == 1:
                return self.copy()
            return self.create_const_expansion(1.)  # alpha = 0
        if isinstance(alpha, float):
            # compose with x -> x ** alpha
            const = self.const
            nilpo = self.get_nilpo_part() * (1. / const)
            terms = np.ones(self._order + 1)
            terms[1:] = np.cumprod([(alpha - i) / (i + 1) for i in range(0, self._order)])
            powered = terms[-1] * nilpo
            powered.const = terms[-2]
            for el in terms[-3::-1]:
                powered *= nilpo
                powered.const = el
            return powered * (const**alpha)
        raise ValueError("Wrong exponent")

    def reciprocal(self) -> "TaylorExpansAbstract":
        """Method defining the reciprocal a.k.a. multiplicative inverse (within the algebra) of a Taylor expansion.
        It is computed as the composition from the left by x -> 1 / x

        Returns:
            TaylorExpansAbstract: multiplicative inverse of Taylor expansion object.

        """
        if self.is_trivial():
            return self.create_const_expansion(1. / self.const)

        # order of at least one
        const_inv = 1. / self.const
        nilpo = self.get_nilpo_part() * (-const_inv)
        inverted = nilpo + 1.
        for __ in range(1, self._order):
            inverted = nilpo * inverted + 1.
        return inverted * const_inv

    @abstractmethod
    def rigorous_integ_once_wrt_var(self, index_var: int) -> "TaylorExpansAbstract":
        """Method performing integration with respect to a given unknown variable. The integration constant is zero.
        This transformation is rigorous as the order of the expansion is increased by one. In other words, the output
        lives in another algebra, of higher dimension.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            TaylorExpansAbstract: integrated Taylor expansion object w.r.t. input variable number.

        """
        raise NotImplementedError

    def integ_once_wrt_var(self, index_var: int) -> "TaylorExpansAbstract":
        """Method performing integration with respect to a given unknown variable while remaining in the same
        algebra and to be implemented by all inheritors. This transformation is not rigorous in the sense that the order
        of the expansion should be increased by one.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.

        Returns:
            TaylorExpansAbstract: integrated Taylor expansion w.r.t. input variable number.

        """
        return self.rigorous_integ_once_wrt_var(index_var).truncated(self._order)

    def integ_wrt_var(self, index_var: int, integ_order: int) -> "TaylorExpansAbstract":
        """Method performing integration an arbitrary number of times with respect to a given unknown variable while
        remaining in the same algebra. This transformation is not rigorous in the sense that the order of the expansion
        should be increased at each integration.

        Args:
            index_var (int): variable number w.r.t. which integration needs to be performed.
            integ_order (int): number of time integration needs to be performed.

        Returns:
            TaylorExpansAbstract: integrated Taylor expansion w.r.t. input variable number.

        """
        if integ_order == 1:
            return self.integ_once_wrt_var(index_var)

        # recursive call with lowered integration order
        return self.integ_wrt_var(index_var, integ_order - 1).integ_once_wrt_var(index_var)

    @abstractmethod
    def deriv_once_wrt_var(self, index_var: int) -> "TaylorExpansAbstract":
        """Abstract method performing differentiation with respect to a given unknown variable while remaining in the
        same algebra and to be implemented by all inheritors. This transformation is not rigorous in the sense that the
        order of the expansion should be decreased by one.

        Args:
            index_var (int): variable number w.r.t. which differentiation needs to be performed.

        Returns:
            TaylorExpansAbstract: differentiated Taylor expansion w.r.t. input variable number.

        """
        raise NotImplementedError

    def deriv_wrt_var(self, index_var: int, diff_order: int) -> "TaylorExpansAbstract":
        """Method performing differentiation an arbitrary number of times with respect to a given unknown variable while
        remaining in the same algebra. This transformation is not rigorous in the sense that the order of the expansion
        should be decreased at each integration.

        Args:
            index_var (int): variable number w.r.t. which differentiation needs to be performed.
            diff_order (int): number of time differentiation needs to be performed.

        Returns:
            TaylorExpansAbstract: differentiated Taylor expansion w.r.t. input variable number.

        """
        if diff_order == 1:
            return self.deriv_once_wrt_var(index_var)

        # recursive call with lowered differentiation order
        return self.deriv_wrt_var(index_var, diff_order - 1).deriv_once_wrt_var(index_var)

    @abstractmethod
    def compose(self, other) -> "TaylorExpansAbstract":
        """Abstract method to be implemented by all inheritors. Performs composition with inputted Taylor expansion
        (must have the same order). The result has the same variables than the input.

        Args:
            other (TaylorExpansAbstract): argument to be composed on the right-hand side.

        Returns:
            TaylorExpansAbstract: composed expansion.

        """
        raise NotImplementedError

    @abstractmethod
    def pointwise_eval(self, x):
        """Abstract method for the evaluation of the Taylor expansion and to be implemented by all inheritors.

        Args:
            x (Union[complex, float, numpy.ndarray]): point of evaluation.

        Returns:
            Union[complex, float, numpy.ndarray]: Taylor expansion evaluated at given point.

        """
        raise NotImplementedError

    @abstractmethod
    def get_low_order_part(self, order: int) -> "TaylorExpansAbstract":
        """Abstract method to be implemented by all inheritors. It removes the high order terms of the expansion whilst
        leaving the order unchanged. Hence it is not a rigorous operation.

        Args:
            order (int): order (included) above which all contributions are removed.

        Returns:
            TaylorExpansAbstract: low order part of the Taylor expansion.

        """
        raise NotImplementedError

    @abstractmethod
    def truncated(self, new_order: int) -> "TaylorExpansAbstract":
        """Abstract method for the truncation at a given order and to be implemented by all inheritors. Output lives
        in another algebra, of lower dimension.

        Args:
            new_order (int): order of algebra in which to truncate the Taylor expansion.

        Returns:
            TaylorExpansAbstract: Taylor expansion truncated at input order.

        """
        raise NotImplementedError

    @abstractmethod
    def get_high_order_part(self, order: int) -> "TaylorExpansAbstract":
        """Abstract method to be implemented by all inheritors. It removes the low order terms of the expansion whilst
        leaving the order unchanged. Hence it is not a rigorous operation.

        Args:
            order (int): order (included) below which all contributions are removed.

        Returns:
            TaylorExpansAbstract: high order part of the Taylor expansion.

        """
        raise NotImplementedError

    @abstractmethod
    def prolong(self, new_order: int) -> "TaylorExpansAbstract":
        """Abstract method for the prolongation (opposite of truncation) at a given order and to be implemented by all
        inheritors. Output lives in another algebra, of higher dimension. It is not a rigorous operation as the possible
        contributions within the former remainder are ignored.

        Args:
            new_order (int): order of algebra in which to extend the Taylor expansion.

        Returns:
            TaylorExpansAbstract: Taylor expansion prolonged at input order.

        """
        raise NotImplementedError

    def prolong_one_order(self) -> "TaylorExpansAbstract":
        """Method for the prolongation (opposite of truncation) of exactly one order. Output lives in another algebra,
        of higher dimension. It is not a rigorous operation as the possible contributions within the former remainder
        are ignored.

        Returns:
            TaylorExpansAbstract: Taylor expansion prolonged of one order.

        """
        return self.prolong(self._order + 1)

    @abstractmethod
    def var_inserted(self, index_new_var: int, unknown_name: Optional[str] = None) -> "TaylorExpansAbstract":
        """Abstract method for the addition of a new variable and to be implemented by all inheritors. Output lives in
        another algebra, of higher dimension. All its terms associated with the new variable are zero and the other ones
        are identical to original expansion.

        Args:
            index_new_var (int): index of new variable to be added.
            unknown_name (str): name of new variable.

        Returns:
            TaylorExpansAbstract: Taylor expansion with an additional variable.

        """
        raise NotImplementedError

    def var_appended(self, unknown_name: Optional[str] = None) -> "TaylorExpansAbstract":
        """Wrapper to add a new variable at the end (stack).

        Args:
            unknown_name (str): name of new variable.

        Returns:
            TaylorExpansAbstract: Taylor expansion with an additional variable.

        """
        return self.var_inserted(self.n_var, unknown_name)

    @abstractmethod
    def var_removed(self, index_var: int) -> "TaylorExpansAbstract":
        """Abstract method for the removal of a variable. Output lives in another algebra, of smaller dimension. All its
        terms associated with the old variables only are identical to original expansion.

        Args:
            index_var (int): index of variable to be removed.

        Returns:
            TaylorExpansAbstract: Taylor expansion object with a variable removed.

        """
        raise NotImplementedError

    def last_var_removed(self) -> "TaylorExpansAbstract":
        """Wrapper to remove the last variable (unstack).

        Returns:
            TaylorExpansAbstract: Taylor expansion object with the last variable removed.

        """
        return self.var_removed(self.n_var - 1)

    @abstractmethod
    def var_eval(self, index_var: int, value: Union[complex, float]) -> "TaylorExpansAbstract":
        """Abstract method returning a Taylor expansion object where a variable has been replaced by a fixed scalar
        value. In other words, it is a partial evaluation of the polynomial part. It is not rigorous as terms of higher
        order hidden in the remainder would need to be considered in this operation.

        Args:
            index_var (int): index of variable to be evaluated.
            value (Union[complex, float]): value to replace given variable.

        Returns:
            TaylorExpansAbstract: Taylor expansion object with removed dependency.

        """
        raise NotImplementedError

    def last_var_eval(self, value: Union[complex, float]) -> "TaylorExpansAbstract":
        """Method returning a Taylor expansion object where the last variable has been replaced by a fixed scalar value.
        It wraps the general method for any variable.

        Args:
            value (Union[complex, float]): value to replace last variable with.

        Returns:
            TaylorExpansAbstract: Taylor expansion object with removed dependency.

        """
        return self.var_eval(self._n_var - 1, value)

    def contrib_removed(self, indices_var: List[int]) -> "TaylorExpansAbstract":
        """Method returning a Taylor expansion object where all coefficients associated to input variables' indices are
        set to zero.

        Args:
            indices_var (List[int]): indices of variables whose contribution is to be removed.

        Returns:
            TaylorExpansAbstract: Taylor expansion object with removed contributions.

        """
        output = self.copy()
        for index_var in indices_var:
            output = output.var_eval(index_var, 0.)
        return output

    def last_contrib_removed(self) -> "TaylorExpansAbstract":
        """Method returning a Taylor expansion object where all coefficients associated to last variable are set to
        zero.

        Returns:
            TaylorExpansAbstract: Taylor expansion object with contributions from last variable removed.

        """
        return self.contrib_removed([self._n_var - 1])

    @abstractmethod
    def divided_by_var(self, index_var: int) -> "TaylorExpansAbstract":
        """Abstract method returning the Taylor expansion divided by the input variable. This is not a rigorous
        operation as the order should be decreased by one.

        Args:
            index_var (int): index of variable to divide with.

        Returns:
            TaylorExpansAbstract: Taylor expansion divided by variable.

        """
        raise NotImplementedError

    def linearly_combine_with_another(self, alpha: Union[complex, float], expansion: "TaylorExpansAbstract",
                                      beta: Union[complex, float]) -> "TaylorExpansAbstract":
        """Method multiplying with a scalar and then adding with a another expansion also multiplied by some scalar.

        Args:
            alpha (Union[complex, float]): multiplier for self.
            expansion (TaylorExpansAbstract): expansion to be linearly combined with.
            beta (Union[complex, float]): multiplier for other expansion.

        Returns:
            TaylorExpansAbstract: linear combination of self and arguments.

        """
        return alpha * self + beta * expansion

    @staticmethod
    def landau_univar(order: int, var_names: List[str]) -> str:
        """Static method returning a string representing with a Landau notation the remainder of the expansion in the
        univariate case.

        Args:
            order (int): order of the expansion.
            var_names (List[str]): name of the variable.

        Returns:
            str: negligible part of the Taylor expansion w.r.t. input order.

        """
        if order == 0:
            return landau_symbol + "(1)"
        if order == 1:
            return landau_symbol + "(" + var_names[0] + ")"
        # order of at least two
        return landau_symbol + "(" + var_names[0] + "**" + str(order) + ")"

    @staticmethod
    def landau_multivar(order: int, var_names: List[str]) -> str:
        """Static method returning a string representing with a Landau notation the remainder of the expansion in the
        multivariate case.

        Args:
            order (int): order of the expansion.
            var_names (List[str]): name of variables.

        Returns:
            str: negligible part of the Taylor expansion w.r.t. input order.

        """
        if order == 0:
            return landau_symbol + "(1)"

        if len(var_names) == 2:
            norm = "|(" + var_names[0] + ", " + var_names[1] + ")|"
        else:
            # at least three variables
            norm = "|(" + var_names[0] + ",...," + var_names[-1] + ")|"

        if order == 1:
            return landau_symbol + "(" + norm + ")"
        # order of at least two
        return landau_symbol + "(" + norm + "**" + str(order) + ")"

    @staticmethod
    def get_default_var_names(n_var: int, unknown_name: str = default_unknown_name) -> List[str]:
        """Static method returning the default name for the variables of a Taylor expansion.

        Args:
            n_var (int): number of variables.
            unknown_name (str): character referring to variable of expansion.

        Returns:
            List[str]: name of variables.

        """
        return [unknown_name] if n_var == 1 else [unknown_name + str(i) for i in range(1, n_var + 1)]

    def exp(self) -> "TaylorExpansAbstract":
        """Exponential of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: exponential of Taylor expansion object.

        """
        if self.is_trivial():
            return self.create_const_expansion(self._exp_cst(self.const))

        nilpo = self.get_nilpo_part()
        order_plus_one = self.order + 1
        seq = np.ones(order_plus_one)
        seq[1:] /= np.cumprod(np.arange(1, order_plus_one))
        expon = nilpo * seq[-1]
        expon.const = seq[-2]
        for el in seq[-3::-1]:
            expon *= nilpo
            expon.const = el
        return expon * self._exp_cst(self.const)

    def log(self) -> "TaylorExpansAbstract":
        """Natural logarithm of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: natural logarithm of Taylor expansion object.

        """
        if self.is_trivial():
            return self.create_const_expansion(self._log_cst(self.const))

        order = self.order
        const = self.const
        nilpo = self.get_nilpo_part() * (1. / const)
        seq = np.append([0.], 1. / np.arange(1., order + 1))
        seq[2:] *= np.cumprod(-np.ones(order - 1))
        logar = nilpo * seq[-1]
        for el in seq[-2:0:-1]:
            logar.const = el
            logar *= nilpo
        logar.const = self._log_cst(const)
        return logar

    def sqrt(self) -> "TaylorExpansAbstract":
        """Square root of Taylor expansion.

        Returns:
            TaylorExpansAbstract: square root of Taylor expansion.

        """
        if self.is_trivial():
            return self.create_const_expansion(self._sqrt_cst(self.const))

        const = self.const
        nilpo = self.get_nilpo_part() * (1. / const)
        order = self.order
        terms = np.append([1.], (1. / 2. + np.arange(0., -order, -1.)) / np.arange(1., order + 1))
        terms = np.cumprod(terms)
        powered = nilpo * terms[-1]
        powered.const = terms[-2]
        for el in terms[-3::-1]:
            powered *= nilpo
            powered.const = el
        return powered * self._sqrt_cst(const)

    @staticmethod
    def seq_c_s_zero(order: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        order_plus_one = order + 1
        c_0, s_0 = np.zeros(order_plus_one), np.zeros(order_plus_one)
        c_0[0] = s_0[1] = 1.
        integers = np.arange(1, order_plus_one)
        factors = eps / (integers[:-1] * integers[1:])
        for i in range(1, order - 1, 2):
            s_0[i + 2] = s_0[i] * factors[i]
        for i in range(0, order - 1, 2):
            c_0[i + 2] = c_0[i] * factors[i]
        return c_0, s_0

    @staticmethod
    @lru_cache(maxsize=1)
    def seq_cos_sin_zero(order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the truncated Maclaurin series of the trigonometric cosine and sine functions.

        Args:
             order (int): order of truncation.

        Returns:
            np.ndarray: truncated Maclaurin series of cosine.
            np.ndarray: truncated Maclaurin series of sine.

        """
        return TaylorExpansAbstract.seq_c_s_zero(order, -1.)

    @staticmethod
    @lru_cache(maxsize=1)
    def seq_cosh_sinh_zero(order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the truncated Maclaurin series of the hyperbolic cosine and sine functions.

        Args:
             order (int): order of truncation.

        Returns:
            numpy.ndarray: truncated Maclaurin series of hyperbolic cosine.
            numpy.ndarray: truncated Maclaurin series of hyperbolic sine.

        """
        return TaylorExpansAbstract.seq_c_s_zero(order, 1.)

    @staticmethod
    def sinusoids(p: "TaylorExpansAbstract", eps: float) -> Tuple["TaylorExpansAbstract", "TaylorExpansAbstract"]:
        order = p.order
        nilpo = p.get_nilpo_part()
        seq_cos, seq_sin = TaylorExpansAbstract.seq_c_s_zero(order, eps)
        cosinus = nilpo * seq_cos[-1]
        cosinus.const = seq_cos[-2]
        sinus = nilpo * seq_sin[-1]
        sinus.const = seq_sin[-2]
        for el1, el2 in zip(seq_cos[-3::-1], seq_sin[-3::-1]):
            cosinus *= nilpo
            cosinus.const = el1
            sinus *= nilpo
            sinus.const = el2
        return cosinus, sinus

    @staticmethod
    def _c_s(p: "TaylorExpansAbstract", eps: float,
             a: Union[complex, float], b: Union[complex, float]) -> "TaylorExpansAbstract":
        cosine_sine = TaylorExpansAbstract.sinusoids(p, eps)
        cosine, sine = cosine_sine[0], cosine_sine[1]
        return a * cosine + b * sine

    def cos(self) -> "TaylorExpansAbstract":
        """Cosine of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: cosine of Taylor expansion object.

        """
        const = self.const
        c, s = self._cos_cst(const), self._sin_cst(const)
        return self.create_const_expansion(c) if self.is_trivial() else self._c_s(self, -1., c, -s)

    def sin(self) -> "TaylorExpansAbstract":
        """Sine of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: sine of Taylor expansion object.

        """
        const = self.const
        c, s = self._cos_cst(const), self._sin_cst(const)
        return self.create_const_expansion(s) if self.is_trivial() else self._c_s(self, -1., s, c)

    def cosh(self) -> "TaylorExpansAbstract":
        """Hyperbolic cosine of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: hyperbolic cosine of Taylor expansion object.

        """
        const = self.const
        ch, sh = self._cosh_cst(const), self._sinh_cst(const)
        return self.create_const_expansion(ch) if self.is_trivial() else self._c_s(self, 1., ch, sh)

    def sinh(self) -> "TaylorExpansAbstract":
        """Hyperbolic sine of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: hyperbolic sine of Taylor expansion object.

        """
        const = self.const
        ch, sh = self._cosh_cst(const), self._sinh_cst(const)
        return self.create_const_expansion(ch) if self.is_trivial() else self._c_s(self, 1., sh, ch)

    @staticmethod
    def seq_tan_tanh(c: Union[complex, float], eps: float, order: int, tan_cst: Callable,
                     tanh_cst: Callable) -> np.ndarray:
        order_plus_one = order + 1
        seq = np.empty(order_plus_one, dtype=c.__class__)
        seq[0] = tanh_cst(c) if eps == -1. else tan_cst(c)
        seq[1] = 1. + eps * seq[0] * seq[0]
        for i in range(2, order_plus_one):
            summed = seq[:i].dot(seq[i - 1::-1])
            seq[i] = summed * eps / i
        return seq

    def tan(self) -> "TaylorExpansAbstract":
        """Tangent of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: tangent of Taylor expansion object.

        """
        if self.is_trivial():
            return self.create_const_expansion(self._tan_cst(self.const))

        nilpo = self.get_nilpo_part()
        seq = TaylorExpansAbstract.seq_tan_tanh(self.const, 1., self.order,
                                                self._tan_cst, self._tanh_cst)
        tangent = nilpo * seq[-1]
        tangent.const = seq[-2]
        for el in seq[-3::-1]:
            tangent *= nilpo
            tangent.const = el
        return tangent

    def tanh(self) -> "TaylorExpansAbstract":
        """Hyperbolic tangent of Taylor expansion object.

        Returns:
            TaylorExpansAbstract: hyperbolic tangent of Taylor expansion object.

        """
        if self.is_trivial():
            return self.create_const_expansion(self._tanh_cst(self.const))

        nilpo = self.get_nilpo_part()
        seq = self.seq_tan_tanh(self.const, -1., self.order, self._tan_cst, self._tanh_cst)
        tangenth = nilpo * seq[-1]
        tangenth.const = seq[-2]
        for el in seq[-3::-1]:
            tangenth *= nilpo
            tangenth.const = el
        return tangenth
