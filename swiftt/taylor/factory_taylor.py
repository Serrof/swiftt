# factory_taylor.py: object factory for Taylor expansions
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

from typing import List, Optional, Union
import numpy as np
from swiftt.taylor.taylor_map import ComplexTaylorMap, RealTaylorMap
from swiftt.taylor.real_univar_taylor import RealMultivarTaylor, RealUnivarTaylor
from swiftt.taylor.complex_univar_taylor import ComplexMultivarTaylor, ComplexUnivarTaylor
from swiftt.taylor.taylor_expans_abstract import default_unknown_name, TaylorExpansAbstract


def const_expansion(n_var: int, order: int, const: Union[complex, float],
                    var_names: Optional[List[str]] = None) -> Union[ComplexMultivarTaylor, RealMultivarTaylor]:
    """Method to create a Taylor expansion with a given number of variables and order with all polynomial coefficients
    at zero except the zeroth-order one if needed. In other words, it writes as const + remainder.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.
        const (Union[complex, float]): coefficient of constant part.
        var_names (List[str]): name of variables.

    Returns:
        Union[ComplexMultivarTaylor, RealMultivarTaylor]: Taylor expansions built from inputs.

    """

    if var_names is None:
        var_names = TaylorExpansAbstract.get_default_var_names(n_var)

    if isinstance(const, complex):
        if n_var == 1:
            output = ComplexUnivarTaylor(order, var_names)
        else:
            # multivariate case
            output = ComplexMultivarTaylor(n_var, order, var_names)
        coeff = np.zeros(output.dim_alg, dtype=np.complex128)

    else:
        # real variable(s) case
        if n_var == 1:
            output = RealUnivarTaylor(order, var_names)
        else:
            # multivariate case
            output = RealMultivarTaylor(n_var, order, var_names)
        coeff = np.zeros(output.dim_alg, dtype=np.float64)

    coeff[0] = const
    output.coeff = coeff

    return output


def zero_expansion(n_var: int, order: int, var_names: Optional[List[str]] = None,
                   dtype: type = np.float64) -> Union[ComplexMultivarTaylor, RealMultivarTaylor]:
    """Method to create a Taylor expansion with a given number of variables and order with a null polynomial part.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.
        var_names (List[str]): name of variables.
        dtype (type): assumed numpy.dtype for coefficients and variables (float or complex).

    Returns:
        Union[ComplexMultivarTaylor, RealMultivarTaylor]: Taylor expansions whose polynomial part is null.

    """
    const = 0. if dtype == np.float64 else complex(0., 0.)
    return const_expansion(n_var, order, const, var_names)


def one_expansion(n_var: int, order: int, var_names: Optional[List[str]] = None,
                  dtype: type = np.float64) -> Union[ComplexMultivarTaylor, RealMultivarTaylor]:
    """Method to create a Taylor expansion with a given number of variables and order with all polynomial coefficients
    at zero except the zeroth-order one that is set to one.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.
        var_names (List[str]): name of variables.
        dtype (tyoe): assumed numpy.dtype for coefficients and variables (float or complex).

    Returns:
        Union[ComplexMultivarTaylor, RealMultivarTaylor]: Taylor expansions whose polynomial part is constant
        and equal to one.

    """
    if dtype == np.float64:
        return const_expansion(n_var, order, 1., var_names)

    # complex variable(s) case
    return const_expansion(n_var, order, complex(1., 0.), var_names)


def unknown_var(n_var: int, order: int, index_var: int, nominal: Union[complex, float], scaling: Union[complex, float],
                var_names: List[str]) -> Union[ComplexMultivarTaylor, RealMultivarTaylor]:
    """Method to create a Taylor expansion with a given number of variables and order whose polynomial part is affine
    (constant + linear) in a single variable and null for the others.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.
        index_var (int): index of variable with affine contribution.
        nominal (Union[complex, float]): value for constant part.
        scaling (Union[complex, float]): value for linear coefficient in variable of interest.
        var_names (List[str]): name of variables.

    Returns:
        Union[ComplexMultivarTaylor, RealMultivarTaylor]: Taylor expansion corresponding to inputs.

    """

    if index_var < 0 or index_var >= n_var:
        raise ValueError("Input index non-positive or larger than number of variables")

    if var_names is None:
        var_names = TaylorExpansAbstract.get_default_var_names(n_var)

    if not isinstance(nominal, complex) and not isinstance(scaling, complex):
        if n_var == 1:
            output = RealUnivarTaylor(order, var_names)
        else:
            # multivariable case
            output = RealMultivarTaylor(n_var, order, var_names)
        coeff = np.zeros(output.dim_alg, dtype=np.float64)
    else:
        # complex variable(s) case
        if n_var == 1:
            output = ComplexUnivarTaylor(order, var_names)
        else:
            # multivariable case
            output = ComplexMultivarTaylor(n_var, order, var_names)
        coeff = np.zeros(output.dim_alg, dtype=np.complex128)

    coeff[0], coeff[index_var + 1] = nominal, scaling
    output.coeff = coeff

    return output


def build_affine_square_map(order: int, consts, lins,
                            var_names: Optional[List[str]] = None) -> ComplexTaylorMap:
    """Method to create a square Taylor map with a given number of variables and order. Each element is affine
    (constant + linear) for a single variable.

    Args:
        order (int): order of expansion.
        consts (Iterable[complex, float]): values for constant part.
        lins (Iterable[complex, float]): value for linear coefficient. First one is for first element and first
        variable, second one for second element and second variable, etc.
        var_names (List[str]): name of variables.

    Returns:
        ComplexTaylorMap: Taylor map corresponding to inputs.

    """

    n_var = len(consts)
    if len(lins) != n_var:
        raise ValueError("Inconsistency between number of input constants and linear terms")
    is_floats = True
    for lin, const in zip(lins, consts):
        if isinstance(lin, complex) or isinstance(const, complex):
            is_floats = False
            break

    dtype = np.float64 if is_floats else np.complex128

    zero = zero_expansion(n_var, order, var_names, dtype)

    coeff = np.zeros(zero.dim_alg, dtype=dtype)

    expansion = zero.copy()
    coeff[0], coeff[1] = consts[0], lins[0]
    expansion.coeff = coeff

    expansions = [expansion]
    for i in range(1, len(consts)):
        coeff[0], coeff[i], coeff[i + 1] = consts[i], 0., lins[i]
        expansions.append(zero.create_expansion_with_coeff(coeff))

    return RealTaylorMap(expansions) if is_floats else ComplexTaylorMap(expansions)


def create_unknown_map(order: int, consts, var_names: Optional[List[str]] = None) -> ComplexTaylorMap:
    """Method to create a square Taylor map with a given number of variables and order with a constant polynomial part.
    Includes the trivial case of order zero.

    Args:
        order (int): order of expansion.
        consts (Iterable[complex, float]): values for constant part.
        var_names (List[str]): name of variables.

    Returns:
        ComplexTaylorMap: Taylor map corresponding to inputs.

    """

    n_var = len(consts)

    is_floats = True
    for const in consts:
        if isinstance(const, complex):
            is_floats = False
            break

    if order == 0:
        zero = zero_expansion(n_var, order, var_names)
        expansions = [zero.create_const_expansion(const) for const in consts]
        return RealTaylorMap(expansions) if is_floats else ComplexTaylorMap(expansions)

    if order > 0:
        if is_floats:
            return build_affine_square_map(order, consts, np.ones(n_var, dtype=np.float64), var_names)

        # complex variable(s) case
        return build_affine_square_map(order, consts, np.ones(n_var, dtype=np.complex128), var_names)

    raise ValueError("The inputted expansion's order is non-positive.")


def from_string(expr: str, order: Optional[int] = None, n_var: Optional[int] = None) -> RealMultivarTaylor:
    """Method to create a Taylor expansion from a string e.g. "x1**2 - x1 + x2 - 3". Uses the sympy library to
    interpret the polynomial part.

    Args:
        expr (str): string representing a polynomial to be converted into a Taylor expansion.
        order (int): order of the expansion (optional). If not present, taken as the degree of the polynomial.
        n_var (int): number of variables in the expansion (optional, if not present interpreted from input).
            Should not be less than the actual number of variables in the string.

    Returns:
        RealMultivarTaylor: Taylor expansions built from input(s).

    """

    try:
        from sympy import poly_from_expr  # here to avoid having it as a mandatory dependency
        poly, opt = poly_from_expr(expr)
        if order is None:
            order = sum(poly.LM("grlex").exponents)

        var_names = [str(el) for el in opt["gens"]]
        if n_var is None:
            n_var = len(opt["gens"])
        elif n_var < len(opt["gens"]):
            raise ValueError("The required number of variables cannot be less than in the inputted string.")
        else:
            # additional variables need to be named
            index = len(var_names) + 1
            while len(var_names) < n_var:
                tentative_name = default_unknown_name + str(index)
                if tentative_name in var_names:
                    index += 1
                else:
                    var_names.append(tentative_name)

        if n_var > 1:
            output = RealMultivarTaylor(len(opt["gens"]), order, var_names)
            coeff = np.zeros(output.dim_alg)
            mapping = output.get_mapping_monom()
            # loop over all non-zero coefficients for monomials with a degree not more than the expansion's order
            for el, monom in zip(poly.coeffs(), poly.monoms()):
                if sum(monom) <= order:
                    coeff[mapping[monom]] = el
        else:
            # univariate case
            output = RealUnivarTaylor(order, var_names)
            coeff = np.zeros(order + 1)
            inter = np.flip(poly.all_coeffs())  # sympy returns canonical coefficients in decreasing order
            index = min(len(coeff), len(inter))
            coeff[:index] = np.array(inter[:index])

        output.coeff = coeff
        return output

    except ImportError:
        raise ImportError("Sympy has not been found. Install it to be able to initialize Taylor expansions from "
                          "strings.")
