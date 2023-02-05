# tables.py: collection of methods to pre-computes tables for a given Taylor algebra
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
from typing import Dict, Tuple, List
import numpy as np

_lru_cache_maxsize = None


@lru_cache(maxsize=_lru_cache_maxsize)
def factorials(up_to: int) -> np.ndarray:
    """Memoized method to compute factorials.

    Args:
        up_to (int): maximum order of factorials.

    Returns:
        numpy.ndarray: all factorials up to inputted order.

    """
    if up_to <= 1:
        return np.ones(up_to + 1, dtype=int)

    inter = factorials(up_to - 1)  # recursive call
    return np.append(inter, [up_to * inter[-1]])


@lru_cache(maxsize=_lru_cache_maxsize)
def algebra_dim(n_var: int, order: int) -> int:
    """Memoized method to compute the dimension of a given algebra i.e. the number of coefficients in the polynomial
    part.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        int: dimension of Taylor algebra.

    """

    if order > n_var:
        return algebra_dim(order, n_var)  # use symmetry
    if order == 0:
        return 1

    # recursive call with smaller order
    return int(algebra_dim(n_var, order - 1) * (n_var + order) / order)


@lru_cache(maxsize=_lru_cache_maxsize)
def mapping_monom(n_var: int, order: int) -> Dict[Tuple[int, ...], int]:
    """Memoized method to build mapping between the indices of the polynomial coefficients and the monomials.
    It assumes that the initial order of a dictionary is kept, which is only the case for Python 3.6 or newer.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        Dict[Tuple[int, ...], int]: mapping where keys are the monomials (tuple of order for each variable)
            and values are the polynomial coefficients' indices.

    Raises:
        ValueError: if called for univariate expansions as it is not needed in that case.

    """

    if n_var == 1:
        raise ValueError("Such dictionary is not helpful with univariate expansions.")

    if n_var == 2:
        mapping = {(0, 0): 0}
        for i in range(1, order + 1):
            for j in range(0, i + 1):
                mapping[(i - j, j)] = len(mapping)
        # keys i.e. monomials are already sorted

    else:
        # at least three variables
        mapping = {}

        # recursive call with one less variable
        mapping_less_variables = mapping_monom(n_var - 1, order)
        monomial_orders = np.sum(np.array(list(mapping_less_variables.keys())), axis=1)

        # complete existing exponents with exponent in last variable
        for exponent, index_coeff in mapping_less_variables.items():
            for i in range(0, order + 1 - monomial_orders[index_coeff]):
                mapping[exponent + (i,)] = len(mapping)

        # sort monomials in graded lexicographic order a.k.a. grlex (that is first in increasing total degree,
        # then in decreasing degree in x1, then in x2, etc.)
        sorted_monomials = sorted(mapping.keys(), key=lambda x: (-sum(x), *x), reverse=True)
        mapping = {key: value for value, key in enumerate(sorted_monomials)}

    return mapping


@lru_cache(maxsize=_lru_cache_maxsize)
def deriv_table(n_var: int, order: int) -> np.ndarray:
    """Memoized method to build differentiation/integration table for given order and number of variables.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        numpy.ndarray: differentiation/integration table.

    Raises:
        ValueError: if called for univariate expansions as it is not needed in that case.

    """

    if n_var == 1:
        raise ValueError("Such table is not helpful with univariate expansions.")

    mapping_indices = mapping_monom(n_var, order)
    dim = len(mapping_indices)
    table_deriv = np.full((dim, n_var), dim, dtype=int)

    for i, exponent in enumerate(list(mapping_indices.keys())[:algebra_dim(n_var, order - 1)]):
        for index_var, power_var in enumerate(exponent):
            new_tuple = exponent[:index_var] + (power_var + 1,) + exponent[index_var + 1:]
            table_deriv[i, index_var] = mapping_indices[new_tuple]

    return table_deriv


@lru_cache(maxsize=_lru_cache_maxsize)
def mul_table(n_var: int, order: int) -> List[List[int]]:
    """Memoized method to build multiplication table for given order and number of variables. On a given row,
    corresponding to a monomial M as ordered in the mapping with the coefficients, the elements are the indices of the
    monomials obtained when multiplying M with all the monomials before M in the map. There is no row for monomials
    x_i**order as it is known that only the monomial 1 can be multiplied with them.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        List[List[int]]: multiplication table.

    Raises:
        ValueError: if called for univariate expansions as it is not needed in that case.

    """

    if n_var == 1:
        raise ValueError("Such table is not helpful with univariate expansions.")

    table = [[0]]

    if order > 0:

        mapping_indices = mapping_monom(n_var, order)
        monomial_list = list(mapping_indices.keys())
        monomial_orders = np.sum(np.array(monomial_list), axis=1)

        for i, exponent_i in enumerate(monomial_list[1:algebra_dim(n_var, order - 1)], 1):
            table_row = []
            orders_i_plus_j = monomial_orders[:i] + monomial_orders[i]
            for j, exponent_j in enumerate(monomial_list[:i]):  # use symmetry between i and j
                if orders_i_plus_j[j] <= order:
                    summed_expo = tuple(index_i + index_j for index_i, index_j in zip(exponent_i, exponent_j))
                    table_row.append(mapping_indices[summed_expo])
                else:
                    break
            table.append(table_row)

    return table


@lru_cache(maxsize=_lru_cache_maxsize)
def flat_mul_table(n_var: int, order: int) -> np.ndarray:
    """Memoized method to flattened algebra's multiplication table.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        numpy.ndarray: flattened multiplication table.

    Raises:
        ValueError: if called for univariate expansions as it is not needed in that case.

    """
    return np.hstack(mul_table(n_var, order))


@lru_cache(maxsize=_lru_cache_maxsize)
def mul_indices(n_var: int, order: int) -> np.ndarray:
    """Memoized method to precompute array indices for multiplication.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        numpy.ndarray: array indices used in multiplication.

    """

    indices = [len(row) for row in mul_table(n_var, order)]

    return np.cumsum(indices)


@lru_cache(maxsize=_lru_cache_maxsize)
def square_indices(n_var: int, order: int) -> np.ndarray:
    """Memoized method to precompute square indices i.e. indices of monomials that are the square of another one in the
    algebra. Those are not already contained in the result of mul_table to save memory.

    Args:
        n_var (int): number of variables.
        order (int): order of expansion.

    Returns:
        numpy.ndarray: so-called square indices i.e. indices of monomials which are the square of a monomial of the
            algebra.

    """

    dim_half_order = algebra_dim(n_var, order // 2)
    indices = np.zeros(dim_half_order, dtype=int)
    mapping_indices = mapping_monom(n_var, order)
    for i, exponent_i in enumerate(list(mapping_indices.keys())[1:dim_half_order], 1):
        indices[i] = mapping_indices[tuple(2 * el for el in exponent_i)]

    return indices
