# interval.py: class implementing interval arithmetic
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

import math
from typing import Union, Optional, Callable
import numpy as np
from swiftt.algebraic_abstract import AlgebraicAbstract

IntervalOrScalar = Union[float, "Interval"]


class Interval(AlgebraicAbstract):
    """Class representing intervals of real numbers.

    Attributes:
        _lb (float): lower bound.
        _ub (float): upper bound.

    """

    def __init__(self, lb: float, ub: float) -> None:
        self._lb = lb  # no call to property on purpose as no sanity check is needed
        self.ub = ub

    def __len__(self) -> float:
        return self._ub - self._lb

    @property
    def ub(self) -> float:
        return self._ub

    @ub.setter
    def ub(self, ub: float) -> None:
        if ub < self.lb:
            raise ValueError("The upper bound cannot be strictly less than the lower one.")
        self._ub = ub

    @property
    def lb(self) -> float:
        return self._lb

    @lb.setter
    def lb(self, lb: float) -> None:
        if lb > self.ub:
            raise ValueError("The lower bound cannot be strictly greater than the upper one.")
        self._lb = lb

    def copy(self) -> "Interval":
        return Interval(self._lb, self._ub)

    @staticmethod
    def singleton(point: float) -> "Interval":
        return Interval(point, point)

    def __add__(self, other: IntervalOrScalar) -> "Interval":
        if isinstance(other, Interval):
            return Interval(self._lb + other.lb, self._ub + other.ub)
        # scalar case
        return Interval(self._lb + other, self._ub + other)

    def contains(self, other: IntervalOrScalar) -> bool:
        if isinstance(other, Interval):
            return other.lb >= self._lb and other.ub <= self._ub
        # scalar case
        return self._ub >= other >= self._lb

    def contains_zero(self) -> bool:
        return self.contains(0.)

    def __neg__(self) -> "Interval":
        return Interval(-self._ub, -self._lb)

    def __sub__(self, other: IntervalOrScalar) -> "Interval":
        if isinstance(other, Interval):
            return self + other.__neg__()
        # scalar case
        return Interval(self._lb - other, self._ub - other)

    def __mul__(self, other: IntervalOrScalar) -> "Interval":
        if isinstance(other, Interval):
            candidates = np.array([self._lb * other.lb, self._ub * other.lb, self._ub * other.ub,
                                   self._lb * other.ub])
            return Interval(np.min(candidates), np.max(candidates))
        # scalar case
        return self * Interval.singleton(other)

    def reciprocal(self) -> "Interval":
        if self.contains_zero():
            return Interval(-np.inf, np.inf)

        if self._ub != 0. and self._lb != 0.:
            inter = np.sort(1. / np.array([self._lb, self._ub]))
            return Interval(inter[0], inter[1])

        if self._ub != 0.:
            return Interval(1. / self._ub, np.inf)

        return Interval(-np.inf, 1. / self._lb)

    def __pow__(self, power: Union[int, float], modulo: Optional[float] = None) -> "Interval":
        if isinstance(power, int):
            if int(power / 2) == power / 2.:
                if self._lb >= 0:
                    return Interval(self._lb**power, self._ub**power)
                if self._ub < 0.:
                    return Interval(self._ub**power, self._lb**power)
                return Interval(0., max(self._lb**power, self._ub**power))
            return Interval(self._lb**power, self._ub**power)
        raise NotImplementedError

    def __str__(self) -> str:
        return "[" + str(self._lb) + ", " + str(self._ub) + "]"

    def __eq__(self, other: IntervalOrScalar) -> bool:
        if isinstance(other, Interval):
            return self._ub == other.ub and self._lb == other.lb
        # scalar case
        return self == self.singleton(other)

    def __abs__(self) -> "Interval":
        fabs = np.fabs([self._lb, self._ub])
        if self.contains_zero():
            return Interval(0., np.max(fabs))
        return Interval(np.min(fabs), np.max(fabs))

    def increasing_intrinsic(self, func: Callable) -> "Interval":
        return Interval(func(self.lb), func(self.ub))

    def cbrt(self) -> "Interval":
        return self.increasing_intrinsic(math.cbrt)

    def sqrt(self) -> "Interval":
        return self.increasing_intrinsic(math.sqrt)

    def exp(self) -> "Interval":
        return self.increasing_intrinsic(math.exp)

    def log(self) -> "Interval":
        return self.increasing_intrinsic(math.log)
