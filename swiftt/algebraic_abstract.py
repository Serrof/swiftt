# algebraic_abstract.py: abstract class for algebraic objects
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

from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Iterable
import math

AlgebraicOrScalar = Union["AlgebraicAbstract", complex, float, Iterable["AlgebraicAbstract"], Iterable[complex],
                          Iterable[float]]


class AlgebraicAbstract(metaclass=ABCMeta):
    """Abstract class for all objects of an algebra.

    """

    # constants for angular conversions
    _rad2deg_float = math.degrees(1.)
    _deg2rad_float = math.radians(1.)

    @abstractmethod
    def __add__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Abstract method defining right-hand side addition to be implemented by all inheritors.

        Args:
            other (AlgebraicOrScalar): object to be added.

        Returns:
            AlgebraicAbstract: summed objects.

        """
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Abstract method defining right-hand side subtraction to be implemented by all inheritors.

        Args:
            other (AlgebraicOrScalar): object to be subtracted.

        Returns:
            AlgebraicAbstract: subtracted objects.

        """
        raise NotImplementedError

    def __radd__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Method defining left-hand side addition from right-hand side.

        Args:
            other (AlgebraicOrScalar): object to be added.

        Returns:
            AlgebraicAbstract: summed objects.

        """
        return self.__add__(other)

    def __rsub__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Method defining left-hand side subtraction.

        Args:
            other (AlgebraicOrScalar): object to perform subtraction on.

        Returns:
            AlgebraicAbstract: subtracted objects.

        """
        return self.__neg__().__add__(other)

    @abstractmethod
    def __mul__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Abstract method defining right-hand side multiplication to be implemented by all inheritors. It must work
        for the external multiplication i.e. with scalars and for the internal one, that is between expansions.

        Args:
            other (AlgebraicOrScalar): object to be multiplied with.

        Returns:
            AlgebraicAbstract: multiplied objects.

        """
        raise NotImplementedError

    def __rmul__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Method defining left-hand side multiplication from right-hand side one.

        Args:
            other (AlgebraicOrScalar): object to be multiplied with.

        Returns:
            AlgebraicAbstract: multiplied objects.

        """
        return self.__mul__(other)

    def square(self) -> "AlgebraicAbstract":
        return self * self

    def __neg__(self) -> "AlgebraicAbstract":
        """Method defining negation (additive inverse).

        Returns:
            AlgebraicAbstract: opposite of object (from the point of view of addition).

        """
        return self.__mul__(-1.)

    def pow2(self) -> "AlgebraicAbstract":
        """Method to raise algebraic objet to the power 2 i.e. compute its multiplicative square.

        Returns:
            AlgebraicAbstract: algebraic object raised to power 2.

        """
        return self * self

    def _pown(self, power: int) -> "AlgebraicAbstract":
        """Method to raise algebraic object at a power than is a natural integer greater or equal to one.

        Args:
            power (int): exponent.

        Returns:
            AlgebraicAbstract: algebraic object raised to input power.

        """
        if power == 2:
            return self.pow2()
        if power == 3:
            return self.pow2() * self

        if power > 3:
            half_power = int(power / 2)
            if float(half_power) == power / 2.:
                return self._pown(half_power).pow2()  # recursive call

            # exponent is odd
            return self._pown(half_power).pow2() * self  # recursive call

        return self.copy()  # power = 1

    @abstractmethod
    def __pow__(self, power: Union[float, int], modulo: Optional[float] = None) -> "AlgebraicAbstract":
        """Abstract method defining power rising to be implemented by all inheritors.

        Args:
            power (Union[float, int]): exponent.

        Returns:
            AlgebraicAbstract: powered object.

        """
        raise NotImplementedError

    def __truediv__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Method defining left-hand side division to be implemented by all inheritors.

        Args:
            other (AlgebraicOrScalar): object to divide with.

        Returns:
            AlgebraicAbstract: object divided by input.

        """
        if isinstance(other, self.__class__):
            return self * other.reciprocal()

        # scalar case
        return self * (1. / other)

    def __rtruediv__(self, other: AlgebraicOrScalar) -> "AlgebraicAbstract":
        """Method defining right-hand side division to be implemented by all inheritors.

        Args:
            other (AlgebraicOrScalar): object divided.

        Returns:
            AlgebraicAbstract: input divided by object.

        """
        return other * self.reciprocal()

    def reciprocal(self) -> "AlgebraicAbstract":
        """Method defining the reciprocal a.k.a. multiplicative inverse (within the algebra).

        Returns:
            AlgebraicAbstract: multiplicative inverse of object.

        """
        raise self.__pow__(-1)

    def degrees(self) -> "AlgebraicAbstract":
        """
        Method to convert from radians to degrees.

        Returns:
             AlgebraicAbstract: object converted into degrees (assuming radians originally)
        """
        return self * self._rad2deg_float

    def radians(self) -> "AlgebraicAbstract":
        """
        Method to convert from degrees to radians.

        Returns:
             AlgebraicAbstract: object converted into radians (assuming degrees originally)
        """
        return self * self._deg2rad_float

    def deg2rad(self) -> "AlgebraicAbstract":
        return self.radians()

    def rad2deg(self) -> "AlgebraicAbstract":
        return self.degrees()

    @abstractmethod
    def __str__(self) -> str:
        """Abstract method to cast object as a string.

        Returns:
            str: string representing the object.

        """
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> "AlgebraicAbstract":
        """Abstract method defining copy to be implemented by all inheritors.

        Returns:
            AlgebraicAbstract: copied object.

        """
        return self.__class__()

    @abstractmethod
    def log(self) -> "AlgebraicAbstract":
        """Napierian logarithm of algebraic object.

        Returns:
            AlgebraicAbstract: Napierian logarithm of algebraic object.

        """
        raise NotImplementedError

    @abstractmethod
    def exp(self) -> "AlgebraicAbstract":
        """Exponential of algebraic object.

        Returns:
            AlgebraicAbstract: exponential of algebraic object.

        """
        raise NotImplementedError

    def sqrt(self) -> "AlgebraicAbstract":
        """Square root of algebraic object.

        Returns:
            AlgebraicAbstract: square root of algebraic object.

        """
        return (self.log() / 2.).exp()
