# map_abstract.py: abstract class for algebraic maps
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

from abc import ABCMeta
import numpy as np
from swiftt.algebraic_abstract import AlgebraicAbstract


class MapAbstract(AlgebraicAbstract, metaclass=ABCMeta):
    """Abstract class for "maps" i.e. vectors of a given algebraic type.

    Attributes:
        _len (int): length.
        _items (numpy.array): components.

    """

    def __init__(self, elements) -> None:
        self._len = len(elements)
        self._items = np.array(elements, dtype=elements[0].__class__)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, item: int):
        return self._items[item]

    def __setitem__(self, item: int, value) -> None:
        self._items[item] = value

    def copy(self) -> "MapAbstract":
        return self.__class__(self._items)

    def __add__(self, other) -> "MapAbstract":
        return self.__class__(self._items + other)

    def __sub__(self, other) -> "MapAbstract":
        return self.__class__(self._items - other)

    def __rsub__(self, other) -> "MapAbstract":
        return self.__class__(-self._items + other)

    def __mul__(self, other) -> "MapAbstract":
        return self.__class__(self._items * other)

    def __truediv__(self, other) -> "MapAbstract":
        return self.__class__(self._items / other)

    def __rtruediv__(self, other) -> "MapAbstract":
        return self.__class__(other / self._items)

    def __pow__(self, power, modulo=None) -> "MapAbstract":
        return self.__class__([el.__pow__(power, modulo) for el in self])

    def __str__(self) -> str:
        string = str(self[0])
        for el in self[1:]:
            string += ", " + str(el)
        return "( " + string + " )"

    def sqrt(self) -> "MapAbstract":
        return self.__class__([el.sqrt() for el in self])

    def exp(self) -> "MapAbstract":
        return self.__class__([el.exp() for el in self])

    def log(self) -> "MapAbstract":
        return self.__class__([el.log() for el in self])

    def dot(self, other):
        return sum(el1 * el2 for el1, el2 in zip(self, other))
