# real_univar_taylor.py: class implementing Taylor expansions of a unique real variable
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

from typing import List, Optional
from swiftt.taylor.real_multivar_taylor import RealMultivarTaylor
from swiftt.taylor.complex_univar_taylor import ComplexUnivarTaylor
from swiftt.interval import Interval
from swiftt.taylor.taylor_expans_abstract import default_unknown_name


class RealUnivarTaylor(ComplexUnivarTaylor, RealMultivarTaylor):
    """Class for Taylor expansions of a single real variable.

    """

    def __init__(self, order: int, var_name: List[str] = default_unknown_name) -> None:
        """Constructor for Taylor expansions of a single real variable. Result has no assigned coefficients.

        Args:
            order (int): order of expansion.
            var_name (List[str]): name of variable.

        """
        RealMultivarTaylor.__init__(self, 1, order, var_name)

    def __str__(self) -> str:
        """Method to cast Taylor expansion as a string. Overwrites parent implementation assuming more than one
        variable.

        Returns:
            str: string representing the Taylor expansion.

        """
        string = str(self.const)
        if not self.is_trivial():
            var_name = self._var_names[0]
            if self._coeff[1] != 0.:
                sign = " + " if self._coeff[1] > 0. else " - "
                string += sign + str(abs(self._coeff[1])) + " * " + var_name
            for j, el in enumerate(self._coeff[2:], 2):
                if el != 0.:
                    sign = " + " if el > 0. else " - "
                    string += sign + str(abs(el)) + " * " + var_name + "**" + str(j)
        return string + " + " + self.remainder_term

    def bounder(self, domain: Interval) -> Interval:
        """Method to evaluate the polynomial part of the Taylor expansion on a segment via interval arithmetic.

        Args:
            domain (Interval): input interval for expansion's variable.

        Returns:
            Interval: image of inputted interval through polynomial part.

        """
        if self.is_trivial():
            return Interval.singleton(self.const)

        # Horner's evaluation is not performed on purpose with intervals
        output = self.const + self._coeff[1] * domain
        for i, el in enumerate(self._coeff[2:], 2):
            output += el * domain**i
        return output

    def var_inserted(self, index_new_var: int, unknown_name: Optional[str] = None) -> RealMultivarTaylor:
        """Method for the addition of a new variable. Output lives in another algebra, of higher dimension. All its
        terms associated with the new variable are zero and the other ones are identical to original expansion.
        Overwrites the parent implementation to make sure the correct object is created (real expansion rather than
        complex).

        Args:
            index_new_var (int): index of new variable to be added.
            unknown_name (str): name of new variable.

        Returns:
            RealMultivarTaylor: Taylor expansion with an additional variable.

        """
        complex_expansion = ComplexUnivarTaylor.var_inserted(self, index_new_var, unknown_name)
        return RealMultivarTaylor.from_complex_expansion(complex_expansion)
