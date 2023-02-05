# math_algebra.py: collection of intrinsic functions for algebraic objects
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

from typing import Callable, Union
import math
from math import pi
import cmath
import numpy as np
from swiftt.taylor.real_multivar_taylor import RealMultivarTaylor
from swiftt.taylor.taylor_map import RealTaylorMap
from swiftt.taylor.taylor_expans_abstract import AlgebraicAbstract, TaylorExpansAbstract
from swiftt.taylor.complex_univar_taylor import ComplexUnivarTaylor


def taylor_real(func_taylor: Callable):
    def taylor_real_decorator(func_real: Callable):
        def function_wrapper(*args, **kwargs):
            p = args[0]
            if isinstance(p, RealMultivarTaylor):
                if p.is_trivial():
                    return p.create_const_expansion(func_real(p.const))
                return func_taylor(p)
            return func_real(*args, **kwargs)
        function_wrapper.__name__ = func_real.__name__
        return function_wrapper
    return taylor_real_decorator


def variance_from_mean(mean_func):
    def variance_wrapper(*args, **kwargs):
        mean_p = mean_func(*args, **kwargs)
        p, *rest = args
        # increase the maximum order by doubling it so that terms are not truncated later
        centered_p = p.prolong(2 * p.order) - mean_p
        return mean_func(centered_p ** 2, *rest, **kwargs)
    return variance_wrapper


def covariance_from_mean(mean_func):
    def covariance_wrapper(*args, **kwargs):
        p, q, *rest = args
        mean_p = mean_func(p, *rest, **kwargs)
        mean_q = mean_func(q, *rest, **kwargs)
        # increase the maximum order by doubling it so that terms are not truncated later
        centered_p = p.prolong(2 * p.order) - mean_p
        centered_q = q.prolong(2 * q.order) - mean_q
        return mean_func(centered_p * centered_q, *rest, **kwargs)
    return covariance_wrapper


def scalar_inversion(func: Callable[[ComplexUnivarTaylor], ComplexUnivarTaylor],
                     p: ComplexUnivarTaylor) -> ComplexUnivarTaylor:
    """Method returning the (composition) inverse of the image by the input 1-D function of the input expansion. If the
    latter is the identity, the result is the expansion of the function's inverse, a.k.a. the truncated reverted series.

    Args:
        func (Callable[[ComplexUnivarTaylor]): univariate function.
        p (ComplexUnivarTaylor): Taylor expansion to be fed to function.

    Returns:
        ComplexUnivarTaylor: inverse of image expansion.

    """
    coeff = np.zeros(p.order + 1)
    coeff[0], coeff[1] = p.const, 1.
    univar_expansion = p.create_expansion_with_coeff(coeff)
    output = func(univar_expansion).get_nilpo_part().compo_inverse().compose(p.get_nilpo_part())
    output.const = coeff[0]
    return output


def sqrt(p: Union[AlgebraicAbstract, complex, float]) -> Union[AlgebraicAbstract, complex, float]:
    """Taylor expansio- and interval--compatible version of the square root function.
    Wraps the math implementation for floats.

    Args:
         p (Union[AlgebraicAbstract, complex, float]): object whose square root needs to be computed.

    Returns:
        Union[AlgebraicAbstract, complex, float]: square root of input.

    """
    if isinstance(p, AlgebraicAbstract):
        return p.sqrt()
    if isinstance(p, (float, int)):
        return math.sqrt(p)
    if isinstance(p, complex):
        return cmath.sqrt(p)
    raise NotImplementedError


def cbrt(p: Union[RealMultivarTaylor, float]) -> Union[RealMultivarTaylor, float]:
    """Taylor expansion- and interval-compatible version of the cubic root function.
    Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, float]): object whose cubic root needs to be computed.

    Returns:
        Union[RealMultivarTaylor, float]: cubic root of input.

    """
    if isinstance(p, RealMultivarTaylor):
        return p.cbrt()
    if isinstance(p, (float, int)):
        return np.cbrt([p])[0]
    raise NotImplementedError


def exp(p: Union[AlgebraicAbstract, complex, float]) -> Union[AlgebraicAbstract, complex, float]:
    """Taylor expansion- and interval-compatible version of the exponential function.
    Wraps the math implementation for complex and floats.

    Args:
         p (Union[AlgebraicAbstract, complex, float]): object whose exponential needs to be computed.

    Returns:
        Union[AlgebraicAbstract, complex, float]: exponential of input.

    """
    if isinstance(p, AlgebraicAbstract):
        return p.exp()
    if isinstance(p, (float, int)):
        return math.exp(p)
    if isinstance(p, complex):
        return cmath.exp(p)
    raise NotImplementedError


def log(p: Union[AlgebraicAbstract, complex, float]) -> Union[AlgebraicAbstract, complex, float]:
    """Taylor expansion- and interval-compatible version of the natural logarithm function.
    Wraps the math implementation for complex and floats.

    Args:
         p (Union[AlgebraicAbstract, complex, float]): object whose natural logarithm needs to be computed.

    Returns:
        Union[AlgebraicAbstract, complex, float]: natural logarithm of input.

    """
    if isinstance(p, AlgebraicAbstract):
        return p.log()
    if isinstance(p, (float, int)):
        return math.log(p)
    if isinstance(p, complex):
        return cmath.log(p)
    raise NotImplementedError


def log10(p: Union[AlgebraicAbstract, complex, float]) -> Union[AlgebraicAbstract, complex, float]:
    return log(p) * (1. / math.log(10.))


def log2(p: Union[AlgebraicAbstract, complex, float]) -> Union[AlgebraicAbstract, complex, float]:
    return log(p) * (1. / math.log(2.))


def cos(p: Union[TaylorExpansAbstract, complex, float]) -> Union[TaylorExpansAbstract, complex, float]:
    """Taylor expansion-compatible version of the cosine function. Wraps the math implementation for complex and floats.

    Args:
         p (Union[TaylorExpansAbstract, complex, float]): object whose cosine needs to be computed.

    Returns:
        Union[TaylorExpansAbstract, complex, float]: cosine of input.

    """
    if isinstance(p, TaylorExpansAbstract):
        return p.cos()
    if isinstance(p, (float, int)):
        return math.cos(p)
    if isinstance(p, complex):
        return cmath.cos(p)
    raise NotImplementedError


def sin(p: Union[TaylorExpansAbstract, complex, float]) -> Union[TaylorExpansAbstract, complex, float]:
    """Taylor expansion-compatible version of the sine function. Wraps the math implementation for complex and floats.

    Args:
         p (Union[TaylorExpansAbstract, complex, float]): object whose sine needs to be computed.

    Returns:
        Union[TaylorExpansAbstract, complex, float]: sine of input.

    """
    if isinstance(p, TaylorExpansAbstract):
        return p.sin()
    if isinstance(p, (float, int)):
        return math.sin(p)
    if isinstance(p, complex):
        return cmath.sin(p)
    raise NotImplementedError


def cosh(p: Union[TaylorExpansAbstract, complex, float]) -> Union[TaylorExpansAbstract, complex, float]:
    """Taylor expansion-compatible version of the hyperbolic cosine function.
    Wraps the math implementation for complex and floats.

    Args:
         p (Union[TaylorExpansAbstract, complex, float]): object whose hyperbolic cosine needs to be computed.

    Returns:
        Union[TaylorExpansAbstract, complex, float]: hyperbolic cosine of input.

    """
    if isinstance(p, TaylorExpansAbstract):
        return p.cosh()
    if isinstance(p, (float, int)):
        return math.cosh(p)
    if isinstance(p, complex):
        return cmath.cosh(p)
    raise NotImplementedError


def sinh(p: Union[TaylorExpansAbstract, complex, float]) -> Union[TaylorExpansAbstract, complex, float]:
    """Taylor expansion-compatible version of the hyperbolic sine function.
    Wraps the math implementation for complex and floats.

    Args:
         p (Union[TaylorExpansAbstract, complex, float]): object whose hyperbolic sine needs to be computed.

    Returns:
        Union[TaylorExpansAbstract, complex, float]: hyperbolic sine of input.

    """
    if isinstance(p, TaylorExpansAbstract):
        return p.sinh()
    if isinstance(p, (float, int)):
        return math.sinh(p)
    if isinstance(p, complex):
        return cmath.sinh(p)
    raise NotImplementedError


def tan(p: Union[TaylorExpansAbstract, complex, float]) -> Union[TaylorExpansAbstract, complex, float]:
    """Taylor expansion-compatible version of the tangent function.
    Wraps the math implementation for complex and floats.

    Args:
         p (Union[TaylorExpansAbstract, complex, float]): object whose tangent needs to be computed.

    Returns:
        Union[TaylorExpansAbstract, complex, float]: tangent of input.

    """
    if isinstance(p, TaylorExpansAbstract):
        return p.tan()
    if isinstance(p, (float, int)):
        return math.tan(p)
    if isinstance(p, complex):
        return cmath.tan(p)
    raise NotImplementedError


def tanh(p: Union[TaylorExpansAbstract, complex, float]) -> Union[TaylorExpansAbstract, complex, float]:
    """Taylor expansion-compatible version of the hyperbolic tangent function.
    Wraps the math implementation for complex and floats.

    Args:
         p (Union[TaylorExpansAbstract, complex, float]): object whose hyperbolic tangent needs to be computed.

    Returns:
        Union[TaylorExpansAbstract, complex, float]: hyperbolic tangent of input.

    """
    if isinstance(p, TaylorExpansAbstract):
        return p.tanh()
    if isinstance(p, (float, int)):
        return math.tanh(p)
    if isinstance(p, complex):
        return cmath.tanh(p)
    raise NotImplementedError


def atan(p: Union[RealMultivarTaylor, RealTaylorMap, float]) -> Union[RealMultivarTaylor, RealTaylorMap, float]:
    """Taylor expansion-compatible version of the inverse tangent function. Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, RealTaylorMap, float]): object whose inverse tangent needs to be computed.

    Returns:
        Union[RealMultivarTaylor, RealTaylorMap, float]: inverse tangent of input.

    """
    if isinstance(p, (RealMultivarTaylor, RealTaylorMap)):
        return p.arctan()
    if isinstance(p, (float, int)):
        return math.atan(p)
    raise NotImplementedError


def atanh(p: Union[RealMultivarTaylor, RealTaylorMap, float]) -> Union[RealMultivarTaylor, RealTaylorMap, float]:
    """Taylor expansion-compatible version of the inverse hyperbolic tangent function.
    Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, RealTaylorMap, float]): object whose inverse hyperbolic tangent needs to
            be computed.

    Returns:
        Union[RealMultivarTaylor, RealTaylorMap, float]: inverse hyperbolic tangent of input.

    """
    if isinstance(p, (RealMultivarTaylor, RealTaylorMap)):
        return p.arctanh()
    if isinstance(p, (float, int)):
        return math.atanh(p)
    raise NotImplementedError


def acos(p: Union[RealMultivarTaylor, RealTaylorMap, float]) -> Union[RealMultivarTaylor, RealTaylorMap, float]:
    """Taylor expansion-compatible version of the inverse cosine function. Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, RealTaylorMap, float]): object whose inverse cosine needs to be computed.

    Returns:
        Union[RealMultivarTaylor, RealTaylorMap, float]: inverse cosine of input.

    """
    if isinstance(p, (RealMultivarTaylor, RealTaylorMap)):
        return p.arccos()
    if isinstance(p, (float, int)):
        return math.acos(p)
    raise NotImplementedError


def asin(p: Union[RealMultivarTaylor, RealTaylorMap, float]) -> Union[RealMultivarTaylor, RealTaylorMap, float]:
    """Taylor expansion-compatible version of the inverse sine function. Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, RealTaylorMap, float]): object whose inverse sine needs to be computed.

    Returns:
        Union[RealMultivarTaylor, RealTaylorMap, float]: inverse sine of input.

    """
    if isinstance(p, (RealMultivarTaylor, RealTaylorMap)):
        return p.arcsin()
    if isinstance(p, (float, int)):
        return math.asin(p)
    raise NotImplementedError


def acosh(p: Union[RealMultivarTaylor, RealTaylorMap, float]) -> Union[RealMultivarTaylor, RealTaylorMap, float]:
    """Taylor expansion-compatible version of the inverse hyperbolic cosine function.
    Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, RealTaylorMap, float]): object whose inverse hyperbolic cosine needs
            to be computed.

    Returns:
        Union[RealMultivarTaylor, RealTaylorMap, float]: inverse hyperbolic cosine of input.

    """
    if isinstance(p, (RealMultivarTaylor, RealTaylorMap)):
        return p.arccosh()
    if isinstance(p, (float, int)):
        return math.acosh(p)
    raise NotImplementedError


def asinh(p: Union[RealMultivarTaylor, RealTaylorMap, float]) -> Union[RealMultivarTaylor, RealTaylorMap, float]:
    """Taylor expansion-compatible version of the inverse hyperbolic sine function.
    Wraps the math implementation for floats.

    Args:
         p (Union[RealMultivarTaylor, RealTaylorMap, float]): object whose inverse hyperbolic sine needs to be computed.

    Returns:
        Union[RealMultivarTaylor, RealTaylorMap, float]: inverse hyperbolic sine of input.

    """
    if isinstance(p, (RealMultivarTaylor, RealTaylorMap)):
        return p.arcsinh()
    if isinstance(p, (float, int)):
        return math.asinh(p)
    raise NotImplementedError


def erf_taylor(p: RealMultivarTaylor) -> RealMultivarTaylor:
    """Version for Taylor expansions of the error function. Assumes the order is at least one.

    Args:
         p (RealMultivarTaylor): real Taylor expansion.

    Returns:
        RealMultivarTaylor: error function of input Taylor expansion.

    """
    order = p.order
    const = p.const
    seq = np.zeros(order)
    seq[0] = 2. * math.exp(-const**2) / math.sqrt(pi)
    if order > 1:
        seq[1] = -2. * seq[0] * const
        for i in range(2, order):
            seq[i] = -2. * (seq[i - 2] + const * seq[i - 1]) / i
    seq /= np.arange(1, order + 1)
    nilpo = p.get_nilpo_part()
    errorfun = seq[-1] * nilpo
    for el in seq[-2::-1]:
        errorfun.const = el
        errorfun *= nilpo
    errorfun.const = math.erf(const)
    return errorfun


def erfc_taylor(p: RealMultivarTaylor) -> RealMultivarTaylor:
    """Version for Taylor expansions of the complementary error function. Assumes the order is at least one.

    Args:
         p (RealMultivarTaylor): real Taylor expansion.

    Returns:
        RealMultivarTaylor: complementary error function of input Taylor expansion.

    """
    return 1. - erf_taylor(p)


def atan2(q: Union[RealMultivarTaylor, float], p: Union[RealMultivarTaylor, float]) -> Union[RealMultivarTaylor, float]:
    """Taylor expansion-compatible version of arctan 2. Wraps the math implementation for floats.

    Args:
         q (Union[RealMultivarTaylor, float]): real expansion.
         p (Union[RealMultivarTaylor, float]): real expansion.

    Returns:
        Union[RealMultivarTaylor, float]: arctan2 of (q, p).

    """
    if isinstance(q, RealMultivarTaylor):
        return q.arctan2(p)
    if isinstance(p, RealMultivarTaylor):
        return p.create_const_expansion(float(q)).arctan2(p)
    return math.atan2(q, p)


def mean_from_normal(p: RealMultivarTaylor, mus, cov_mat) -> float:
    """ Assuming the variables of a Taylor expansion form a Gaussian random vector, compute its analytical mean. It uses
    the moment generating function of the normal law to obtain all the moments involved.

    Args:
          p (RealMultivarTaylor): Taylor expansion.
          mus (Iterable[float]): mean values of Gaussian vector.
          cov_mat(np.array): covariance matrix of random vector.

    Returns:
         float: mean value of the Taylor expansion obtained by evaluating its variables as the components of the
            inputted normal law.

    """

    # sanity checks
    if p.is_trivial():
        return p.const
    if len(mus) != p.n_var:
        raise ValueError("The inputted mean values do not have same dimension as number of variables in expansion.")
    if cov_mat.shape[0] != len(mus) or cov_mat.shape[1] != len(mus):
        raise ValueError("The inputted covariance matrix has wrong shape.")

    # build a new map of unknowns
    coeff = np.zeros(p.dim_alg)
    ts = []
    for i in range(0, p.n_var):
        coeff[i], coeff[i + 1] = 0., 1.
        ts.append(p.create_expansion_with_coeff(coeff))

    # compute moment generating function in algebra
    half_variances = np.diag(cov_mat) / 2.
    rhs = (mus[0] + half_variances[0] * ts[0]) * ts[0]
    for i, (mu, t, half_var) in enumerate(zip(mus[1:], ts[1:], half_variances[1:]), 1):
        inter = t.linearly_combine_with_many(half_var, ts[:i], cov_mat[i, :i])
        rhs += (mu + inter) * t
    moment_generating_func = exp(rhs)

    # compute mean by linearity i.e. adding all the contributions (coefficient times mean of monomial)
    mean = p.const
    coeff = p.coeff
    mapping = dict(ts[0].get_mapping_monom())
    del mapping[tuple([0] * ts[0].n_var)]
    for exponents, index_coeff in mapping.items():
        if coeff[index_coeff] != 0.:
            mean += coeff[index_coeff] * moment_generating_func.get_partial_deriv(exponents)

    return mean


def mean_from_uniform(p: RealMultivarTaylor, lbs, ubs) -> float:
    """ Assuming the variables of a Taylor expansion form a uniformly distributed random vector, compute its analytical
    mean. It uses a closed form expression for all the moments involved.

    Args:
          p (RealMultivarTaylor): Taylor expansion.
          lbs (Iterable[float]): lower bounds of independent uniform distributions.
          ubs (Iterable[float]): upper bounds of independent uniform distributions.

    Returns:
         float: mean value of the Taylor expansion obtained by evaluating its variables as the components of the
            inputted uniform law.

    """

    # sanity checks
    if p.is_trivial():
        return p.const
    if len(lbs) != p.n_var:
        raise ValueError("The inputted lower bounds do not have same dimension as number of variables in expansion.")
    if len(ubs) != len(lbs):
        raise ValueError("The size of the upper bounds does not match the one of the lower bounds.")
    if len(ubs[ubs <= lbs]) != 0:
        raise ValueError("At least one upper bound is lower than lower bound.")

    moments = np.ones((len(ubs), p.order))
    for i, (lb, ub) in enumerate(zip(lbs, ubs)):
        if lb != 0.:
            ratio = power_ratio = ub / lb
            power_lb = lb
            summed = 1. + power_ratio
            moments[i, 0] = summed * power_lb
            for j in range(1, p.order):
                power_lb *= lb
                power_ratio *= ratio
                summed += power_ratio
                moments[i, j] = summed * power_lb
        else:
            moments[i, :] = np.cumprod(np.full(p.order, ub))
        moments[i, :] /= np.arange(2., p.order + 2., 1.)

    # compute mean by linearity i.e. adding all the contributions (coefficient times mean of monomial)
    mean = p.const
    coeff = p.coeff
    mapping = dict(p.get_mapping_monom())
    del mapping[tuple([0] * p.n_var)]
    for exponents, index_coeff in mapping.items():
        if coeff[index_coeff] != 0.:
            inter = 1.
            for j, el in enumerate(exponents):
                if el != 0:
                    inter *= moments[j, el - 1]
            mean += coeff[index_coeff] * inter

    return mean


# define (co)variance functions from the mean
var_from_normal = variance_from_mean(mean_from_normal)
covar_from_normal = covariance_from_mean(mean_from_normal)
var_from_uniform = variance_from_mean(mean_from_uniform)
covar_from_uniform = covariance_from_mean(mean_from_uniform)


# wrap functions from module math
arccos = acos
arcsin = asin
arctan = atan
arccosh = acosh
arcsinh = asinh
arctanh = atanh
erf = taylor_real(erf_taylor)(math.erf)
erfc = taylor_real(erfc_taylor)(math.erfc)
