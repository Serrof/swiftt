# integrators.py: range of classes implementing integrators
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

from typing import Optional, List, Iterable, Tuple, Callable
from abc import ABCMeta, abstractmethod
import numpy as np
from swiftt.taylor.complex_multivar_taylor import ComplexMultivarTaylor


class Integrator(metaclass=ABCMeta):
    """Abstract class for the implementation of numerical integrators.

    Attributes:
        _func (Callable): function of the independent variable and the state vector defining the derivative of the
        latter w.r.t. the former.
        _order (int): order of integration scheme.

    """

    def __init__(self, func: Callable, order: int) -> None:
        """Constructor for class Integrator.

        Args:
             func (Callable):
             function of the independent variable and the state vector defining the derivative of the latter
                w.r.t. the former.
             order (int): order of integration scheme.

        """

        self._func = func
        self._order = order

    @abstractmethod
    def integrate(self, t0: float, tf: float, x0: np.ndarray, n_step: int, keep_history: bool) -> Tuple:
        """Abstract method to implement. Performs the numerical integration between initial to final values of
        independent variable, with provided initial conditions.

        Args:
             t0 (float): initial time.
             tf (float): final time.
             x0 (numpy.ndarray): initial conditions.
             n_step (int): number of integrations steps.
             keep_history (bool): set to True to return the whole history of successful steps, False to return
                only the initial and final states.

        """
        raise NotImplementedError


class FixedstepIntegrator(Integrator, metaclass=ABCMeta):
    """Abstract class for the implementation of numerical integrators with a fixed step-size.

    """

    @staticmethod
    def step_size(t0: float, tf: float, n_step: int) -> float:
        """Static method computing the constant step-size corresponding to initial and final values of independent
        variable as well as number of integration steps.

        Args:
            t0 (float): initial value of independent variable.
            tf (float): final time.
            n_step (int): number of steps.

        Returns:
            float: step-size.

        """
        return (tf - t0) / n_step

    @abstractmethod
    def integration_step(self, t: float, x: np.ndarray, h):
        """Abstract method to be overwritten in classes inheriting from abstract class. Performs a single integration
        step.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): step-size.

        """
        raise NotImplementedError

    def _precompute(self, h) -> None:
        """Function to overwrite in order to perform pre-computations, if any, ahead of integration.

        Args:
            h (Union[float, ComplexMultivarTaylor]): step-size.

        """
        return

    def integrate(self, t0: float, tf: float, x0: np.ndarray, n_step: int, keep_history: bool) -> Tuple[List, List]:
        """Function that performs integration between two values of independent variable. It is vectorized w.r.t. x0 if
        self._func is: in other words, several initial states can be propagated in one call (with the same value for the
        initial independent variable and the same number of steps).

        Args:
            t0 (float): initial value of independent variable.
            tf (float): final time.
            x0 (numpy.ndarray): initial conditions.
            n_step (int): number of integration steps to be performed.
            keep_history (bool): set to True to return the whole history of successful steps, False to return
                only the initial and final states.

        Returns:
            List[numpy.ndarray]: state vectors at integration steps of interest.
            List[float]: values taken by the independent variable at integration steps.

        """

        h = self.step_size(t0, tf, n_step)
        Ts, Xs = [t0], [x0]
        self._precompute(h)

        if keep_history:
            for k in range(0, n_step):
                Xs.append(self.integration_step(Ts[k], Xs[k], h))
                Ts.append(Ts[k] + h)
        else:
            # first step
            Xs.append(self.integration_step(t0, x0, h))
            Ts.append(t0 + h)
            # rest of integration
            for __ in range(1, n_step):
                Xs[1] = self.integration_step(Ts[1], Xs[1], h)
                Ts[1] += h

        return Xs, Ts


class Euler(FixedstepIntegrator):
    """Class implementing the classic Euler integration scheme.

    """

    def __init__(self, func: Callable) -> None:
        """Constructor for Euler class.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.

        """
        FixedstepIntegrator.__init__(self, func, order=1)

    def integration_step(self, t: float, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + h where h is the step-size.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): step-size.

        Returns:
            numpy.ndarray: state vector at t + h.

        """

        return x + h * self._func(t, x)


class Heun(FixedstepIntegrator):
    """Class implementing the Heun integration scheme.

    Attributes:
        _half_step (Union[float, ComplexMultivarTaylor]): stored half step-size.

    """

    def __init__(self, func: Callable) -> None:
        """Constructor for Heun class.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.

        """
        FixedstepIntegrator.__init__(self, func, order=2)
        self._half_step = None

    def _precompute(self, h) -> None:
        """Overload parent implementation in order to pre-compute the half step-size.

        Args:
            h (Union[float, ComplexMultivarTaylor]): step-size.

        """

        self._half_step = 0.5 * h

    def integration_step(self, t: float, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + h where h is the step-size.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): step-size.

        Returns:
            numpy.ndarray: state vector at t + h.

        """

        f1 = self._func(t, x)  # function call

        x1 = x + h * f1
        f2 = self._func(t + h, x1)  # function call

        return x + self._half_step * (f1 + f2)


class RK4(FixedstepIntegrator):
    """Class implementing the classic Runge-Kutta 4 integration scheme.

    Attributes:
        _half_step (Union[float, ComplexMultivarTaylor]): stored half step-size.
        _one_third_step (Union[float, ComplexMultivarTaylor]): stored one third of step-size.
        _one_sixth_step (Union[float, ComplexMultivarTaylor]): stored one sixth of step-size.

    """

    def __init__(self, func: Callable) -> None:
        """Constructor for RK4 class.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.

        """
        FixedstepIntegrator.__init__(self, func, order=4)
        self._half_step = None
        self._one_third_step = None
        self._one_sixth_step = None

    def _precompute(self, h) -> None:
        """Overload parent implementation in order to pre-compute quantities such as the half step-size.

        Args:
            h (Union[float, ComplexMultivarTaylor]): step-size.

        """

        self._half_step = h / 2.
        self._one_third_step = h / 3.
        self._one_sixth_step = h / 6.

    def integration_step(self, t: float, x, h):
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + h where h is the step-size.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): step-size.

        Returns:
            numpy.ndarray: state vector at t + h.

        """

        middle_time = t + self._half_step

        f1 = self._func(t, x)  # function call

        x1 = x + self._half_step * f1
        f2 = self._func(middle_time, x1)  # function call

        x2 = x + self._half_step * f2
        f3 = self._func(middle_time, x2)  # function call

        x3 = x + h * f3
        f4 = self._func(t + h, x3)  # function call

        return x + (self._one_sixth_step * (f1 + f4) + self._one_third_step * (f2 + f3))


class BS(FixedstepIntegrator):
    """Class implementing the Bulirsch-Stoer integration scheme.

    Attributes:
         _sequence (numpy.ndarray): Bulirsch sequence of integers to be used in scheme.

    """

    def __init__(self, func: Callable, order: int) -> None:
        """Constructor for BS class.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.
             order (int): order of integrator.

        """
        FixedstepIntegrator.__init__(self, func, order)

        self._sequence = np.zeros(self._order, dtype=int)
        self._sequence[0] = 2
        if self._order > 1:
            self._sequence[1] = 4
            if self._order > 2:
                self._sequence[2] = 6
                for k in range(3, self._order):
                    self._sequence[k] = 2 * self._sequence[k - 2]

        # pre-compute intermediate quantities for extrapolation
        self._aux_extrap = np.zeros((self._order + 1, self._order + 1))
        inter = 1. / np.flip(self._sequence)
        for i, el in enumerate(self._sequence[1:], 1):
            self._aux_extrap[i + 1, : i] = el * inter[-i:]
        self._aux_extrap = 1. / (self._aux_extrap ** 2 - 1.)

    def integration_step(self, t: float, x: np.ndarray, H) -> np.ndarray:
        """Function performing a single integration step i.e. given the state vector at the current value t of
        the independent variable, approximates its value at t + H where H is the step-size.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            H (Union[float, ComplexMultivarTaylor]): step-size.

        Returns:
            numpy.ndarray: state vector at t + H.

        """

        M = self._extrapolation(self._order, H, x, t)

        return M[-1]

    def _midpoint(self, n: int, H, y: np.ndarray, t: float) -> np.ndarray:
        """Function applying the mid-point rule of the Bulirsch-Stoer method.

        Args:
            n (int): order.
            H (Union[float, ComplexMultivarTaylor]): step-size.
            y (numpy.ndarray): current state vector.
            t (float): current value of independent variable.

        Returns:
            numpy.ndarray: output of mid-point rule.

        """

        h = H / float(n)
        h2 = 2. * h

        u0 = None

        f1 = self._func(t, y)  # function call
        u1 = y + h * f1

        f2 = self._func(t + h, u1)  # function call
        u2 = y + h2 * f2

        for j in range(2, n + 1):
            f = self._func(t + j * h, u2)  # function call
            u2, u1, u0 = u1 + h2 * f, u2, u1

        return 0.25 * (u0 + u2) + 0.5 * u1

    def _extrapolation(self, i: int, H, y: np.ndarray, t: float) -> List[np.ndarray]:
        """Function performing the extrapolation according to the Bulirsch-Stoer algorithm.

        Args:
            i (int): extrapolation order.
            H (Union[float, ComplexMultivarTaylor]): step-size.
            y (numpy.ndarray): current state vector.
            t (float): current value of independent variable.

        Returns:
            List[numpy.ndarray]: concatenated extrapolated vectors.

        """

        M = [self._midpoint(self._sequence[i - 1], H, y, t)]

        if i > 1:
            Mp = self._extrapolation(i - 1, H, y, t)  # recursive call
            for j, el in enumerate(Mp):
                eta = M[j]
                M.append(eta + (eta - el) * self._aux_extrap[i, j])

        return M


class MultistepIntegrator(FixedstepIntegrator, metaclass=ABCMeta):
    """Abstract class for the implementation of multi-step integrators with fixed step-size.

    Attributes:
         saved_steps (List[numpy.ndarray]): values of state derivative at previous steps.
         _stepsize (Union[float, ComplexMultivarTaylor]): step-size.
         _beta (numpy.ndarray): vector of numbers used in integration scheme.
         _initializer (FixedstepIntegrator): integrator used to initialize the multi-step method.

    """

    def __init__(self, func: Callable, order: int) -> None:
        """Constructor for class MultistepIntegrator.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.
             order (int): order of integrator.

        """
        FixedstepIntegrator.__init__(self, func, order)

        self._stepsize = 0.
        self.saved_steps = []
        self._beta = None
        self._initializer: FixedstepIntegrator = None

    def update_saved_steps(self, t: float, x: np.ndarray) -> None:
        """Function updating the saved values of self._func at the past self._order steps.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.

        """

        self.saved_steps = self.saved_steps[1:]  # shift
        f = self._func(t, x)  # function call
        self.saved_steps.append(f)

    def update_state(self, x: np.ndarray) -> np.ndarray:
        """Function propagating the state vector over one integration step.

        Args:
            x (numpy.ndarray): current state vector.

        Returns:
            numpy.ndarray: state vector at next integration step.

        """

        dx = sum(step * beta for step, beta in zip(self.saved_steps, self._beta))

        return x + self._stepsize * dx

    def integration_step(self, t: float, x: np.ndarray, h=None) -> np.ndarray:
        """Function performing a single integration step.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): step-size (dummy variable in multi-step integrator here to
                match parent signature)

        Returns:
            numpy.ndarray: state vector at t + self._stepsize.

        """

        xf = self.update_state(x)

        self.update_saved_steps(t + self._stepsize, xf)

        return xf

    def initialize(self, t0: float, x0: np.ndarray, h) -> Tuple[List, List]:
        """Function initializing the integrator with a single-step scheme.

        Args:
            t0 (float): initial value of independent variable.
            x0 (numpy.ndarray): state vector at t0.
            h (Union[float, ComplexMultivarTaylor]): step-size

        Returns:
            List[numpy.ndarray]: history of state vector after initialization with single-step integrator.
            List[float]: values of independent variable corresponding to history of state vector.

        """

        self._stepsize = h
        n_steps = self._order - 1
        states, ind_vars = self._initializer.integrate(t0, t0 + float(n_steps) * h, x0, n_steps, keep_history=True)

        self.saved_steps = [self._func(ind_var, state) for ind_var, state in zip(ind_vars, states)]

        return states, ind_vars

    def integrate(self, t0: float, tf: float, x0: np.ndarray, n_step: int, keep_history: bool,
                  saved_steps: Optional[List] = None) -> Tuple[List, List]:
        """Function that performs integration between two values of independent variable. It is vectorized w.r.t. x0 if
        self._func is: in other words, several initial states can be propagated in one call (with the same value for the
        initial independent variable and the same number of steps).

        Args:
            t0 (float): initial value of independent variable.
            tf (float): final value of independent variable.
            x0 (numpy.ndarray): state vector at t0.
            n_step (int): number of integration steps to be performed.
            keep_history (bool): set to True to return the whole history of successful steps, False to return
                only the initial and final states.
            saved_steps (List[numpy.ndarray]): past values of self._func.

        Returns:
            List[numpy.ndarray]: state vectors at integration steps of interest.
            List[float]: values taken by the independent variable at integration steps.

        """
        self.saved_steps = []
        if saved_steps is not None and len(saved_steps) == self._order:
            # input saved steps are recyclable
            self.saved_steps = list(saved_steps)

        h = self.step_size(t0, tf, n_step)

        # initialize steps
        if self._stepsize != h or self.saved_steps == []:
            Xs, Ts = self.initialize(t0, x0, h)
            n_start = len(Ts) - 1  # number of steps already performed
            if n_step <= n_start:
                # enough steps have already been performed, integration is over
                if keep_history:
                    return Xs[:n_step + 1], Ts[:n_step + 1]
                # keep only initial state and last step
                return [x0, Xs[n_step + 1]], [t0, Ts[n_step + 1]]
            if not keep_history:
                Ts, Xs = [t0, Ts[-1]], [x0, Xs[-1]]
        else:
            # step-size has not changed and there are available saved steps
            Ts, Xs = [t0, t0 + self._stepsize], [x0, self.integration_step(t0, x0)]
            n_start = 1  # number of steps already performed

        # perform the rest of the integration
        if keep_history:
            for k in range(n_start, n_step):
                Xs.append(self.integration_step(Ts[k], Xs[k]))
                Ts.append(Ts[k] + self._stepsize)
        else:
            for __ in range(n_start, n_step):
                Xs[1] = self.integration_step(Ts[1], Xs[1], self._stepsize)
                Ts[1] += self._stepsize

        return Xs, Ts


class AB8(MultistepIntegrator):
    """Class implementing the Adam-Bashforth integration scheme of order 8.

    """

    def __init__(self, func: Callable) -> None:
        """Constructor for class AB8.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.

        """

        MultistepIntegrator.__init__(self, func, order=8)

        self._beta = np.array([-36799., 295767., -1041723., 2102243., -2664477., 2183877., -1152169., 434241.]) / 120960.
        self._initializer = BS(self._func, (self._order + 1) // 2)


class ABM8(MultistepIntegrator):
    """Class implementing the Adam-Bashforth-Moulton integration scheme of order 8.

    """
    def __init__(self, func: Callable) -> None:
        """Constructor for class ABM8.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.

        """

        MultistepIntegrator.__init__(self, func, order=8)

        self._beta = np.array([1375., -11351., 41499., -88547., 123133., -121797., 139849., 36799.]) / 120960.
        self._predictor = self._initializer = AB8(self._func)

    def integration_step(self, t: float, x: np.ndarray, h=None) -> np.ndarray:
        """Function performing a single integration step.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): step-size (dummy variable in multi-step integrator here to
                match parent signature)

        Returns:
            numpy.ndarray: state vector at t + self._stepsize.

        """

        self._predictor.integration_step(t, x)  # (hides a function call)

        self.saved_steps = list(self._predictor.saved_steps)

        xf = self.update_state(x)

        f = self._func(t + self._stepsize, xf)  # function call

        del self._predictor.saved_steps[-1]
        self._predictor.saved_steps.append(f)

        return xf


class VariableStepIntegrator(Integrator, metaclass=ABCMeta):
    """Abstract class for the implementation of integrators with step-size control.

    Attributes:
        _dim_state (int): dimension of state vector.
        _last_step_ok (bool): false if last step didn't satisfy the constraint on the absolute error, true
        otherwise.
        _error_exponent (float): exponent used in derivation of estimate for integration error.
        _abs_tol (Iterable[float]): tolerance vector on estimated absolute error. Should have same number of
        components than there are state variables. Default is 1.e-8 for each.
        _rel_tol (Iterable[float]): tolerance vector on estimated relative error. Should have same number of
        components than there are state variables. Default is 1.e-4 for each.
        _max_stepsize (float): maximum step-size allowed. Default is + infinity.
        _step_multiplier (float): multiplicative factor to increase step-size when an integration step has
        been successful.

    """

    def __init__(self, func: Callable, order: int, dim_state: int, abs_error_tol=None, rel_error_tol=None,
                 max_stepsize: Optional[float] = None, step_multiplier: Optional[float] = None,
                 event_func: Optional[Callable] = None, tol_event: Optional[float] = None) -> None:
        """Constructor for class VariableStepIntegrator.

        Args:
             func (Callable): function of the independent variable and the state vector defining the derivative of the
                latter w.r.t. the former.
             order (int): order of integrator.
             dim_state (int): dimension of state factor.
             abs_error_tol (): tolerance vector on estimated absolute error. Should have same
                number of components than there are state variables. Default is 1.e-8 for each.
             rel_error_tol (): tolerance vector on estimated relative error. Should have same
                number of components than there are state variables. Default is 1.e-4 for each.
             max_stepsize (float): maximum step-size allowed. Default is + infinity.
             step_multiplier (float): multiplicative factor to increase step-size when an integration step has
                been successful.

        """

        Integrator.__init__(self, func, order)

        self._dim_state = dim_state

        self._last_step_ok = True
        self._error_exponent: float = None

        if event_func is None:
            self._event_func = None
        else:
            self._event_func = event_func
            self._tol_event = tol_event
            if tol_event is None:
                raise ValueError("An event-detection tolerance must be provided if an event function is given.")

        default_step_multiplier = 2.
        if step_multiplier is None:
            self._step_multiplier = default_step_multiplier
        else:
            if 1. <= step_multiplier <= 5.:
                self._step_multiplier = float(step_multiplier)
            else:
                print("input step multiplier is not in [1, 5], switching to default value of "
                      + str(default_step_multiplier))
                self._step_multiplier = default_step_multiplier

        self._max_stepsize = np.inf if max_stepsize is None else max_stepsize

        default_abs_tol = 1.e-8
        self._abs_tol = np.full(self._dim_state, default_abs_tol)
        if abs_error_tol is not None:
            if len(abs_error_tol) != self._dim_state:
                raise ValueError("Wrong input in VariableStepIntegrator: tolerance on absolute error must have same "
                                 "dimension than state vector")
            for i, tol in enumerate(abs_error_tol):
                if tol <= 0.:
                    print("Input tolerance on absolute error is negative, switching to default value of "
                          + str(default_abs_tol) + " with state variable " + str(i))
                else:
                    self._abs_tol[i] = tol

        default_rel_tol = 1.e-4
        self._rel_tol = np.full(self._dim_state, default_rel_tol)
        if rel_error_tol is not None:
            if len(rel_error_tol) != self._dim_state:
                raise ValueError("Wrong input in VariableStepIntegrator: tolerance on relative error must have same "
                                 "dimension than state vector")
            for i, tol in enumerate(rel_error_tol):
                if tol <= 0.:
                    print("input tolerance on relative error is negative, switching to default value of "
                          + str(default_rel_tol) + " with state variable " + str(i))
                else:
                    self._rel_tol[i] = tol

    @abstractmethod
    def integration_step(self, t: float, x: np.ndarray, h):
        """Abstract method to be overwritten in classes inheriting from abstract class. Performs a single integration
        step.

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): current step-size.

        """
        raise NotImplementedError

    def integrate(self, t0: float, tf: float, x0: np.ndarray, n_step: int, keep_history: bool) -> Tuple[List, List]:
        """Function that performs integration between two values of independent variable.

        Args:
            t0 (float): initial value of independent variable.
            tf (float): final value of independent variable.
            x0 (numpy.ndarray): state vector at t0.
            n_step (int): initial guess for number of integration steps.
            keep_history (bool): set to True to return the whole history of successful steps, False to return
                only the initial and final states.

        Returns:
            List[numpy.ndarray]: state vectors at integration steps of interest.
            List[float]: values taken by the independent variable at integration steps.

        """

        if len(x0) != self._dim_state:
            raise ValueError("Wrong input in integrate: state vector has different dimension than the one given when "
                             "the integrator was instantiated")

        # call dedicated method if there is an event function
        if self._event_func is None:
            return self._integrate_without_events(t0, tf, x0, n_step, keep_history)

        return self._integrate_with_events(t0, tf, x0, n_step, keep_history)

    def _integrate_without_events(self, t0: float, tf: float, x0: np.ndarray, n_step: int,
                                  keep_history: bool) -> Tuple[List, List]:
        """Function that performs integration between two values of independent variable without event detection.

        Args:
            t0 (float): initial value of independent variable.
            tf (float): final value of independent variable.
            x0 (numpy.ndarray): state vector at t0.
            n_step (int): initial guess for number of integration steps.
            keep_history (bool): set to True to return the whole history of successful steps, False to return
                only the initial and final states.

        Returns:
            List[numpy.ndarray]: state vectors at integration steps of interest.
            List[float]: values taken by the independent variable at integration steps.

        """

        # initial guess for step-size
        h = FixedstepIntegrator.step_size(t0, tf, n_step)

        # save direction of integration
        forward = tf > t0

        if keep_history:
            Ts, Xs = [t0], [x0]
        else:
            Ts, Xs = [t0, t0], [x0, x0]

        t = t0
        abs_dt = abs(tf - t0)
        while abs(t - t0) < abs_dt:
            # check and possibly decrease step-size
            if abs(h) > self._max_stepsize:
                h = self._max_stepsize if forward else -self._max_stepsize
            if (t + h > tf and forward) or (t + h < tf and not forward):
                h = tf - t

            # compute candidate new state and associated integration error
            x, err = self.integration_step(t, Xs[-1], h)

            # check viability of integration step
            if isinstance(x[0], ComplexMultivarTaylor):
                tol = self._abs_tol + np.array([el.norm for el in x]) * self._rel_tol
                err_ratios = np.array([el.norm for el in err]) / tol
            else:
                tol = self._abs_tol + np.fabs(x) * self._rel_tol
                err_ratios = np.fabs(err) / tol
            max_err_ratio = np.max(err_ratios)
            self._last_step_ok = max_err_ratio < 1.

            if self._last_step_ok:
                factor = self._step_multiplier
                t += h
                if keep_history:
                    Ts.append(t)
                    Xs.append(x)
                else:
                    Ts[1], Xs[1] = t, x
            else:
                # step was not successful
                factor = 0.9 * (1. / float(max_err_ratio)) ** self._error_exponent

            # step-size update
            h *= factor

        return Xs, Ts

    def _integrate_with_events(self, t0: float, tf: float, x0: np.ndarray, n_step: int,
                               keep_history: bool) -> Tuple[List, List]:
        """Function that performs integration between two values of independent variable whilst taking event detection
         into account, so possibly stopping early.

        Args:
            t0 (float): initial value of independent variable.
            tf (float): final value of independent variable.
            x0 (numpy.ndarray): state vector at t0.
            n_step (int): initial guess for number of integration steps.
            keep_history (bool): set to True to return the whole history of successful steps, False to return
                only the initial and final states.

        Returns:
            List[numpy.ndarray]: state vectors at integration steps of interest.
            List[float]: values taken by the independent variable at integration steps.

        """

        # initial guess for step-size
        h = FixedstepIntegrator.step_size(t0, tf, n_step)

        # save direction of integration
        forward = tf > t0

        if keep_history:
            Ts, Xs = [t0], [x0]
        else:
            Ts, Xs = [t0, t0], [x0, x0]

        t = t0
        abs_dt = abs(tf - t0)
        while abs(t - t0) < abs_dt:
            # check and possibly decrease step-size
            if abs(h) > self._max_stepsize:
                h = self._max_stepsize if forward else -self._max_stepsize
            if (t + h > tf and forward) or (t + h < tf and not forward):
                h = tf - t

            # compute candidate new state and associated integration error
            x, err = self.integration_step(t, Xs[-1], h)

            # check viability of integration step
            if isinstance(x[0], ComplexMultivarTaylor):
                tol = self._abs_tol + np.array([el.norm for el in x]) * self._rel_tol
                err_ratios = np.array([el.norm for el in err]) / tol
            else:
                tol = self._abs_tol + np.fabs(x) * self._rel_tol
                err_ratios = np.fabs(err) / tol
            max_err_ratio = np.max(err_ratios)
            self._last_step_ok = max_err_ratio < 1.

            if self._last_step_ok:
                factor = self._step_multiplier
                if self._event_func(t, x):
                    if abs(h) > self._tol_event:
                        # detection tolerance is not satisfied so step-size is halved
                        factor = 0.5
                    else:
                        # event has been detected satisfyingly, integration is stopped
                        tf = t
                        abs_dt = abs(t - t0)
                        if keep_history:
                            Ts.append(t)
                            Xs.append(x)
                        else:
                            Ts[1], Xs[1] = t, x
                else:
                    t += h
                    if keep_history:
                        Ts.append(t)
                        Xs.append(x)
                    else:
                        Ts[1], Xs[1] = t, x
            else:
                # step was not successful
                factor = 0.9 * (1. / float(max_err_ratio)) ** self._error_exponent

            # step-size update
            h *= factor

        return Xs, Ts


class RKF45(VariableStepIntegrator):
    """Class implementing the Runge-Kutta-Fehlberg 4(5) integration scheme.

    Attributes:
        _factor_t3 (Union[float, ComplexMultivarTaylor]): pre-computed factor involved in calculation of t3
        _factor_t4 (Union[float, ComplexMultivarTaylor]): pre-computed factor involved in calculation of t4
        _factor_x2 (Union[float, ComplexMultivarTaylor]): pre-computed factor involved in calculation of x2
        _factor_x3 (Union[float, ComplexMultivarTaylor]): pre-computed factor involved in calculation of x3
        _factor_x4_f1 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f1 to obtain x4
        _factor_x4_f3 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f3 to obtain x4
        _factor_x4_f4 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f4 to obtain x4
        _factor_x5_f1 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f1 to obtain x5
        _factor_x5_f3 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f3 to obtain x5
        _factor_x5_f4 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f4 to obtain x5
        _factor_x5_f5 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f5 to obtain x5
        _factor_xf_f1 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f1 to obtain xf
        _factor_xf_f3 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f3 to obtain xf
        _factor_xf_f4 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f4 to obtain xf
        _factor_xf_f5 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f5 to obtain xf
        _factor_err_f1 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f1 to obtain err
        _factor_err_f3 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f3 to obtain err
        _factor_err_f4 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f4 to obtain err
        _factor_err_f5 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f5 to obtain err
        _factor_err_f6 (Union[float, ComplexMultivarTaylor]): pre-computed factor multiplied by f6 to obtain err

    """

    def __init__(self, func: Callable, dim_state: int, abs_error_tol=None, rel_error_tol=None,
                 max_stepsize: Optional[float] = None, step_multiplier: Optional[float] = None,
                 event_func: Optional[Callable] = None, tol_event: Optional[float] = None) -> None:
        VariableStepIntegrator.__init__(self, func, order=4, dim_state=dim_state, abs_error_tol=abs_error_tol,
                                        rel_error_tol=rel_error_tol, max_stepsize=max_stepsize,
                                        step_multiplier=step_multiplier, event_func=event_func, tol_event=tol_event)
        self._error_exponent = 1. / (self._order + 1.)
        self._factor_t3 = 3. / 8.
        self._factor_t4 = 12. / 13.
        self._factor_x2 = 3. / 32.
        self._factor_x3 = 1. / 2197.
        self._factor_x4_f1 = 439. / 216.
        self._factor_x4_f3 = 3680. / 513.
        self._factor_x4_f4 = -845. / 4104.
        self._factor_x5_f1 = -8. / 27.
        self._factor_x5_f3 = -3544. / 2565.
        self._factor_x5_f4 = 1859. / 4104.
        self._factor_x5_f5 = -11. / 40.
        self._factor_xf_f1 = 25. / 216.
        self._factor_xf_f3 = 1408. / 2565.
        self._factor_xf_f4 = 2197. / 4104.
        self._factor_xf_f5 = -1. / 5.
        self._factor_err_f1 = 16. / 135.
        self._factor_err_f3 = 6656. / 12825.
        self._factor_err_f4 = 28561. / 56430.
        self._factor_err_f5 = -9. / 50.
        self._factor_err_f6 = 2. / 55.

    def integration_step(self, t: float, x: np.ndarray, h) -> Tuple[np.ndarray, np.ndarray]:
        """Method performing a single integration step (satisfying the error tolerance or not).

        Args:
            t (float): current value of independent variable.
            x (numpy.ndarray): state vector at t.
            h (Union[float, ComplexMultivarTaylor]): current step-size.

        Returns:
            numpy.ndarray: tentative state vector at t + h.
            numpy.ndarray: estimated error vector.

        """
        # values of independent variable where the model will be evaluated
        t1 = t
        dt2 = 0.25 * h
        t2 = t + dt2
        t3 = t + h * self._factor_t3
        t4 = t + h * self._factor_t4
        t5 = t + h
        t6 = t + 0.5 * h

        f1 = self._func(t1, x)  # function call

        x1 = x + dt2 * f1
        f2 = self._func(t2, x1)  # function call

        x2 = x + h * self._factor_x2 * (f1 + f2 * 3.)
        f3 = self._func(t3, x2)  # function call

        x3 = x + h * self._factor_x3 * (f1 * 1932. + f2 * (-7200.) + f3 * 7296.)
        f4 = self._func(t4, x3)  # function call

        x4 = x + h * (f1 * self._factor_x4_f1 + f2 * (-8.) + f3 * self._factor_x4_f3 + f4 * self._factor_x4_f4)
        f5 = self._func(t5, x4)  # function call

        x5 = x + h * (f1 * self._factor_x5_f1 + f2 * 2. + f3 * self._factor_x5_f3 + f4 * self._factor_x5_f4
                      + f5 * self._factor_x5_f5)
        f6 = self._func(t6, x5)  # function call

        inter1 = f1 * self._factor_xf_f1 + f3 * self._factor_xf_f3 + f4 * self._factor_xf_f4 + f5 * self._factor_xf_f5
        xf = h * inter1
        inter2 = f1 * self._factor_err_f1 + f3 * self._factor_err_f3 + f4 * self._factor_err_f4 \
                 + f5 * self._factor_err_f5 + f6 * self._factor_err_f6
        x_hat = h * inter2
        err = xf - x_hat
        xf += x
        return xf, err
