import unittest
import numpy as np
from swiftt import integrators
from swiftt.taylor import taylor_map, real_multivar_taylor, factory_taylor
from swiftt.math_algebra import cos

tol_coeff = 1.e-12

null_expansion_2var_order2 = factory_taylor.zero_expansion(2, 2)

null_expansion_2var_order3 = factory_taylor.zero_expansion(2, 3)

null_expansion_3var_order2 = factory_taylor.zero_expansion(3, 2)

null_expansion_4var_order5 = factory_taylor.zero_expansion(4, 5)


def test_integrator_history(integ):
    coeff = np.zeros(null_expansion_2var_order3.dim_alg)
    coeff[1] = 1.
    x0 = [null_expansion_2var_order3.create_expansion_with_coeff(coeff)]
    coeff[1], coeff[2] = 0., 1.
    x0.append(null_expansion_2var_order3.create_expansion_with_coeff(coeff))
    xf1 = integ.integrate(0., 1., x0, 10, keep_history=True)[0][-1]
    xf2 = integ.integrate(0., 1., x0, 10, keep_history=False)[0][-1]
    for el1, el2 in zip(xf1, xf2):
        if el1 != el2:
            return False
    return True


class TestIntegrator(unittest.TestCase):

    def test_propag_dim2(self):
        order = 4
        expansions = factory_taylor.create_unknown_map(order, [1., 0.])

        tf = 1.
        n_steps = 20

        def func(tau, x):
            return taylor_map.RealTaylorMap([x[1], 2. * (tau * x[1] + x[0])])

        integ = integrators.ABM8(func)  # RKF45(func, max_stepsize=None, step_multiplier=None)
        states, times = integ.integrate(0., tf, expansions, n_steps, keep_history=False)

        if np.fabs(states[-1][0].const - np.exp(tf * tf)) > 1e-4:
            self.fail()

    def test_integ_euler(self):
        if not test_integrator_history(integrators.Euler(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                               dtype=real_multivar_taylor.RealMultivarTaylor))):
            self.fail()

    def test_integ_heun(self):
        if not test_integrator_history(integrators.Heun(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                              dtype=real_multivar_taylor.RealMultivarTaylor))):
            self.fail()

    def test_integ_rk4(self):
        if not test_integrator_history(integrators.RK4(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                             dtype=real_multivar_taylor.RealMultivarTaylor))):
            self.fail()

    def test_integ_bs(self):
        if not test_integrator_history(integrators.BS(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                            dtype=real_multivar_taylor.RealMultivarTaylor), 4)):
            self.fail()

    def test_integ_ab8(self):
        if not test_integrator_history(integrators.AB8(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                             dtype=real_multivar_taylor.RealMultivarTaylor))):
            self.fail()

    def test_integ_abm8(self):
        if not test_integrator_history(integrators.ABM8(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                              dtype=real_multivar_taylor.RealMultivarTaylor))):
            self.fail()

    def test_integ_rkf45(self):
        if not test_integrator_history(integrators.RKF45(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                               dtype=real_multivar_taylor.RealMultivarTaylor), 2)):
            self.fail()

    def test_integ_rkf45_event(self):
        if not test_integrator_history(integrators.RKF45(lambda t, x: np.array([cos(x[1]), -x[0]],
                                                                               dtype=real_multivar_taylor.RealMultivarTaylor), 2,
                                                         event_func=lambda t, x: t > 1000., tol_event=0.1)):
            self.fail()
    #
    # def test_event_detec(self):
    #
    #     t0 = 0.
    #     tf = 10.
    #     x0 = np.array([0.5, 1.])
    #     n_steps = int((tf - t0) / 0.01)
    #     tol_event = 0.01
    #
    #     def fun(t, x):
    #         return np.array([x[1], cos(x[0])])
    #
    #     def event(t, x):
    #         return x[0] < 0.
    #
    #     inteRKF45 = integrators.RKF45(fun, 2, event_func=event, tol_event=tol_event)
    #     statesRKF45, timesRKF45 = inteRKF45.integrate(t0, tf, x0, n_steps, keep_history=False)
    #
    #     inteTaylor = integrators_algebra.TaylorVarsize(fun, order=4, dim_state=2, event_func=event,
    #                                                    tol_event=tol_event)
    #     statesTaylor, timesTaylor = inteTaylor.integrate(t0, tf, x0, n_steps, keep_history=False)
    #
    #     if not np.allclose(timesRKF45[-1], timesTaylor[-1], atol=1.e-2, rtol=1.e-3):
    #         self.fail()


if __name__ == '__main__':
    unittest.main()
