"""Mathematical and rollback checks for the transmission adapters."""

import unittest

import numpy as np

try:
    from .adapters import make_adapter
except ImportError:
    from adapters import make_adapter


NAMES = ("instant", "leaky", "prospective", "ideal")


def integrate_rk4(u, a, f, tau, tau_a, duration, gain, steps=12000):
    """Independent fine-step integration of the differential equations."""
    y = np.stack((u.copy(), a.copy()))
    h = duration / steps

    def rhs(state):
        v, adaptation = state
        return np.stack((
            (-v + f) / tau + gain * (f - adaptation) / tau_a,
            (-adaptation + f) / tau_a,
        ))

    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


class AdapterTests(unittest.TestCase):
    def test_initialization_is_equilibrated_and_constant_input_stays_constant(self):
        f = np.array([-1.0, 0.5, 3.0])
        for name in NAMES:
            with self.subTest(name=name):
                adapter = make_adapter(name, len(f))
                for _ in range(10):
                    np.testing.assert_array_equal(adapter.apply(f), f)

    def test_adaptive_exact_solution_matches_independent_ode_integrator(self):
        initial = np.array([0.4, -0.9, 1.2])
        f = np.array([-1.5, 2.0, 0.3])
        for tau, tau_a, gain in ((2.0, 0.2, 1.0), (0.4, 1.1, 0.7), (0.8, 0.8, 1.3)):
            with self.subTest(tau=tau, tau_a=tau_a, gain=gain):
                adapter = make_adapter("prospective", 3, tau=tau, tau_a=tau_a, dt=0.7, gain=gain)
                adapter.apply(initial)
                result = adapter.apply(f)
                expected_u, expected_a = integrate_rk4(initial, initial, f, tau, tau_a, 0.7, gain)
                np.testing.assert_allclose(result, expected_u, rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(adapter.snapshot()["state"]["a"], expected_a, rtol=2e-11, atol=2e-11)

    def test_equal_and_nearly_equal_time_constants_use_finite_correct_limit(self):
        tau, dt, gain = 0.7, 1.1, 1.2
        start = np.array([0.0, 0.5])
        f = np.array([1.0, -2.0])
        decay = np.exp(-dt / tau)
        expected = f + (start - f) * decay - gain * (start - f) * (dt / tau) * decay
        for factor in (1.0, 1.0 + 1e-12, 1.0 - 1e-12):
            with self.subTest(factor=factor):
                adapter = make_adapter("prospective", 2, tau=tau, tau_a=tau * factor, dt=dt, gain=gain)
                adapter.apply(start)
                np.testing.assert_allclose(adapter.apply(f), expected, rtol=2e-12, atol=2e-12)

    def test_held_input_solution_is_independent_of_interval_partition(self):
        start = np.array([-0.7, 0.8])
        target = np.array([1.5, -0.3])
        for name in ("leaky", "prospective"):
            with self.subTest(name=name):
                whole = make_adapter(name, 2, tau=1.2, tau_a=0.18, dt=0.8)
                half = make_adapter(name, 2, tau=1.2, tau_a=0.18, dt=0.4)
                whole.apply(start)
                half.apply(start)
                expected = whole.apply(target)
                half.apply(target)
                actual = half.apply(target)
                np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)

    def test_zero_prospective_gain_recovers_leaky_dynamics(self):
        leaky = make_adapter("leaky", 4)
        adaptive = make_adapter("prospective", 4, gain=0)
        rng = np.random.default_rng(81)
        for f in rng.normal(size=(30, 4)):
            np.testing.assert_array_equal(adaptive.apply(f), leaky.apply(f))

    def test_ideal_inverse_equals_instantaneous_on_arbitrary_inputs(self):
        rng = np.random.default_rng(22)
        for tau in (0.01, 2.0, 100.0):
            adapter = make_adapter("ideal", 8, tau=tau)
            for f in rng.normal(size=(100, 8)):
                np.testing.assert_array_equal(adapter.apply(f), f)

    def test_finite_response_to_smooth_and_abrupt_inputs(self):
        t = np.arange(300) * 0.1
        signals = np.stack((np.sin(t), np.cos(0.3 * t), (t > 8).astype(float) * 4 - 2), axis=1)
        for name in NAMES:
            for gain in (1.0, 2.0):
                with self.subTest(name=name, gain=gain):
                    adapter = make_adapter(name, 3, dt=0.1, gain=gain)
                    outputs = np.stack([adapter.apply(f) for f in signals])
                    self.assertTrue(np.isfinite(outputs).all())
                    self.assertLess(np.abs(outputs).max(), 20.0)

    def test_rollback_matches_fresh_replay_without_aliases(self):
        prefix = [np.array([0.0, 1.0]), np.array([2.0, -1.0])]
        accepted_correction = np.array([-0.2, 0.6])
        for name in NAMES:
            with self.subTest(name=name):
                adapter = make_adapter(name, 2)
                before_initialization = adapter.snapshot()
                for f in prefix:
                    adapter.apply(f)
                saved = adapter.snapshot()
                clone = adapter.clone()
                discarded_output = adapter.apply(np.array([9.0, -8.0]))
                discarded_output[:] = 100  # Returned arrays do not own live state.
                adapter.restore(saved)
                if saved["state"] is not None:
                    for value in saved["state"].values():
                        value[:] = -100  # Restoring does not retain snapshot aliases.
                actual = adapter.apply(accepted_correction)
                expected = clone.apply(accepted_correction)
                np.testing.assert_array_equal(actual, expected)
                adapter.restore(before_initialization)
                np.testing.assert_array_equal(adapter.apply(accepted_correction), accepted_correction)

    def test_snapshot_rejects_other_adapter_or_configuration(self):
        adapter = make_adapter("prospective", 2)
        with self.assertRaises(ValueError):
            adapter.restore(make_adapter("leaky", 2).snapshot())
        with self.assertRaises(ValueError):
            adapter.restore(make_adapter("prospective", 2, tau=3).snapshot())

    def test_invalid_inputs_fail_before_changing_state(self):
        adapter = make_adapter("prospective", 2)
        adapter.apply(np.array([1.0, 2.0]))
        reference = adapter.clone()
        for bad in (np.zeros(3), np.array([np.nan, 1.0]), np.array([np.inf, 0.0])):
            with self.assertRaises(ValueError):
                adapter.apply(bad)
        f = np.array([3.0, -1.0])
        np.testing.assert_array_equal(adapter.apply(f), reference.apply(f))
        for kwargs in ({"tau": 0}, {"tau_a": -1}, {"dt": np.nan}, {"gain": np.inf}):
            with self.assertRaises(ValueError):
                make_adapter("prospective", 2, **kwargs)


if __name__ == "__main__":
    unittest.main()
