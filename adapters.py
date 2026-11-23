"""Causal transmission adapters for a speculative-decoding experiment.

The signal being transmitted is a vector of draft logits. Token arrivals are
piecewise-constant input events separated by ``dt`` simulation-time units.
``ProspectiveAdapter`` integrates the adaptive-current equations (6--7) in
Zucchet et al., arXiv:2511.14917, exactly over each constant-input interval:

    tau_a * da/dt = -a + f
    tau * du/dt = -u + f + gain * tau/tau_a * (f - a)

This is an inference-time signal adapter, not a reproduction of NLA training.
Persistent autoregressive memory belongs to the surrounding Context, not to
these transmission states. All stateful adapters initialize at the first input
to avoid giving one condition an artificial warm-up penalty.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


class Adapter:
    """Shared input validation and independent state snapshots."""

    name = "base"
    # Conservative protocol: only an explicit False permits the sampler to
    # omit checkpoints and evaluations that exist solely to advance state.
    requires_state_updates = True
    state_keys: tuple[str, ...] = ()

    def __init__(
        self,
        vocab: int,
        tau: float = 2.0,
        tau_a: float = 0.2,
        dt: float = 1.0,
        gain: float = 1.0,
    ) -> None:
        if isinstance(vocab, bool) or not isinstance(vocab, (int, np.integer)) or vocab < 1:
            raise ValueError("vocab must be a positive integer")
        self.vocab = int(vocab)
        self.tau, self.tau_a, self.dt, self.gain = map(float, (tau, tau_a, dt, gain))
        for key in ("tau", "tau_a", "dt"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not math.isfinite(self.gain):
            raise ValueError("gain must be finite")
        self._state: dict[str, np.ndarray] | None = None

    def _input(self, f_logits: np.ndarray) -> np.ndarray:
        f = np.asarray(f_logits, dtype=np.float64)
        if f.shape != (self.vocab,):
            raise ValueError(f"expected logits with shape ({self.vocab},), got {f.shape}")
        if not np.isfinite(f).all():
            raise ValueError("logits must be finite")
        return f.copy()

    def _initialize(self, f: np.ndarray) -> np.ndarray:
        self._state = {key: f.copy() for key in self.state_keys}
        return f.copy()

    def apply(self, f_logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        """Capture every transient state, with no mutable aliases."""
        return {
            "name": self.name,
            "vocab": self.vocab,
            "parameters": (self.tau, self.tau_a, self.dt, self.gain),
            "state": copy.deepcopy(self._state),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore an adapter checkpoint, including an uninitialized state."""
        if (
            snapshot.get("name") != self.name
            or snapshot.get("vocab") != self.vocab
            or snapshot.get("parameters") != (self.tau, self.tau_a, self.dt, self.gain)
        ):
            raise ValueError("snapshot belongs to an incompatible adapter")
        state = snapshot.get("state")
        if state is None:
            self._state = None
            return
        if not isinstance(state, dict) or set(state) != set(self.state_keys):
            raise ValueError("snapshot has invalid state fields")
        # Validate all fields before mutating the live adapter.
        restored = {key: self._input(state[key]) for key in self.state_keys}
        self._state = restored

    def clone(self) -> Adapter:
        return copy.deepcopy(self)


class InstantAdapter(Adapter):
    """The unfiltered drafter: the baseline compensation aims to recover."""

    name = "instant"
    requires_state_updates = False

    def apply(self, f_logits: np.ndarray) -> np.ndarray:
        return self._input(f_logits)


class LeakyAdapter(Adapter):
    """Exact integration of tau * du/dt = -u + f for held input f."""

    name = "leaky"
    state_keys = ("u",)

    def apply(self, f_logits: np.ndarray) -> np.ndarray:
        f = self._input(f_logits)
        if self._state is None:
            return self._initialize(f)
        decay = math.exp(-self.dt / self.tau)
        u = f + (self._state["u"] - f) * decay
        self._state = {"u": u}
        return u.copy()


class ProspectiveAdapter(Adapter):
    """Exact adaptive-current lead compensation with finite tau_a."""

    name = "prospective"
    state_keys = ("u", "a")

    def _lead_coefficient(self) -> float:
        """Return tau/(tau-tau_a) * (exp(-dt/tau_a)-exp(-dt/tau)).

        A short exprel expression avoids cancellation at equal time constants.
        Its exact limit there is -(dt/tau) * exp(-dt/tau).
        """
        decay_u = math.exp(-self.dt / self.tau)
        z = self.dt * (self.tau_a - self.tau) / (self.tau * self.tau_a)
        if abs(z) < 1e-5:
            exprel = 1.0 if z == 0 else math.expm1(z) / z
            return -(self.dt / self.tau_a) * decay_u * exprel
        decay_a = math.exp(-self.dt / self.tau_a)
        return self.tau / (self.tau - self.tau_a) * (decay_a - decay_u)

    def apply(self, f_logits: np.ndarray) -> np.ndarray:
        f = self._input(f_logits)
        if self._state is None:
            return self._initialize(f)
        u0, a0 = self._state["u"], self._state["a"]
        decay_u = math.exp(-self.dt / self.tau)
        decay_a = math.exp(-self.dt / self.tau_a)
        a1 = f + (a0 - f) * decay_a
        u1 = f + (u0 - f) * decay_u + self.gain * (a0 - f) * self._lead_coefficient()
        self._state = {"u": u1, "a": a1}
        return u1.copy()


class IdealAdapter(Adapter):
    """Matched backward-Euler inverse-filter control, not a physical circuit.

    With a=dt/(tau+dt) and b=tau/(tau+dt), the update is
    u[n]=(1-a)u[n-1]+a*f[n]+b*(f[n]-f[n-1]). Therefore the
    residual u[n]-f[n] decays by b; equilibrium initialization makes its
    output identical to the instantaneous baseline up to roundoff. ``gain``
    is intentionally unused: this control always uses matched compensation.
    """

    name = "ideal"
    state_keys = ("u", "previous_input")

    def apply(self, f_logits: np.ndarray) -> np.ndarray:
        f = self._input(f_logits)
        if self._state is None:
            return self._initialize(f)
        b = self.tau / (self.tau + self.dt)
        # Algebraically equivalent residual form avoids subtraction of large
        # lead and lag terms and preserves the exact zero-residual invariant.
        u = f + b * (self._state["u"] - self._state["previous_input"])
        self._state = {"u": u, "previous_input": f.copy()}
        return u.copy()


def make_adapter(
    name: str,
    vocab: int,
    tau: float = 2.0,
    tau_a: float = 0.2,
    dt: float = 1.0,
    gain: float = 1.0,
) -> Adapter:
    """Construct an independent adapter with the common experiment API."""
    classes = {
        "instant": InstantAdapter,
        "leaky": LeakyAdapter,
        "prospective": ProspectiveAdapter,
        "ideal": IdealAdapter,
    }
    if name not in classes:
        raise ValueError(f"unknown adapter {name!r}; choose from {', '.join(classes)}")
    return classes[name](vocab=vocab, tau=tau, tau_a=tau_a, dt=dt, gain=gain)
