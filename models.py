"""NumPy CPU autoregressive teacher and a distilled small neural drafter.

No pretrained language model or external training data is used. Persistent
prefix memory is separate from the transmission adapter. Draft training uses
ordinary supervised logit distillation with manually differentiated Adam; it
does not reproduce the NLA or teaching-signal learning rules.
"""

from dataclasses import dataclass
import math

import numpy as np


VOCAB = 16
MEMORY_DIM = 16
INPUT_DIM = 24


@dataclass
class Context:
    embeddings: np.ndarray
    memory: np.ndarray
    decay: float
    index: int
    phase: float

    def features(self):
        angles = np.array([0.037, 0.083, 0.137, 0.211]) * self.index + self.phase
        return np.concatenate((self.memory, np.sin(angles), np.cos(angles))).astype(np.float32)

    def advance(self, token):
        self.memory = self.decay * self.memory + (1.0 - self.decay) * self.embeddings[token]
        self.index += 1

    def clone(self):
        return Context(self.embeddings, self.memory.copy(), self.decay, self.index, self.phase)


def make_embeddings(seed):
    return np.random.default_rng(seed + 1000).normal(0, 0.7, (VOCAB, MEMORY_DIM))


def make_context(embeddings, decay, seed):
    rng = np.random.default_rng(seed)
    context = Context(embeddings, np.zeros(MEMORY_DIM), decay, 0, float(rng.uniform(0, 2 * np.pi)))
    for token in rng.integers(VOCAB, size=64):
        context.advance(int(token))
    return context


def training_inputs(embeddings, n, seed):
    """Generate causal-prefix feature coverage using independent data seeds.

    Every memory is a convex combination of token embeddings. Half of the
    examples use a single embedding to cover abrupt, effectively memoryless
    contexts; the others cover mixed prefix histories.
    """
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(VOCAB, 0.12), size=n)
    memories = weights @ embeddings
    abrupt = rng.random(n) < 0.5
    memories[abrupt] = embeddings[rng.integers(VOCAB, size=int(abrupt.sum()))]
    indices = rng.integers(0, 100000, size=n)
    phases = rng.uniform(0, 2 * np.pi, (n, 1))
    angles = indices[:, None] * np.array([0.037, 0.083, 0.137, 0.211])[None, :] + phases
    return np.concatenate((memories, np.sin(angles), np.cos(angles)), axis=1).astype(np.float32)


class NeuralTarget:
    """Frozen random tanh MLP: 24 -> width -> width -> 16, no biases.

    Weights use row-input orientation and float32 to support ordinary BLAS.
    Output calibration is folded into the final weight matrix, so the weights
    fully specify the target evaluated by any optional external backend.
    """

    def __init__(self, seed, width=2048):
        if width < 1:
            raise ValueError("target width must be positive")
        rng = np.random.default_rng(seed)
        dimensions = (INPUT_DIM, int(width), int(width), VOCAB)
        self.weights = [
            np.ascontiguousarray(rng.normal(0, 1 / math.sqrt(n_in), (n_in, n_out)), dtype=np.float32)
            for n_in, n_out in zip(dimensions[:-1], dimensions[1:])
        ]
        self.logit_scale = 1.0

    def evaluate(self, inputs):
        w1, w2, w3 = self.weights
        x = np.asarray(inputs, dtype=np.float32)
        return np.tanh(np.tanh(x @ w1) @ w2) @ w3

    __call__ = evaluate

    @property
    def parameter_count(self):
        return sum(w.size for w in self.weights)

    def calibrate(self, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("calibration scale must be positive and finite")
        self.weights[-1] *= np.float32(scale)
        self.logit_scale *= float(scale)


class NumpyDraft:
    """Small tanh drafter with vector or batched CPU inference."""

    def __init__(self, seed=0, width=64):
        rng = np.random.default_rng(seed + 20000)
        # Retain output-by-input orientation in saved draft checkpoints.
        self.w1 = rng.uniform(-1 / math.sqrt(INPUT_DIM), 1 / math.sqrt(INPUT_DIM), (width, INPUT_DIM))
        self.b1 = np.zeros(width, dtype=np.float64)
        self.w2 = rng.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width), (VOCAB, width))
        self.b2 = np.zeros(VOCAB, dtype=np.float64)

    def __call__(self, inputs):
        x = np.asarray(inputs, dtype=np.float64)
        return np.tanh(x @ self.w1.T + self.b1) @ self.w2.T + self.b2

    @property
    def parameters(self):
        return (self.w1, self.b1, self.w2, self.b2)

    @property
    def parameter_count(self):
        return sum(p.size for p in self.parameters)

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: np.array(stored[key], dtype=np.float64, copy=True)
                      for key in ("w1", "b1", "w2", "b2")}
        width = arrays["b1"].shape[0]
        expected = {"w1": (width, INPUT_DIM), "b1": (width,),
                    "w2": (VOCAB, width), "b2": (VOCAB,)}
        if any(arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
               for key, shape in expected.items()):
            raise ValueError("invalid draft checkpoint arrays")
        draft = cls(width=width)
        for key, value in arrays.items():
            setattr(draft, key, value)
        return draft

    def loss_and_gradients(self, inputs, centered_targets):
        """Centered-logit MSE and its analytic gradients, in parameter order."""
        x = np.asarray(inputs, dtype=np.float64)
        hidden = np.tanh(x @ self.w1.T + self.b1)
        output = hidden @ self.w2.T + self.b2
        output -= output.mean(axis=1, keepdims=True)
        error = output - centered_targets
        loss = float(np.mean(error ** 2))
        delta = (2.0 / error.size) * error
        # Differentiate the logit-centering operation explicitly.
        delta -= delta.mean(axis=1, keepdims=True)
        grad_w2 = delta.T @ hidden
        grad_b2 = delta.sum(axis=0)
        hidden_delta = (delta @ self.w2) * (1.0 - hidden ** 2)
        grad_w1 = hidden_delta.T @ x
        grad_b1 = hidden_delta.sum(axis=0)
        return loss, (grad_w1, grad_b1, grad_w2, grad_b2)


def synchronize(device):
    """NumPy evaluation is synchronous; its return is the timing boundary."""
    return None


class TargetEvaluator:
    def __init__(self, model, device="cpu", backend=None):
        if device != "cpu" and backend is None:
            raise ValueError("a non-CPU target needs an explicit synchronous backend")
        self.model = model
        self.device = device
        self.backend = backend

    def logits(self, inputs, batch_size=512):
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
            raise ValueError(f"expected input matrix with {INPUT_DIM} columns")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(inputs) == 0:
            return np.empty((0, VOCAB), dtype=np.float64)
        arrays = []
        evaluator = self.model.evaluate if self.backend is None else self.backend
        for start in range(0, len(inputs), batch_size):
            result = evaluator(np.ascontiguousarray(inputs[start:start + batch_size]))
            arrays.append(np.asarray(result, dtype=np.float64))
        return np.concatenate(arrays)

    def __call__(self, inputs):
        logits = self.logits(inputs)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def train_pair(seed, device="cpu", width=2048, train_size=8192, steps=1600):
    """Distill a frozen synthetic target using minibatch NumPy Adam on CPU."""
    if device != "cpu":
        raise ValueError("train_pair uses the NumPy CPU implementation")
    if train_size < 1 or steps < 0:
        raise ValueError("train_size must be positive and steps nonnegative")
    embeddings = make_embeddings(seed)
    train_x = training_inputs(embeddings, train_size, seed + 40000)
    valid_x = training_inputs(embeddings, 2048, seed + 50000)
    target_model = NeuralTarget(seed, width)
    target = TargetEvaluator(target_model, "cpu")
    raw = target.logits(train_x[:1024])
    scale = 1.5 / max(float(np.std(raw - raw.mean(axis=1, keepdims=True))), 1e-12)
    target_model.calibrate(scale)
    train_logits = target.logits(train_x)
    valid_logits = target.logits(valid_x)
    y = train_logits - train_logits.mean(axis=1, keepdims=True)
    x = train_x.astype(np.float64)
    draft = NumpyDraft(seed)
    parameters = draft.parameters
    first_moments = [np.zeros_like(p) for p in parameters]
    second_moments = [np.zeros_like(p) for p in parameters]
    rng = np.random.default_rng(seed + 60000)
    initial_validation = draft(valid_x)
    initial_validation -= initial_validation.mean(axis=1, keepdims=True)
    valid_centered = valid_logits - valid_logits.mean(axis=1, keepdims=True)
    initial_mse = float(np.mean((initial_validation - valid_centered) ** 2))
    final_loss = None
    for step in range(1, steps + 1):
        indices = rng.integers(train_size, size=256)
        final_loss, gradients = draft.loss_and_gradients(x[indices], y[indices])
        for parameter, gradient, first, second in zip(parameters, gradients, first_moments, second_moments):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient ** 2
            m_hat = first / (1.0 - 0.9 ** step)
            v_hat = second / (1.0 - 0.999 ** step)
            parameter -= 0.003 * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_out = draft(valid_x)
    val_out -= val_out.mean(axis=1, keepdims=True)
    metrics = {
        "validation_logit_mse": float(np.mean((val_out - valid_centered) ** 2)),
        "initial_validation_logit_mse": initial_mse,
        "final_minibatch_mse": final_loss,
        "teacher_parameters": target_model.parameter_count,
        "draft_parameters": draft.parameter_count,
        "teacher_logit_scale": float(scale),
        "train_size": train_size,
        "training_steps": steps,
        "training_method": "NumPy analytic-gradient Adam, centered-logit MSE",
    }
    return target, draft, embeddings, metrics
