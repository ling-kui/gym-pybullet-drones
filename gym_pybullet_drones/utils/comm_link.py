"""One-way communication link with delay, packet loss, and noise.

Models the inter-drone link used by formation control: the sender pushes a
state vector every control step; the receiver gets the sample delayed by
`delay_steps` steps, and each delivery independently fails with probability
`loss_prob` (on failure the receiver keeps its last successfully received
sample). Gaussian noise (std `noise_std`) is added to every delivered
sample. The link is deterministic for a given seed.

Note: faults live ONLY in this link. Every drone's own state (its onboard
sensors) stays perfect, matching the phase-3 experiment design where the
inter-drone channel—not the onboard sensing—is degraded.
"""
from collections import deque

import numpy as np


class CommLink:
    """One-way state link: sender -> [delay -> loss -> noise] -> receiver."""

    def __init__(self,
                 delay_steps: int = 0,
                 loss_prob: float = 0.0,
                 noise_std: float = 0.0,
                 seed: int = 0
                 ):
        self.delay = max(0, int(delay_steps))
        self.loss = min(1.0, max(0.0, float(loss_prob)))
        self.noise = max(0.0, float(noise_std))
        self.rng = np.random.default_rng(seed)
        self.buf = deque(maxlen=self.delay + 1)
        self.last_good = None  # (sent_step, value) of last delivered sample
        self.step = -1

    def prefill(self, value):
        """Fill the buffer with `value` so the receiver is not blind before
        the first real packet arrives (initial formation is known)."""
        for _ in range(self.delay + 1):
            self.send(value)

    def send(self, value):
        """Sender pushes the current state sample (called once per step)."""
        self.step += 1
        self.buf.append(np.array(value, dtype=float))

    def recv(self):
        """Receiver polls the link (called once per step).

        Returns
        -------
        (value | None, age_steps)
            `value` is the (possibly noisy) sample delayed by `delay_steps`,
            or None when this step's packet was lost. `age_steps` is how old
            the newest delivered sample is, so the receiver can extrapolate.
        """
        self.step += 1
        if self.rng.random() < self.loss:
            age = (self.step - self.last_good[0]) if self.last_good is not None else self.delay
            return None, age
        value = self.buf[0].copy()
        if self.noise > 0:
            value = value + self.rng.normal(0.0, self.noise, size=value.shape)
        self.last_good = (self.step - self.delay, value)
        return value, self.delay
