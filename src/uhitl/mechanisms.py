"""Transparency mechanisms: uncertainty estimate + normalisation, intent label, budgeted requests, novelty/risk gate."""

from collections import deque

import numpy as np


class UncertaintyEstimator:
    """Raw uncertainty / risk signal of the current state for the chosen method (higher = more likely to fail)."""

    def __init__(self, agent, method="risk", rng=None):
        # risk: predicted failure risk = -mean critic Q(s, a*) (best failure predictor, results/S1/quality_pretrain.json)
        # ensemble: critic disagreement (std); mcdropout: dropout std; random: control
        assert method in ("risk", "ensemble", "mcdropout", "random")
        self.agent, self.method = agent, method
        self.rng = rng or np.random.default_rng()

    def __call__(self, o):
        if self.method == "risk":
            return float(-np.mean(self.agent.q_values(o)))
        if self.method == "ensemble":
            return float(np.std(self.agent.q_values(o)))
        if self.method == "mcdropout":
            return float(np.std(self.agent.mc_dropout_q(o)))
        return float(self.rng.random())


class SlidingNormalizer:
    """Maps raw values to their percentile rank within a sliding window (empirical CDF).

    Rank normalisation gives every signal the same uniform [0, 1] distribution, so signals can only differ in how
    well they rank dangerous moments, not in their average level (min-max scaling confounded the two; see
    reports/night_timeline.txt, 18:23).
    """

    def __init__(self, init_values, window=2000, refresh=50):
        self.buf = deque(init_values, maxlen=window)
        self.refresh, self.count = refresh, 0
        self._update()

    def _update(self):
        self.sorted = np.sort(np.asarray(self.buf))

    def __call__(self, x):
        self.buf.append(x)
        self.count += 1
        if self.count % self.refresh == 0:
            self._update()
        lo = np.searchsorted(self.sorted, x, side="left")
        hi = np.searchsorted(self.sorted, x, side="right")
        return float((lo + hi) / 2 / len(self.sorted))


class IntentLabel:
    """Template intent: dominant axis of the smoothed deterministic action, as a unit vector."""

    def __init__(self, smoothing=3):
        self.hist = deque(maxlen=smoothing)

    def reset(self):
        self.hist.clear()

    def __call__(self, det_action):
        self.hist.append(det_action[:3])
        m = np.mean(self.hist, axis=0)
        v = np.zeros(3)
        if np.abs(m).max() > 1e-3:
            i = int(np.argmax(np.abs(m)))
            v[i] = np.sign(m[i])
        return v

    @staticmethod
    def text(v, gripper):
        names = {
            (0, 1): "forward",
            (0, -1): "back",
            (1, 1): "left",
            (1, -1): "right",
            (2, 1): "up",
            (2, -1): "down",
        }
        nz = np.flatnonzero(v)
        move = (
            f"Moving {names[(int(nz[0]), int(np.sign(v[nz[0]])))]}"
            if len(nz)
            else "Holding still"
        )
        grip = (
            " and closing gripper"
            if gripper > 1.5
            else (" and opening gripper" if gripper < 0.5 else "")
        )
        return move + grip


class BudgetedRequester:
    """Request help when normalised uncertainty stays above the (1-b) calibration quantile for k steps."""

    def __init__(self, calib_u, budget=0.2, k=3, release=0.8):
        self.tau = float(np.quantile(calib_u, 1 - budget))
        self.k, self.release = k, release
        self.reset()

    def reset(self):
        self.count, self.armed = 0, True

    def __call__(self, u):
        if u > self.tau:
            self.count += 1
        elif u < self.release * self.tau:
            self.count, self.armed = 0, True
        if self.armed and self.count >= self.k:
            self.armed = False
            return True
        return False


class NoveltyRiskGate:
    """ThriftyDAgger-style gate adapted to RL: novelty = distance to visited states, risk = low normalised Q."""

    def __init__(self, agent, calib_obs, calib_q, budget=0.2, k=3):
        self.agent = agent
        self.mu, self.sd = calib_obs.mean(0), calib_obs.std(0) + 1e-6
        self.ref = (calib_obs - self.mu) / self.sd
        nov = np.array(
            [self._novelty(o, loo=True) for o in calib_obs[:: max(1, len(calib_obs) // 500)]]
        )
        self.q_lo, self.q_hi = np.percentile(calib_q, 5), np.percentile(calib_q, 95)
        risk = 1 - np.clip(
            (calib_q - self.q_lo) / max(self.q_hi - self.q_lo, 1e-8), 0, 1
        )
        self.tau_n = float(np.quantile(nov, 1 - budget / 2))
        self.tau_r = float(np.quantile(risk, 1 - budget / 2))
        self.k = k
        self.reset()

    def _novelty(self, o, loo=False):
        x = (o - self.mu) / self.sd
        d = np.sqrt(((self.ref - x) ** 2).sum(1))
        return float(np.partition(d, 1)[1] if loo else d.min())  # loo: skip the point itself

    def add_reference(self, obs_batch):
        x = (obs_batch - self.mu) / self.sd
        self.ref = np.concatenate([self.ref, x])[-5000:]

    def reset(self):
        self.count, self.armed = 0, True

    def __call__(self, o):
        q = float(np.mean(self.agent.q_values(o)))
        risk = 1 - np.clip((q - self.q_lo) / max(self.q_hi - self.q_lo, 1e-8), 0, 1)
        hit = self._novelty(o) > self.tau_n or risk > self.tau_r
        self.count = self.count + 1 if hit else 0
        if not hit:
            self.armed = True
        if self.armed and self.count >= self.k:
            self.armed = False
            return True
        return False
