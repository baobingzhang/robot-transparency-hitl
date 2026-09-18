"""Simulated human supervisor: operator skill, reaction delay, attention, decision rule, trust dynamics.

Grounding (see data/literature/supervisor_parameters.yaml):
  skill          fitted to robomimic multi-human operator statistics (worse / okay / better)
  reaction delay swept up to the 2.7 s mean take-over time of Zhang et al. (2019); auditory requests are answered faster
  trust          Beta-distribution trust dynamics of Guo & Yang (2020), updated after every episode
Attention, cue weighting and reliance on requests are modelling assumptions and are swept in the experiments.
"""

from dataclasses import dataclass

import numpy as np

from uhitl.envs import block_pos, gripper_ctrl, tcp_pos

HOVER = 0.06
MECHANISMS = ("C1_none", "C2_uncertainty", "C3_intent", "C4_request", "C5_thrifty", "C6_full")
REQUEST_MECHS = ("C4_request", "C5_thrifty", "C6_full")  # robot may ask for help
DISPLAY_MECHS = ("C2_uncertainty", "C6_full")  # supervisor sees the risk signal


# ---------------------------------------------------------------- scripted operator (skill model)


class ScriptedOperator:
    """Latched phase controller (approach, descend, close, lift); skill = speed gain and Gaussian command noise.

    The phase is re-inferred from the scene whenever control is (re)taken, so it can start mid-episode.
    """

    def __init__(self, gain=1.0, noise=0.05, rng=None):
        self.gain, self.noise = gain, noise
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.phase, self.close_steps, self.anchor = None, 0, None

    @staticmethod
    def _holding(o):
        return gripper_ctrl(o) > 200 and np.linalg.norm(block_pos(o) - tcp_pos(o)) < 0.03

    def sync(self, o):
        """Infer the phase from the current scene (called when the operator takes control)."""
        t, b = tcp_pos(o), block_pos(o)
        self.close_steps, self.anchor = 0, b.copy()
        if self._holding(o):
            self.phase = "lift"
        elif gripper_ctrl(o) > 50:
            self.phase = "release"
        elif np.linalg.norm((b - t)[:2]) < 0.012 and t[2] - b[2] < HOVER:
            self.phase = "descend"
        else:
            self.phase = "approach"

    def ideal_direction(self, o):
        """Unit direction a competent operator would move the end effector in (the visual cue)."""
        t, b = tcp_pos(o), block_pos(o)
        if self._holding(o):
            return np.array([0.0, 0.0, 1.0])
        tgt = b + np.array([0, 0, HOVER]) if np.linalg.norm((b - t)[:2]) > 0.012 else b
        d = tgt - t
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.zeros(3)

    def act(self, o):
        if self.phase is None:
            self.sync(o)
        t, b = tcp_pos(o), block_pos(o)
        if self.phase in ("descend", "close") and np.linalg.norm(b[:2] - self.anchor[:2]) > 0.03:
            self.sync(o)  # block was pushed away: start over
        a = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        if self.phase == "release":
            a[:3] = [0.0, 0.0, 1.0]
            a[3] = 0.0
            if gripper_ctrl(o) < 50:
                self.phase = "approach"
        elif self.phase == "approach":
            a[:3] = np.clip((b + [0, 0, HOVER] - t) / 0.025, -1, 1)
            a[3] = 0.0
            if np.linalg.norm((b - t)[:2]) < 0.012 and abs(t[2] - b[2] - HOVER) < 0.02:
                self.phase, self.anchor = "descend", b.copy()
        elif self.phase == "descend":
            a[:3] = np.clip((b - t) / 0.025, -1, 1)
            a[3] = 0.0
            if t[2] - b[2] < 0.01:
                self.phase = "close"
        elif self.phase == "close":
            a[3] = 2.0
            self.close_steps += 1
            if self.close_steps >= 4 and gripper_ctrl(o) > 200:
                self.phase = "lift"
        else:  # lift
            a[:3] = [0.0, 0.0, 1.0]
            a[3] = 2.0
            if gripper_ctrl(o) > 200 and np.linalg.norm(b - t) > 0.05:
                self.sync(o)  # dropped the block
        a[:3] = np.clip(a[:3] * self.gain + self.rng.normal(0, self.noise, 3), -1, 1)
        return a


# ---------------------------------------------------------------- supervisor


@dataclass
class SupervisorParams:
    skill_gain: float = 1.0
    skill_noise: float = 0.05
    delay_s: float = 1.5  # mean reaction time for visually detected problems
    request_delay_factor: float = (
        0.7  # auditory requests answered faster (direction from Zhang et al. 2019)
    )
    p_attend: float = 0.6
    theta: float = 0.55  # decision threshold on the combined cue
    detect_noise: float = 0.15  # noise on visually judged state cues
    display_weight: float = 0.5  # reliance on the displayed signal (C2, C3)
    reliance: float = (
        0.0  # attention drop when the robot is known to request help (C4, C5)
    )
    hold_min: int = 8
    hold_max: int = 20
    cooldown: int = 5
    trust_enabled: bool = True
    alpha0: float = 100.0
    beta0: float = 50.0
    w_s: float = 20.0
    w_f: float = 50.0
    trust_attention_gain: float = 0.5  # how strongly trust lowers attention
    control_dt: float = 0.1


@dataclass
class SupervisorLog:
    attend: bool = False
    cue: float = 0.0
    trigger: bool = False
    pending: int = 0
    trust: float = 0.0
    source: str = ""  # at takeover start: "request" (robot asked) or "self" (supervisor noticed)


class SimulatedSupervisor:
    def __init__(self, params: SupervisorParams, mechanism: str, seed: int = 0):
        assert mechanism in MECHANISMS
        self.p, self.mech = params, mechanism
        self.rng = np.random.default_rng(seed)
        self.op = ScriptedOperator(params.skill_gain, params.skill_noise, self.rng)
        self.alpha, self.beta = params.alpha0, params.beta0
        self.mis_mean = None  # long-run average intent mismatch (C3)
        self.reset_episode()

    # ------------------------------------------------------------ episode bookkeeping
    def reset_episode(self):
        self.op.reset()
        self.intervening = False
        self.hold = 0
        self.pending = -1
        self.pending_source = ""
        self.cool = 0
        self.intervened_this_episode = False
        self.recent_state_cue = []

    @property
    def trust(self):
        return self.alpha / (self.alpha + self.beta)

    def end_episode(self, autonomous_success: bool):
        """Guo & Yang (2020) Eq. (3): robot performance p_i = 1 only if it succeeded without help."""
        if not self.p.trust_enabled:
            return
        if autonomous_success:
            self.alpha += self.p.w_s
        else:
            self.beta += self.p.w_f

    # ------------------------------------------------------------ cues
    def _delay_steps(self, factor=1.0):
        mean = self.p.delay_s * factor / self.p.control_dt
        sigma = 0.5  # lognormal shape; SD grows with the mean (mean-SD correlation in the meta-analysis)
        return max(
            1, int(round(self.rng.lognormal(np.log(mean) - sigma**2 / 2, sigma)))
        )

    def state_cue(self, o, agent_action):
        """How wrong the robot's motion looks: disagreement between commanded and ideal direction."""
        ideal = self.op.ideal_direction(o)
        cmd = agent_action[:3]
        n = np.linalg.norm(cmd)
        mis = (
            0.5
            if n < 1e-3 or np.linalg.norm(ideal) < 1e-6
            else (1 - float(np.dot(cmd / n, ideal))) / 2
        )
        self.recent_state_cue = (self.recent_state_cue + [mis])[-3:]
        return float(np.mean(self.recent_state_cue))

    def _attention_prob(self):
        p = self.p.p_attend
        if self.p.trust_enabled:
            p *= 1 + self.p.trust_attention_gain * (0.5 - self.trust) * 2
        if self.mech in REQUEST_MECHS:
            p *= 1 - self.p.reliance
        return float(np.clip(p, 0.02, 1.0))

    # ------------------------------------------------------------ main step
    def step(self, o, agent_action, signals):
        """Return (executed_action, is_intervention, log). signals: u (0-1), intent (3-vector), request (bool)."""
        log = SupervisorLog(trust=self.trust)
        if self.intervening:
            self.hold -= 1
            if self.hold <= 0:
                self.intervening, self.cool = False, self.p.cooldown
            else:
                return self.op.act(o), True, log
        if self.pending >= 0:
            self.pending -= 1
            log.pending = self.pending
            if self.pending < 0:
                self.intervening = True
                self.intervened_this_episode = True
                self.hold = int(self.rng.integers(self.p.hold_min, self.p.hold_max + 1))
                self.op.sync(o)
                log.source = self.pending_source
                return self.op.act(o), True, log
            return agent_action, False, log
        if self.cool > 0:
            self.cool -= 1
            return agent_action, False, log

        if self.mech in REQUEST_MECHS and signals.get("request", False):
            log.trigger = True
            self.pending = self._delay_steps(self.p.request_delay_factor)
            self.pending_source = "request"
            return agent_action, False, log

        log.attend = bool(self.rng.random() < self._attention_prob())
        if log.attend:
            cue = self.state_cue(o, agent_action) + self.rng.normal(
                0, self.p.detect_noise
            )
            # displays shift the supervisor's judgement relative to the signal's usual level (level-neutral), so a
            # display can only redistribute interventions towards riskier moments, not raise the overall rate
            w = self.p.display_weight
            if self.mech in DISPLAY_MECHS:
                cue = cue + w * (signals["u"] - 0.5)  # u is a percentile rank: mean 0.5
            elif self.mech == "C3_intent":
                ideal = self.op.ideal_direction(o)
                mis = (
                    0.5
                    if np.linalg.norm(ideal) < 1e-6
                    else (1 - float(np.dot(signals["intent"], ideal))) / 2
                )
                self.mis_mean = mis if self.mis_mean is None else 0.99 * self.mis_mean + 0.01 * mis
                cue = cue + w * (mis - self.mis_mean)
            theta = self.p.theta
            if self.p.trust_enabled:
                theta += 0.2 * (self.trust - 0.5)
            log.cue = float(cue)
            if cue > theta:
                log.trigger = True
                self.pending = self._delay_steps()
                self.pending_source = "self"
        return agent_action, False, log
