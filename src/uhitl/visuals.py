"""Illustrative renders of the simulated task and of the counterfactual necessity judge.

These figures illustrate the setting and the method; they carry no new measurements. Everything is produced
from the frozen pre-trained checkpoint so that the pictures match the state the learner starts every run in.

Output: figures/visual/
"""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # otherwise the rollouts are not bit-reproducible between runs

from uhitl.agent import SACAgent  # noqa: E402
from uhitl.envs import Snapshot, flat_obs, make_env  # noqa: E402
from uhitl.mechanisms import (  # noqa: E402
    BudgetedRequester,
    IntentLabel,
    SlidingNormalizer,
    UncertaintyEstimator,
)
from uhitl.style import MUTED, C, plt, save  # noqa: E402
from uhitl.supervisor import SimulatedSupervisor, SupervisorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/visual"
MAX_T = 100
W, H = 900, 660


def camera(model, azimuth=120.0, elevation=-19.0, distance=1.24):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = [0.28, 0.0, 0.20]
    return cam


class Scene:
    """High-resolution offscreen renderer for the pick-cube environment."""

    def __init__(self, env, width=W, height=H):
        m = env.unwrapped._model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.env, self.model = env, m
        self.renderer = mujoco.Renderer(m, height=height, width=width)
        self.cam = camera(m)

    def frame(self):
        self.renderer.update_scene(self.env.unwrapped._data, camera=self.cam)
        return self.renderer.render()


def load_agent(device="cpu"):
    import torch

    agent = SACAgent(num_critics=5, device=device, bc_weight=1.0, seed=0)
    path = ROOT / "results/pretrain/checkpoint.pt"
    if not path.exists():  # the release ships the policy weights without the replay buffers
        path = ROOT / "results/pretrain/policy_weights.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(ck["agent"])
    return agent


def calibrate(agent, env, episodes=8):
    unc = UncertaintyEstimator(agent, method="risk")
    raw = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=500_000 + ep)
        o = flat_obs(obs)
        for _ in range(MAX_T):
            raw.append(unc(o))
            obs, _, term, _, _ = env.step(agent.act(o, deterministic=True))
            o = flat_obs(obs)
            if term:
                break
    norm = SlidingNormalizer(raw)
    return unc, norm, np.array([norm(x) for x in raw])


def run_episodes(agent, env, scene, unc, norm, calib_u, n_episodes=16, seed=7):
    """One supervised interaction under display + request, keeping a frame for every step."""
    sup = SimulatedSupervisor(
        SupervisorParams(
            skill_gain=0.8,
            skill_noise=0.5,
            delay_s=1.5,
            p_attend=0.6,
            theta=0.55,
            display_weight=0.5,
            reliance=0.3,
        ),
        mechanism="C6_full",
        seed=seed,
    )
    requester = BudgetedRequester(calib_u, budget=0.2)
    intent = IntentLabel()
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=90_000 + ep)
        o = flat_obs(obs)
        sup.reset_episode()
        requester.reset()
        intent.reset()
        rec = {
            "frames": [],
            "u": [],
            "interv": [],
            "request": [],
            "takeovers": [],
            "success": False,
        }
        prev = False
        for t in range(MAX_T):
            det = agent.act(o, deterministic=True)
            u = norm(unc(o))
            signals = {"u": u, "intent": intent(det), "request": requester(u)}
            a_exec, is_interv, log = sup.step(o, agent.act(o), signals)
            if is_interv and not prev:
                rec["takeovers"].append(
                    {"t": t, "u": u, "snapshot": Snapshot(env), "source": log.source}
                )
            prev = is_interv
            rec["frames"].append(scene.frame())
            rec["u"].append(u)
            rec["interv"].append(bool(is_interv))
            rec["request"].append(bool(signals["request"]))
            obs, _, term, _, info = env.step(a_exec)
            o = flat_obs(obs)
            if term:
                rec["success"] = bool(info.get("succeed", False))
                break
        episodes.append(rec)
    return episodes


def rollout_alone(agent, env, scene, snap, t_now, horizon=50):
    """Restore the exact state and let the policy act alone, keeping the frames."""
    o = snap.restore(env)
    frames, success = [scene.frame()], False
    for _ in range(min(MAX_T - t_now, horizon)):
        obs, _, term, _, info = env.step(agent.act(o, deterministic=True))
        o = flat_obs(obs)
        frames.append(scene.frame())
        if term:
            success = bool(info.get("succeed", False))
            break
    snap.restore(env)
    return frames, success


def pick(frames, n):
    """Up to n evenly spaced frames, never repeating one when the rollout is shorter than n."""
    idx = np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
    return [frames[i] for i in idx], idx


def strip(ax_list, frames, idx, labels=None, edge=None):
    for k, (ax, f) in enumerate(zip(ax_list, frames)):
        ax.imshow(f)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(edge or "#D3DADF")
            s.set_linewidth(1.6 if edge else 0.8)
        if labels is not None:
            ax.set_title(labels[k], fontsize=7.5, color=MUTED, pad=3)


# ------------------------------------------------------------------ Fig: the task
def fig_task(scene, env):
    env.reset(seed=90_000)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.imshow(scene.frame())
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#D3DADF")
    return save(fig, "visual/visual1_task")


# ------------------------------------------------------------------ Fig: one supervised episode
def fig_filmstrip(episodes):
    ep = max(episodes, key=lambda e: (e["success"], len(e["takeovers"])))
    frames, idx = pick(ep["frames"], 6)
    fig, axes = plt.subplots(1, 6, figsize=(7.4, 1.65))
    who = ["supervisor" if ep["interv"][i] else "robot" for i in idx]
    strip(axes, frames, idx, [f"step {i}\n{w}" for i, w in zip(idx, who)])
    for ax, w in zip(axes, who):
        for s in ax.spines.values():
            s.set_color(C["C6_full"] if w == "supervisor" else C["C1_none"])
            s.set_linewidth(1.8)
    return save(fig, "visual/visual2_episode")


# ------------------------------------------------------------------ Fig: the counterfactual judge
def fig_counterfactual(agent, env, scene, episodes):
    cases = []
    for ep in episodes:
        for tk in ep["takeovers"]:
            frames, ok = rollout_alone(agent, env, scene, tk["snapshot"], tk["t"])
            cases.append({"ok": ok, "frames": frames, "t": tk["t"], "u": tk["u"]})
    print("  counterfactual cases:", len(cases), "| policy succeeded alone:", sum(c["ok"] for c in cases))
    nec = next((c for c in cases if not c["ok"]), None)
    unnec = next((c for c in cases if c["ok"]), None)
    rows = [
        (nec, "fails", "NECESSARY take-over", C["C6_full"]),
        (unnec, "succeeds", "UNNECESSARY take-over", C["C2_uncertainty"]),
    ]
    rows = [r for r in rows if r[0] is not None]
    fig, axes = plt.subplots(
        len(rows), 5,
        figsize=(7.4, 1.45 * len(rows) + 0.3),
        gridspec_kw={"hspace": 0.45, "wspace": 0.06},
    )
    axes = np.atleast_2d(axes)
    for r, (case, what, verdict, col) in enumerate(rows):
        frames, idx = pick(case["frames"], 5)
        labels = [
            "state restored"
            if k == 0
            else ("+1 step" if i == 1 else f"+{i} steps")
            for k, i in enumerate(idx)
        ]
        strip(axes[r][: len(frames)], frames, idx, labels)
        for ax in axes[r][len(frames) :]:
            ax.set_visible(False)
        for ax in axes[r][: len(frames)]:
            for s in ax.spines.values():
                s.set_color(col)
                s.set_linewidth(1.6)
        axes[r][0].set_ylabel(
            f"{verdict}\npolicy alone {what}",
            fontsize=8,
            color=col,
            weight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=6,
        )
    return save(fig, "visual/visual3_counterfactual", tight=False)


# ------------------------------------------------------------------ Fig: risk trace of one episode
def fig_timeline(episodes, calib_u):
    ep = max(episodes, key=lambda e: len(e["takeovers"]))
    u = np.array(ep["u"])
    tau = float(np.quantile(calib_u, 0.8))
    fig, ax = plt.subplots(figsize=(7.1, 2.2))
    t = np.arange(len(u))
    interv = np.array(ep["interv"])
    start = None
    for i in range(len(interv)):
        if interv[i] and start is None:
            start = i
        if start is not None and (not interv[i] or i == len(interv) - 1):
            ax.axvspan(
                start - 0.5,
                i - 0.5,
                color=C["C6_full"],
                alpha=0.13,
                zorder=1,
                label="supervisor in control" if start == np.argmax(interv) else None,
            )
            start = None
    ax.plot(
        t, u, color=C["C2_uncertainty"], lw=1.5, zorder=3, label="displayed risk $u(s)$"
    )
    ax.axhline(
        tau, color=MUTED, lw=0.9, ls="--", zorder=2, label="request threshold $\\tau$"
    )
    req = t[np.array(ep["request"], dtype=bool)]
    if len(req):
        ax.plot(
            req,
            u[req],
            "v",
            ms=6,
            color=C["C4_request"],
            zorder=4,
            label="robot asks for help",
        )
    ax.set_xlabel("Step within the episode")
    ax.set_ylabel("Normalised risk")
    ax.set_ylim(0, 1.32)
    ax.set_xlim(0, len(u) - 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", fontsize=7, ncol=4, columnspacing=1.0, handletextpad=0.5)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    return save(fig, "visual/visual4_timeline")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(seed=0)
    env.reset(seed=0)
    scene = Scene(env)
    agent = load_agent()
    unc, norm, calib_u = calibrate(agent, env)
    episodes = run_episodes(agent, env, scene, unc, norm, calib_u, n_episodes=24)
    print("episodes:", [(e["success"], len(e["takeovers"])) for e in episodes])
    print(fig_task(scene, env))
    print(fig_filmstrip(episodes))
    print(fig_counterfactual(agent, env, scene, episodes))
    print(fig_timeline(episodes, calib_u))


if __name__ == "__main__":
    main()
