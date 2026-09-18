"""Method figures for the paper: system overview and simulated-supervisor components (no experiment data needed)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from uhitl.supervisor import SimulatedSupervisor, SupervisorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
TEAL, BLUE, AMBER, PURPLE, GREY = "#0A7470", "#2F6A9E", "#96570F", "#74509A", "#5A6874"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)


def _box(ax, xy, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            fc="white",
            ec=color,
            lw=1.2,
        )
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=7.5,
        weight="bold",
        color=color,
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h - 0.085,
        body,
        ha="center",
        va="top",
        fontsize=6.3,
        color="#15212A",
        linespacing=1.3,
    )


def _arrow(ax, a, b, text="", color=GREY, rad=0.0, dy=0.02):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=1.0,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + dy,
            text,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=color,
        )


def fig_system_overview():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top, h = 0.56, 0.4
    _box(ax, (0.015, top), 0.2, h, "Learning robot",
         "SAC + 5-critic ensemble\nBC loss on interventions\nPanda pick-cube (MuJoCo)", TEAL)
    _box(ax, (0.265, top), 0.2, h, "Transparency signal",
         "risk = -mean Q(s, a*)\npercentile-rank normalised\nu in [0, 1]", BLUE)
    _box(ax, (0.515, top), 0.2, h, "Mechanism",
         "none   risk display   intent display\nactive request   novelty gate\nrisk display + request", PURPLE)
    _box(ax, (0.765, top), 0.215, h, "Simulated supervisor",
         "skill (robomimic fit)\ndelay (take-over meta-analysis)\nattention, decision rule\ntrust (Guo & Yang 2020)", AMBER)
    _box(ax, (0.015, 0.04), 0.2, 0.34, "Replay buffers", "robot transitions\nintervention transitions\n(50/50 sampling)", TEAL)
    _box(ax, (0.765, 0.04), 0.215, 0.34, "Counterfactual judge",
         "restore MuJoCo snapshot\nroll out policy alone:\nwas the takeover necessary?", GREY)
    _arrow(ax, (0.215, 0.76), (0.265, 0.76), "Q", TEAL)
    _arrow(ax, (0.465, 0.76), (0.515, 0.76), "u(s)", BLUE)
    _arrow(ax, (0.715, 0.76), (0.765, 0.76), "", PURPLE)
    _arrow(ax, (0.872, 0.56), (0.872, 0.38), "", AMBER)
    ax.text(0.88, 0.47, "takeover\n(snapshot)", fontsize=6.2, color=AMBER, va="center")
    _arrow(ax, (0.765, 0.2), (0.215, 0.2), "executed actions (robot or supervisor)", AMBER)
    _arrow(ax, (0.115, 0.38), (0.115, 0.56), "", TEAL)
    ax.text(0.125, 0.47, "updates", fontsize=6.2, color=TEAL, va="center")
    ax.text(0.49, 0.47, "episode outcome (autonomous success?) updates supervisor trust", ha="center", fontsize=6.2,
            color=AMBER)
    fig.tight_layout()
    fig.savefig(FIG / "fig_system_overview.png")
    fig.savefig(FIG / "fig_system_overview.pdf")
    plt.close(fig)


def fig_supervisor_components():
    rng_seed = 0
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    # (a) reaction delay distributions
    ax = axes[0]
    for delay, c in zip((0.5, 1.5, 2.7), (TEAL, BLUE, AMBER)):
        sup = SimulatedSupervisor(
            SupervisorParams(delay_s=delay), "C1_none", seed=rng_seed
        )
        samples = np.array([sup._delay_steps() for _ in range(5000)]) * 0.1
        ax.hist(
            samples,
            bins=np.arange(0, 8, 0.2),
            histtype="step",
            color=c,
            lw=1.2,
            density=True,
            label=f"mean {delay} s",
        )
    ax.axvline(2.7, color=GREY, ls="--", lw=0.8)
    ax.text(
        2.8,
        ax.get_ylim()[1] * 0.9,
        "meta-analysis\nmean 2.7 s",
        fontsize=6,
        color=GREY,
        va="top",
    )
    ax.set_xlabel("Reaction delay (s)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Take-over delay", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    # (b) trust dynamics (Guo & Yang 2020, Eq. 3)
    ax = axes[1]
    outcomes = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    for wf, c in ((20.0, TEAL), (50.0, AMBER)):
        sup = SimulatedSupervisor(SupervisorParams(w_f=wf), "C1_none", seed=0)
        traj = [sup.trust]
        for o in outcomes:
            sup.end_episode(bool(o))
            traj.append(sup.trust)
        ax.plot(
            range(len(traj)),
            traj,
            "-o",
            ms=2.5,
            color=c,
            label=f"w_f = {wf:g}, w_s = 20",
        )
    for i, o in enumerate(outcomes, start=1):
        if not o:
            ax.axvspan(i - 0.5, i + 0.5, color="#F5E8D6", lw=0)
    ax.set_xlabel("Episode (shaded: robot needed help / failed)")
    ax.set_ylabel("Trust")
    ax.set_title("(b) Trust dynamics", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    # (c) attention vs trust
    ax = axes[2]
    trust = np.linspace(0, 1, 101)
    for gain, c in ((0.0, GREY), (0.25, BLUE), (0.5, TEAL)):
        p = np.clip(0.6 * (1 + gain * (0.5 - trust) * 2), 0.02, 1)
        ax.plot(trust, p, color=c, label=f"trust gain {gain}")
    ax.set_xlabel("Trust")
    ax.set_ylabel("P(attend) (base 0.6)")
    ax.set_title("(c) Trust lowers attention", fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_supervisor_components.png")
    fig.savefig(FIG / "fig_supervisor_components.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_system_overview()
    fig_supervisor_components()
    print("ok")
