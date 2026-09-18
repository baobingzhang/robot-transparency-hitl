"""Figures for the progress report and the paper (written to figures/)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def fig_skill_fit():
    fit = json.loads((ROOT / "data/robomimic/skill_fit.json").read_text())["fits"]
    stats = ("length", "path_ratio", "dir_noise")
    labels = ("Episode length", "Path ratio", "Direction noise")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, level in zip(axes, ("okay", "worse")):
        x = np.arange(len(stats))
        ax.bar(
            x - 0.18,
            [fit[level]["robomimic_ratio_to_better"][s] for s in stats],
            0.36,
            label="robomimic humans",
            color="#2F6A9E",
        )
        ax.bar(
            x + 0.18,
            [fit[level]["sim_ratio_to_better"][s] for s in stats],
            0.36,
            label="simulated supervisor",
            color="#9ACBC7",
            hatch="//",
            edgecolor="#0A7470",
        )
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(x, labels)
        ax.set_title(
            f"'{level}' operators (gain {fit[level]['gain']}, noise {fit[level]['noise']})"
        )
    axes[0].set_ylabel("Ratio to 'better' operators")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_skill_fit.png")
    plt.close(fig)


def fig_learner_diagnostics():
    runs = {
        "SAC, sparse (T0=1)": "logs/diag_sac_sparse.log",
        "SAC, sparse (T0=0.01)": "logs/diag_sac_sparse_t001.log",
        "SAC, dense (T0=0.01)": "logs/diag_sac_dense_t001.log",
        "SAC + BC, sparse (T0=0.01)": "logs/diag_sac_sparse_bc.log",
    }
    styles = ["-", "--", "-.", "-"]
    colors = ["0.6", "#2F6A9E", "#74509A", "#0A7470"]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for (name, path), ls, c in zip(runs.items(), styles, colors):
        pts = []
        for line in (ROOT / path).read_text().splitlines():
            if line.startswith("("):
                step, succ = line.split(")")[0].strip("(").split(",")
                pts.append((int(step), float(succ)))
        if pts:
            s, v = zip(*pts)
            ax.plot(s, v, ls, color=c, marker="o", ms=3, label=name)
    ax.set_xlabel("Environment steps (20 demos, no supervisor)")
    ax.set_ylabel("Autonomous success (20 ep.)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_learner_diagnostics.png")
    plt.close(fig)


def fig_pretrain():
    d = json.loads((ROOT / "results/pretrain/pretrain_curve.json").read_text())
    s = [c["step"] for c in d["curve"]]
    v = [c["success_20"] for c in d["curve"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhspan(*d["band"], color="#DCEFEC", label="target band")
    ax.plot(s, v, "-o", ms=3, color="#0A7470", label="20-episode eval")
    full = [(c["step"], c["success_100"]) for c in d["curve"] if "success_100" in c]
    if full:
        ax.plot(*zip(*full), "D", color="#96570F", ms=5, label="100-episode eval")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Autonomous success")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pretrain.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_skill_fit()
    fig_learner_diagnostics()
    if (ROOT / "results/pretrain/pretrain_curve.json").exists():
        fig_pretrain()
    print(sorted(p.name for p in FIG.glob("*.png")))
