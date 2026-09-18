"""House style for all paper figures: palette, typography, helpers. Every figure is written as PNG (300 dpi) + PDF."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"

# condition palette: neutral baseline, teal family for displays, blue for requests, warm for the full method
C = {
    "C1_none": "#8C99A4",
    "C2_uncertainty": "#3E9B8F",
    "C3_intent": "#8B6BB1",
    "C4_request": "#2F6FA8",
    "C5_thrifty": "#B07A2B",
    "C6_full": "#C6522B",
    "HG_DAgger": "#C7CDD2",
}
SHORT = {
    "C1_none": "No transparency",
    "C2_uncertainty": "Risk display",
    "C3_intent": "Intent display",
    "C4_request": "Active request",
    "C5_thrifty": "Novelty gate",
    "C6_full": "Display + request",
    "HG_DAgger": "HG-DAgger",
}
# two-line variants for panels where the labels sit under vertical bars
STACK = {
    "C1_none": "No\ntransparency",
    "C2_uncertainty": "Risk\ndisplay",
    "C3_intent": "Intent\ndisplay",
    "C4_request": "Active\nrequest",
    "C5_thrifty": "Novelty\ngate",
    "C6_full": "Display\n+ request",
    "HG_DAgger": "HG-\nDAgger",
}
LONG = {
    "C1_none": "No transparency (baseline)",
    "C2_uncertainty": "Risk display",
    "C3_intent": "Intent display",
    "C4_request": "Active request",
    "C5_thrifty": "Novelty gate (ThriftyDAgger-style)",
    "C6_full": "Risk display + active request",
    "HG_DAgger": "HG-DAgger (interactive imitation)",
}
ORDER = [
    "C1_none",
    "C2_uncertainty",
    "C3_intent",
    "C4_request",
    "C5_thrifty",
    "C6_full",
    "HG_DAgger",
]
INK, MUTED, GRID = "#1B2430", "#5A6874", "#DCE3E8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "text.color": INK,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def grid(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def bars_with_ci(
    ax, keys, means, los, his, labels=None, annotate=None, width=0.68, fontsize=7
):
    x = np.arange(len(keys))
    for i, k in enumerate(keys):
        ax.bar(i, means[i], width=width, color=C[k], zorder=3, linewidth=0)
        ax.errorbar(
            i,
            means[i],
            yerr=[[means[i] - los[i]], [his[i] - means[i]]],
            color=INK,
            lw=0.8,
            capsize=2.2,
            zorder=4,
        )
        if annotate is not None and annotate[i]:
            ax.annotate(
                annotate[i],
                (i, his[i]),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=fontsize,
                color=C[k],
                weight="bold",
            )
    ax.set_xticks(x, labels or [SHORT[k] for k in keys])
    ax.set_xlim(-0.65, len(keys) - 0.35)
    grid(ax)


def hbars_with_ci(ax, keys, means, los, his, labels=None, annotate=None, height=0.68, fontsize=7):
    """Horizontal bars: condition names read on the y axis, which is what the long names need."""
    y = np.arange(len(keys))[::-1]
    for yi, i in zip(y, range(len(keys))):
        k = keys[i]
        ax.barh(yi, means[i], height=height, color=C[k], zorder=3, linewidth=0)
        ax.errorbar(means[i], yi, xerr=[[means[i] - los[i]], [his[i] - means[i]]],
                    color=INK, lw=0.8, capsize=2.2, zorder=4)
        if annotate is not None and annotate[i]:
            ax.annotate(annotate[i], (his[i], yi), textcoords="offset points", xytext=(4, -2.5),
                        fontsize=fontsize, color=C[k], weight="bold")
    ax.set_yticks(y, labels or [SHORT[k] for k in keys])
    ax.set_ylim(-0.65, len(keys) - 0.35)
    grid(ax, "x")
    return y


def legend_conditions(target, keys, ncol=4, loc="lower center", **kw):
    """Condition legend; pass a Figure to keep a wide legend inside the canvas."""
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=6, color=C[k], label=LONG[k])
        for k in keys
    ]
    return target.legend(handles=handles, ncol=ncol, loc=loc, **kw)


def save(fig, name, tight=True):
    FIG.mkdir(exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    return name


def pct(new, old):
    """Relative change, formatted for annotation (e.g. '+116%')."""
    if old == 0:
        return ""
    v = (new - old) / abs(old) * 100
    return f"{v:+.0f}%"
