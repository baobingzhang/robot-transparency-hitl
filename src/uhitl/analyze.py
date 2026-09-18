"""Aggregate finished runs into summary tables, paired statistics against C1, and figures."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uhitl.metrics import run_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
MECH_ORDER = [
    "C1_none",
    "C2_uncertainty",
    "C3_intent",
    "C4_request",
    "C5_thrifty",
    "C6_full",
    "HG_DAgger",
]
MECH_LABEL = {
    "C1_none": "No transparency",
    "C2_uncertainty": "Risk display",
    "C3_intent": "Intent display",
    "C4_request": "Active request",
    "C5_thrifty": "Novelty gate",
    "C6_full": "Display + request",
    "HG_DAgger": "HG-DAgger",
}
KEY_METRICS = [
    "final_success",
    "gain",
    "control_steps",
    "attend_steps",
    "takeovers",
    "unnecessary_takeovers",
    "precision",
    "recall",
    "trust_sq_error",
    "overtrust_fraction",
    "gain_per_100_control",
]


def collect(experiment):
    rows = []
    for done in sorted((ROOT / "results" / experiment).rglob("DONE")):
        d = done.parent
        try:
            m = run_metrics(d)
        except Exception as e:  # keep going; report broken runs
            print(f"skip {d}: {e}", file=sys.stderr)
            continue
        parts = d.relative_to(ROOT / "results" / experiment).parts
        m["group"] = parts[0]
        m["variant"] = parts[1] if len(parts) > 2 else ""
        rows.append(m)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(ROOT / f"results/summary_{experiment}.csv", index=False)
    return df


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_vs_c1(df, metric):
    """Paired differences (mechanism - C1) over matching (supervisor group, seed); sign-flip permutation p-values."""
    base = df[df["mechanism"] == "C1_none"].set_index(["group", "seed"])[metric]
    out = []
    for mech in MECH_ORDER[1:]:
        cur = df[df["mechanism"] == mech].set_index(["group", "seed"])[metric]
        diff = (cur - base).dropna()
        if len(diff) < 3:
            continue
        rng = np.random.default_rng(1)
        obs = diff.mean()
        signs = rng.choice([-1, 1], (10_000, len(diff)))
        p = float((np.abs((signs * diff.to_numpy()).mean(1)) >= abs(obs)).mean())
        lo, hi = bootstrap_ci(diff.to_numpy())
        out.append(
            {
                "mechanism": mech,
                "metric": metric,
                "n_pairs": len(diff),
                "mean_diff": obs,
                "ci_low": lo,
                "ci_high": hi,
                "p_perm": p,
            }
        )
    res = pd.DataFrame(out)
    if len(res):
        res["p_holm"] = holm(res["p_perm"])
    return res


def fig_s2_overview(df):
    metrics = [
        ("gain", "Learning gain (success)"),
        ("control_steps", "Supervisor control steps"),
        ("unnecessary_takeovers", "Unnecessary takeovers"),
        ("trust_sq_error", "Trust calibration error"),
    ]
    mechs = [m for m in MECH_ORDER if m in set(df["mechanism"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 2.8))
    colors = ["0.55", "#0A7470", "#74509A", "#2F6A9E", "#96570F", "0.8"]
    for ax, (m, title) in zip(axes, metrics):
        data = [df[df["mechanism"] == k][m].dropna().to_numpy() for k in mechs]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks(
            range(1, len(mechs) + 1),
            [MECH_LABEL[k].split(" ")[0] for k in mechs],
            fontsize=7,
        )
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_S2_overview.png")
    plt.close(fig)


def fig_s2_interaction(df):
    """C4 - C1 learning gain as a function of attention and reaction delay (supervisor population)."""
    wide = df.pivot_table(index=["group", "seed"], columns="mechanism", values="gain")
    if not {"C1_none", "C4_request"} <= set(wide.columns):
        return
    params = df.drop_duplicates("group").set_index("group")[
        ["sup_p_attend", "sup_delay_s"]
    ]
    diff = (wide["C4_request"] - wide["C1_none"]).groupby("group").mean()
    p = params.loc[diff.index]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    sc = ax.scatter(
        p["sup_p_attend"],
        p["sup_delay_s"],
        c=diff,
        cmap="PuOr",
        vmin=-0.3,
        vmax=0.3,
        s=40,
        edgecolor="0.3",
    )
    fig.colorbar(sc, ax=ax, label="Gain: active request vs no transparency")
    ax.set_xlabel("Attention probability")
    ax.set_ylabel("Reaction delay (s)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_S2_interaction_attention_delay.png")
    plt.close(fig)


def fig_s1_quality():
    q = json.loads((ROOT / "results/S1/quality_pretrain.json").read_text())
    hs = ["H5", "H10", "H20"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(hs))
    ax.bar(
        x - 0.18,
        [q[h][f"auroc_ens_std_fail_within_{h[1:]}"] for h in hs],
        0.36,
        color="#9ACBC7",
        edgecolor="#0A7470",
        hatch="//",
        label="ensemble disagreement",
    )
    ax.bar(
        x + 0.18,
        [q[h][f"auroc_q_mean_fail_within_{h[1:]}"] for h in hs],
        0.36,
        color="#0A7470",
        label="predicted risk (-mean Q)",
    )
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
    ax.set_xticks(x, [f"H = {h[1:]}" for h in hs])
    ax.set_xlabel("Failure within H steps")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 0.9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_S1_signal_quality.png")
    plt.close(fig)


def main(experiments):
    FIG.mkdir(exist_ok=True)
    if (ROOT / "results/S1/quality_pretrain.json").exists():
        fig_s1_quality()
    report = {}
    for exp in experiments:
        df = collect(exp)
        report[exp] = {"runs": len(df)}
        if not len(df):
            continue
        report[exp]["by_mechanism"] = (
            df.groupby("mechanism")[KEY_METRICS].mean().round(3).to_dict()
        )
        if exp == "S2":
            stats = pd.concat(
                [
                    paired_vs_c1(df, m)
                    for m in (
                        "gain",
                        "control_steps",
                        "unnecessary_takeovers",
                        "precision",
                        "trust_sq_error",
                    )
                ],
                ignore_index=True,
            )
            stats.to_csv(ROOT / "results/stats_S2_vs_C1.csv", index=False)
            fig_s2_overview(df)
            fig_s2_interaction(df)
    (ROOT / "results/analysis_summary.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(json.dumps({k: v["runs"] for k, v in report.items()}))


if __name__ == "__main__":
    main(sys.argv[1:] or ["S1", "S2", "S3", "S4"])
