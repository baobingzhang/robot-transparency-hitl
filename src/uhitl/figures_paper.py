"""Publication figures built with the house style (src/uhitl/style.py). Numbers come from results/processed + tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from uhitl.analyze import bootstrap_ci
from uhitl.style import plt  # noqa: E402
from uhitl.style import (
    INK,
    MUTED,
    ORDER,
    SHORT,
    STACK,
    C,
    bars_with_ci,
    grid,
    hbars_with_ci,
    legend_conditions,
    pct,
    save,
)

ROOT = Path(__file__).resolve().parents[2]
SKILL_LABEL = {
    "worse": "less-skilled supervisors",
    "okay": "average supervisors",
    "better": "skilled supervisors",
}


def s2():
    d = pd.read_csv(ROOT / "results/processed/S2_runs.csv")
    d["skill"] = d["group"].str.extract(r"(worse|okay|better)")[0]
    return d


def stat_cfg(cfg, mech, n_boot=10000, seed=0):
    """Mean and 95% bootstrap interval over supervisor configurations."""
    v = cfg[mech].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, (n_boot, len(v))).mean(1)
    return v.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stat(df, mech, col):
    v = df[df["mechanism"] == mech][col].dropna().to_numpy()
    lo, hi = bootstrap_ci(v)
    return v.mean(), lo, hi


# ------------------------------------------------------------------ Fig: main result (skill interaction)
def fig_main_result():
    """Horizontal bars: the condition names are long enough that they belong on the y axis.

    Intervals are bootstrapped over supervisor configurations, the unit every confirmatory test uses;
    bootstrapping over runs would treat the five seeds of a configuration as independent.
    """
    d = s2()
    keys = ORDER
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True, sharex=True)
    for ax, sk in zip(axes, ["worse", "okay", "better"]):
        sub = d[d["skill"] == sk]
        cfg = sub.groupby(["group", "mechanism"])["gain"].mean().unstack()
        m, lo, hi = zip(*[stat_cfg(cfg, k) for k in keys])
        ann = [
            (
                pct(m[i], m[0])
                if keys[i] in ("C4_request", "C6_full") and sk == "worse"
                else ""
            )
            for i in range(len(keys))
        ]
        hbars_with_ci(ax, keys, m, lo, hi, annotate=ann)
        ax.axvline(m[0], color=MUTED, lw=0.7, ls=":", zorder=2)
        ax.set_title(
            f"({'abc'[['worse', 'okay', 'better'].index(sk)]}) {SKILL_LABEL[sk]}",
            fontsize=8.5,
        )
    axes[0].set_xlim(0, None)
    for ax in axes:
        ax.set_xlabel("Learning gain")
    fig.tight_layout()
    return save(fig, "fig1_main_result", tight=False)


# ------------------------------------------------------------------ Fig: risk of harmful supervision
def fig_risk_reduction():
    d = s2()
    keys = [k for k in ORDER if k != "HG_DAgger"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    y = np.arange(len(keys))[::-1]

    ax = axes[0]
    allv = [float((d[d["mechanism"] == k]["gain"] < 0).mean()) * 100 for k in keys]
    worsev = [
        float((d[(d["mechanism"] == k) & (d["skill"] == "worse")]["gain"] < 0).mean()) * 100
        for k in keys
    ]
    ax.barh(y + 0.19, allv, 0.36, color=[C[k] for k in keys], zorder=3, linewidth=0)
    ax.barh(y - 0.19, worsev, 0.36, color=[C[k] for k in keys], alpha=0.45, hatch="///",
            edgecolor="white", linewidth=0.4, zorder=3)
    for yi, v in zip(y + 0.19, allv):
        ax.annotate(f"{v:.0f}", (v, yi), textcoords="offset points", xytext=(3, -2.2), fontsize=6.8)
    for yi, v in zip(y - 0.19, worsev):
        ax.annotate(f"{v:.0f}", (v, yi), textcoords="offset points", xytext=(3, -2.2), fontsize=6.8)
    ax.set_yticks(y, [SHORT[k] for k in keys])
    ax.set_ylim(-0.65, len(keys) - 0.35)
    ax.set_xlabel("Runs where supervision hurt the robot (%)")
    ax.set_title("(a) Harmful-supervision rate", fontsize=8.5)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=MUTED),
            plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45, hatch="///"),
        ],
        labels=["all supervisors", "less-skilled supervisors"],
        loc="lower right",
        fontsize=7,
    )
    grid(ax, "x")

    ax = axes[1]
    per_cfg = d.groupby(["group", "mechanism"])["gain"].mean().unstack()
    for yi, k in zip(y, keys):
        v = np.sort(per_cfg[k].to_numpy())
        worst = v[: max(1, int(np.ceil(0.1 * len(v))))].mean()
        ax.barh(yi, worst, height=0.68, color=C[k], zorder=3, linewidth=0)
        ax.annotate(f"{worst:+.2f}", (worst, yi), textcoords="offset points",
                    xytext=(4 if worst >= 0 else -26, -2.5), fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Worst-10% of supervisor profiles: mean learning gain")
    ax.set_title("(b) Worst-case robustness", fontsize=8.5)
    grid(ax, "x")
    return save(fig, "fig2_risk_reduction")


# ------------------------------------------------------------------ Fig: supervision cost (confirmatory)
def fig_supervision_cost():
    """Configuration-level paired differences vs no transparency, supervision-cost family (n = 33).

    attend_steps is deliberately NOT shown: the reliance parameter lowers attention only for request mechanisms
    (see results/tables/confirmatory_attention_confound.csv), so that saving is a property of the supervisor model.
    """
    paired = pd.read_csv(ROOT / "results/tables/confirmatory_paired_config_level.csv")
    keys = [k for k in ORDER if k != "C1_none"]
    panels = [
        ("takeovers", "Takeovers per run"),
        ("control_steps", "Supervisor control steps"),
        ("unnecessary_takeovers", "Unnecessary takeovers"),
        ("precision", "Takeover precision"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True)
    for j, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        sub = paired[paired["metric"] == metric].set_index("mechanism").reindex(keys)
        y = np.arange(len(keys))[::-1]
        ax.errorbar(
            sub["mean"],
            y,
            xerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            fmt="o",
            ms=4,
            lw=1.0,
            color=INK,
            ecolor=MUTED,
            zorder=3,
        )
        for yi, k in zip(y, keys):
            r = sub.loc[k]
            ax.plot(r["mean"], yi, "o", ms=5.5, color=C[k], zorder=4)
            if r["p_holm_within_metric"] < 0.05:
                ax.annotate(
                    "*",
                    (r["ci_high"], yi),
                    textcoords="offset points",
                    xytext=(3, -3),
                    fontsize=10,
                    color=C[k],
                    weight="bold",
                )
        ax.axvline(0, color=MUTED, lw=0.8, ls="--")
        if j == 0:
            ax.set_yticks(y, [SHORT[k] for k in keys])
        ax.set_title(title, fontsize=8)
        grid(ax, "x")
    axes[0].set_ylabel("difference vs no transparency (95% CI)")
    return save(fig, "fig3_supervision_cost")


# ------------------------------------------------------------------ Fig: equivalence + tail risk
def fig_equivalence_tail():
    eq = pd.read_csv(ROOT / "results/tables/confirmatory_equivalence_tost.csv")
    tail = pd.read_csv(ROOT / "results/tables/confirmatory_tail_risk.csv")
    # 90% CIs (the interval TOST actually checks against the margin) come from the same paired configuration means
    runs = pd.read_csv(ROOT / "results/processed/S2_runs.csv")
    runs["skill"] = runs["group"].str.extract(r"(worse|okay|better)")[0]
    wide = (
        runs[runs["skill"] != "worse"]
        .groupby(["group", "mechanism"])["final_success"]
        .mean()
        .unstack("mechanism")
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    ax = axes[0]
    sub = eq[eq["subset"] == "okay+better"].set_index("mechanism")
    keys = [k for k in ("C2_uncertainty", "C4_request", "C6_full") if k in sub.index]
    y = np.arange(len(keys))[::-1]
    ax.axvspan(-0.05, 0.05, color="#E8F1EF", zorder=0)
    for m, ha in ((-0.05, "left"), (0.05, "right")):
        ax.axvline(m, color="#9FBDB7", lw=0.9)
    ax.annotate(
        "equivalence margin",
        (-0.05, 2.62),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7,
        color="#4C7F77",
        ha="left",
        va="center",
    )
    for yi, k in zip(y, keys):
        r = sub.loc[k]
        d = (wide[k] - wide["C1_none"]).to_numpy()
        half = 1.7247 * d.std(ddof=1) / np.sqrt(len(d))  # t(0.95, 21)
        ax.errorbar(
            r["mean"],
            yi,
            xerr=half,
            fmt="o",
            ms=7,
            color=C[k],
            ecolor=C[k],
            lw=1.6,
            capsize=3,
            zorder=3,
        )
        ax.annotate(
            f"TOST p = {r['p_tost']:.3f}",
            (r["mean"] + half, yi),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7.5,
            color=C[k],
            weight="bold",
        )
    ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(-0.085, 0.085)
    ax.set_ylim(-0.6, 2.8)
    ax.set_yticks(y, [SHORT[k] for k in keys])
    ax.set_xlabel("Final success difference vs no transparency (mean, 90% CI)")
    ax.set_title(
        "(a) No cost for competent supervisors\n(equivalence margin +/- 0.05, n = 22 configs)",
        fontsize=8.5,
    )
    grid(ax, "x")

    ax = axes[1]
    keys2 = ["C2_uncertainty", "C4_request", "C6_full"]
    width = 0.36
    for j, q in enumerate((0.10, 0.25)):
        for i, k in enumerate(keys2):
            r = tail[(tail["mechanism"] == k) & (tail["quantile"] == q)].iloc[0]
            ax.bar(
                i + (j - 0.5) * width,
                r["delta"],
                width,
                color=C[k],
                alpha=1.0 if q == 0.10 else 0.5,
                zorder=3,
                linewidth=0,
            )
            ax.errorbar(
                i + (j - 0.5) * width,
                r["delta"],
                yerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
                color=INK,
                lw=0.8,
                capsize=2,
            )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(keys2)), [STACK[k] for k in keys2])
    ax.set_ylabel("Improvement of the tail mean")
    ax.set_title(
        "(b) Worst-case protection\n(dark: worst 10%, light: worst 25% of configs)",
        fontsize=8.5,
    )
    grid(ax)
    return save(fig, "fig7_equivalence_tail")


# ------------------------------------------------------------------ Fig: why the gate baseline fails
def fig_request_diagnosis():
    diag = pd.read_csv(
        ROOT / "results/tables/confirmatory_request_diagnosis.csv"
    ).set_index("mechanism")
    keys = ["C4_request", "C5_thrifty", "C6_full"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    ax = axes[0]
    for i, k in enumerate(keys):
        ax.bar(
            i, diag.loc[k, "mean_u_at_request"], 0.6, color=C[k], zorder=3, linewidth=0
        )
        ax.annotate(
            f"{diag.loc[k, 'mean_u_at_request']:.2f}",
            (i, diag.loc[k, "mean_u_at_request"]),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(range(len(keys)), [STACK[k] for k in keys])
    ax.set_ylabel("Risk when the request is issued")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) The gate asks too early", fontsize=8.5)
    grid(ax)
    ax = axes[1]
    x = np.arange(len(keys))
    ax.bar(
        x - 0.19,
        [diag.loc[k, "necessity_request"] for k in keys],
        0.38,
        color=[C[k] for k in keys],
        zorder=3,
        linewidth=0,
    )
    ax.bar(
        x + 0.19,
        [diag.loc[k, "necessity_self"] for k in keys],
        0.38,
        color=[C[k] for k in keys],
        alpha=0.45,
        hatch="///",
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for xi, k in zip(x, keys):
        ax.annotate(
            f"{diag.loc[k, 'necessity_request']:.2f}",
            (xi - 0.19, diag.loc[k, "necessity_request"]),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7,
        )
        ax.annotate(
            f"{diag.loc[k, 'necessity_self']:.2f}",
            (xi + 0.19, diag.loc[k, "necessity_self"]),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x, [STACK[k] for k in keys])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Necessary share of takeovers")
    ax.set_title(
        "(b) Robot-initiated vs supervisor-initiated\n(solid: after a request, hatched: self-initiated)",
        fontsize=8.5,
    )
    grid(ax)
    return save(fig, "fig8_request_diagnosis")


# ------------------------------------------------------------------ Fig: per-configuration improvement
def fig_paired_arrows():
    d = s2()
    per_cfg = d.groupby(["group", "mechanism"])["gain"].mean().unstack()
    per_cfg = per_cfg.sort_values("C1_none")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    ax = axes[0]
    y = np.arange(len(per_cfg))
    for yi, (c1, c6) in zip(y, per_cfg[["C1_none", "C6_full"]].to_numpy()):
        ax.annotate(
            "",
            xy=(c6, yi),
            xytext=(c1, yi),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.0,
                color=C["C6_full"] if c6 >= c1 else "#B23A48",
                alpha=0.9 if c6 >= c1 else 0.6,
            ),
        )
    ax.scatter(per_cfg["C1_none"], y, s=11, color=C["C1_none"], zorder=3)
    ax.scatter(per_cfg["C6_full"], y, s=11, color=C["C6_full"], zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_xlabel("Learning gain")
    ax.set_ylabel("33 supervisor profiles (sorted by baseline)")
    ax.set_title("(a) No transparency -> display + request,\nper supervisor profile", fontsize=8.5)
    n_up = int((per_cfg["C6_full"] > per_cfg["C1_none"]).sum())
    ax.annotate(
        f"improves in {n_up}/{len(per_cfg)} configurations;\nno configuration left with a negative gain",
        (0.03, 0.05),
        xycoords="axes fraction",
        fontsize=7,
        color=INK,
    )
    grid(ax, "x")

    ax = axes[1]
    d2 = d.pivot_table(index=["group", "seed"], columns="mechanism", values="gain")
    x = d2.loc[(slice(None), [0, 1]), "C1_none"].groupby(level=0).mean()
    for k in ("C4_request", "C6_full"):
        yv = (
            (
                d2.loc[(slice(None), [2, 3, 4]), k]
                - d2.loc[(slice(None), [2, 3, 4]), "C1_none"]
            )
            .groupby(level=0)
            .mean()
        )
        idx = x.index.intersection(yv.index)
        xs, ys = x[idx].to_numpy(), yv[idx].to_numpy()
        ax.scatter(
            xs, ys, s=18, color=C[k], alpha=0.8, zorder=3, label=f"{SHORT[k]} vs baseline"
        )
        fit = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min(), xs.max(), 40)
        ax.plot(gx, np.polyval(fit, gx), color=C[k], lw=1.2, zorder=2)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.annotate(
            f"{SHORT[k]}: r = {r:.2f}",
            (0.03, 0.14 if k == "C4_request" else 0.04),
            xycoords="axes fraction",
            fontsize=7,
            color=C[k],
            weight="bold",
        )
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Baseline gain without mechanism (held-out seeds)")
    ax.set_ylabel("Gain difference vs no transparency")
    ax.set_title("(b) Benefit concentrates where supervision fails", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7)
    grid(ax, "both")
    return save(fig, "fig4_paired_improvement")


# ------------------------------------------------------------------ Fig: signal quality
def fig_signal_quality():
    import json

    q = json.loads((ROOT / "results/S1/quality_pretrain.json").read_text())
    hs = ["H5", "H10", "H20"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    ax = axes[0]
    x = np.arange(len(hs))
    ens = [q[h][f"auroc_ens_std_fail_within_{h[1:]}"] for h in hs]
    risk = [q[h][f"auroc_q_mean_fail_within_{h[1:]}"] for h in hs]
    ax.bar(
        x - 0.19, ens, 0.38, color="#B9C6CC", zorder=3, label="ensemble disagreement"
    )
    ax.bar(
        x + 0.19,
        risk,
        0.38,
        color=C["C2_uncertainty"],
        zorder=3,
        label="predicted risk (-mean Q)",
    )
    for xi, v in zip(x - 0.19, ens):
        ax.annotate(
            f"{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7,
        )
    for xi, v in zip(x + 0.19, risk):
        ax.annotate(
            f"{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7,
            color=C["C2_uncertainty"],
            weight="bold",
        )
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(x, ["5 steps", "10 steps", "20 steps"])
    ax.set_xlabel("Failure horizon")
    ax.set_ylabel("AUROC (predicting failure)")
    ax.set_ylim(0.3, 0.92)
    ax.set_title("(a) What the robot knows about its own risk", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7)
    grid(ax)

    ax = axes[1]
    d = pd.read_csv(ROOT / "results/processed/S1_runs.csv")
    groups = [
        ("C1_none", ["none"]),
        ("C2_uncertainty", ["risk", "ensemble", "random"]),
        ("C4_request", ["risk", "ensemble", "random"]),
        ("C6_full", ["risk", "ensemble", "random"]),
    ]
    alpha = {"risk": 1.0, "ensemble": 0.55, "random": 0.28, "none": 1.0}
    # one y slot per condition; the three signals sit inside it, told apart by shade (see legend)
    h, centres = 0.26, []
    for gi, (mech, sigs) in enumerate(groups):
        yc = len(groups) - 1 - gi
        centres.append((yc, mech))
        offs = [0.0] if len(sigs) == 1 else [h, 0.0, -h]
        for sig, off in zip(sigs, offs):
            v = d[(d["mechanism"] == mech) & (d["group"] == sig)]["gain"].to_numpy()
            if not len(v):
                continue
            ax.barh(yc + off, v.mean(), h * 0.92, color=C[mech], alpha=alpha[sig],
                    zorder=3, linewidth=0)
            ax.scatter(v, np.full(len(v), yc + off), s=6, color=INK, zorder=4, alpha=0.6)
    ax.set_yticks([yc for yc, _ in centres], [SHORT[m] for _, m in centres], fontsize=7.5)
    ax.set_ylim(-0.55, len(groups) - 0.45)
    ax.set_xlim(0, 0.86)  # right margin kept clear so the legend sits over no data
    ax.set_xlabel("Learning gain")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=a)
            for a in (1.0, 0.55, 0.28)
        ],
        labels=["predicted risk", "ensemble disagr.", "random control"],
        loc="upper right",
        fontsize=6.5,
        title="displayed / gating signal",
        title_fontsize=6.5,
        borderpad=0.5,
    )
    ax.set_title("(b) Closed loop: signal type does not separate\n(3 seeds per cell)", fontsize=8.5)
    grid(ax, "x")
    return save(fig, "fig5_signal_quality")


# ------------------------------------------------------------------ Fig: supervisor model validation
def fig_supervisor_validation():
    import json

    fit = json.loads((ROOT / "data/robomimic/skill_fit.json").read_text())["fits"]
    stats = ("length", "path_ratio", "dir_noise")
    labels = ("Episode\nlength", "Path\nratio", "Direction\nnoise")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    for ax, level in zip(axes[:2], ("okay", "worse")):
        x = np.arange(len(stats))
        ax.bar(
            x - 0.19,
            [fit[level]["robomimic_ratio_to_better"][s] for s in stats],
            0.38,
            color="#2F6FA8",
            zorder=3,
            label="robomimic operators",
        )
        ax.bar(
            x + 0.19,
            [fit[level]["sim_ratio_to_better"][s] for s in stats],
            0.38,
            color="#9CCFC6",
            zorder=3,
            hatch="//",
            edgecolor="#3E9B8F",
            linewidth=0.6,
            label="fitted simulated supervisor",
        )
        ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_title(
            f"({'ab'[('okay', 'worse').index(level)]}) '{level}' operators",
            fontsize=8.5,
        )
        grid(ax)
    axes[0].set_ylabel("Ratio to skilled operators")
    axes[0].legend(loc="upper left", fontsize=7)
    ax = axes[2]
    from uhitl.supervisor import SimulatedSupervisor, SupervisorParams

    for delay, color in zip((0.5, 1.5, 2.7), ("#3E9B8F", "#2F6FA8", "#C6522B")):
        sup = SimulatedSupervisor(SupervisorParams(delay_s=delay), "C1_none", seed=0)
        samples = np.array([sup._delay_steps() for _ in range(4000)]) * 0.1
        ax.hist(
            samples,
            bins=np.arange(0, 7, 0.2),
            histtype="step",
            lw=1.3,
            density=True,
            color=color,
            label=f"mean {delay} s",
        )
    ax.axvline(2.7, color=MUTED, lw=0.9, ls="--")
    ax.annotate(
        "meta-analysis mean\n2.7 s (129 studies)",
        (2.85, 1.1),
        fontsize=6.8,
        color=MUTED,
    )
    ax.set_xlabel("Take-over reaction delay (s)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Reaction delay", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7)
    grid(ax)
    return save(fig, "fig6_supervisor_validation")


if __name__ == "__main__":
    names = [
        fig_main_result(),
        fig_risk_reduction(),
        fig_supervision_cost(),
        fig_equivalence_tail(),
        fig_request_diagnosis(),
        fig_paired_arrows(),
        fig_signal_quality(),
        fig_supervisor_validation(),
    ]
    print(names)
