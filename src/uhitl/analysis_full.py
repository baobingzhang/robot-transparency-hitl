"""Full analysis of S1-S6: processed data tables (results/processed), statistics (results/tables), figures (figures/).

Run: PYTHONPATH=src envs/uhitl/bin/python -m uhitl.analysis_full [S1 S2 S3 S4 S6]
Every figure is produced only from finished runs (runs with a DONE file); partially finished experiments are analysed
with whatever is complete, and the number of runs used is written into each table.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uhitl.analyze import (  # noqa: E402
        MECH_ORDER,
    bootstrap_ci,
    collect,
    paired_vs_c1,
)
from uhitl.metrics import load_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG, PROC, TAB = ROOT / "figures", ROOT / "results/processed", ROOT / "results/tables"
COLORS = {
    "C1_none": "#8A949C",
    "C2_uncertainty": "#0A7470",
    "C3_intent": "#74509A",
    "C4_request": "#2F6A9E",
    "C5_thrifty": "#96570F",
    "C6_full": "#C2410C",
    "HG_DAgger": "#C9CED2",
}
PARAM_LABEL = {
    "sup_skill_noise": "skill noise",
    "sup_delay_s": "take-over delay",
    "sup_p_attend": "attention probability",
    "sup_display_weight": "display weight",
    "sup_reliance": "reliance",
    "sup_theta": "decision threshold",
    "sup_trust_attention_gain": "trust-attention gain",
    "sup_w_f": "trust failure weight",
}
MECH_SHORT_DESC = {"C2_uncertainty": "risk display", "C4_request": "active request",
                   "C6_full": "display + request"}
SHORT = {
    "C1_none": "No transparency",
    "C2_uncertainty": "Risk display",
    "C3_intent": "Intent display",
    "C4_request": "Active request",
    "C5_thrifty": "Novelty gate",
    "C6_full": "Display + request",
    "HG_DAgger": "HG-DAgger",
}
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)
SUP_PARAMS = [
    "sup_skill_noise",
    "sup_delay_s",
    "sup_p_attend",
    "sup_display_weight",
    "sup_reliance",
    "sup_theta",
    "sup_trust_attention_gain",
    "sup_w_f",
]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def ci_table(df, by, metrics):
    rows = []
    for key, g in df.groupby(by):
        row = dict(
            zip(
                by if isinstance(by, list) else [by],
                key if isinstance(key, tuple) else (key,),
            )
        )
        row["n_runs"] = len(g)
        for m in metrics:
            v = g[m].dropna().to_numpy()
            lo, hi = bootstrap_ci(v)
            row[f"{m}_mean"], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = (
                (v.mean() if len(v) else np.nan),
                lo,
                hi,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def learning_curves(df):
    rows = []
    for _, r in df.iterrows():
        ev = pd.read_csv(Path(r["run"]) / "evals.csv")
        ev = ev.iloc[
            1:
        ].copy()  # first row is the 10-episode eval of the shared checkpoint
        ev["mechanism"], ev["group"], ev["seed"] = r["mechanism"], r["group"], r["seed"]
        rows.append(ev)
    curves = pd.concat(rows, ignore_index=True)
    last = curves.groupby(["group", "mechanism", "seed"])["step"].transform("max")
    curves.loc[
        curves["step"].eq(last)
        & curves.duplicated(["group", "mechanism", "seed", "step"], keep="last"),
        "final",
    ] = True
    return curves


# ---------------------------------------------------------------- S1
def analyse_s1():
    df = collect("S1")
    if not len(df):
        return
    df["signal"] = df["group"]
    df.to_csv(PROC / "S1_runs.csv", index=False)
    metrics = [
        "gain",
        "control_steps",
        "unnecessary_takeovers",
        "precision",
        "requests",
        "request_precision",
    ]
    tab = ci_table(df, ["mechanism", "signal"], metrics)
    tab.to_csv(TAB / "S1_signal_by_mechanism.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    order = [
        ("C1_none", "none"),
        ("C2_uncertainty", "risk"),
        ("C2_uncertainty", "ensemble"),
        ("C2_uncertainty", "random"),
        ("C4_request", "risk"),
        ("C4_request", "ensemble"),
        ("C4_request", "random"),
    ]
    labels = [
        "none",
        "display\nrisk",
        "display\nens.",
        "display\nrand.",
        "request\nrisk",
        "request\nens.",
        "request\nrand.",
    ]
    for ax, (m, title) in zip(
        axes,
        [
            ("gain", "Learning gain"),
            ("control_steps", "Supervisor control steps"),
            ("unnecessary_takeovers", "Unnecessary takeovers"),
        ],
    ):
        vals = [
            df[(df["mechanism"] == a) & (df["signal"] == b)][m].to_numpy()
            for a, b in order
        ]
        for i, (v, (a, b)) in enumerate(zip(vals, order)):
            if len(v):
                ax.bar(
                    i,
                    v.mean(),
                    color=COLORS[a],
                    alpha=0.35 if b == "random" else (0.65 if b == "ensemble" else 1),
                )
                ax.scatter(
                    np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)),
                    v,
                    s=6,
                    color="k",
                    zorder=3,
                )
        ax.set_xticks(range(len(order)), labels, fontsize=6)
        ax.set_title(title, fontsize=8)
    save(fig, "fig_S1_closed_loop_signals")


# ---------------------------------------------------------------- S2 / S5
def analyse_s2():
    df = collect("S2")
    if not len(df):
        return None
    df["skill"] = df["group"].str.extract(r"(worse|okay|better)")[0]
    df.to_csv(PROC / "S2_runs.csv", index=False)
    metrics = [
        "final_success",
        "gain",
        "control_steps",
        "attend_steps",
        "takeovers",
        "unnecessary_takeovers",
        "precision",
        "recall",
        "missed_failures",
        "trust_sq_error",
        "overtrust_fraction",
        "gain_per_100_control",
        "auc_success",
        "steps_to_60",
        "steps_to_80",
        "requests",
        "request_precision",
    ]
    ci_table(df, ["mechanism"], metrics).to_csv(
        TAB / "S2_by_mechanism.csv", index=False
    )
    ci_table(df, ["skill", "mechanism"], metrics).to_csv(
        TAB / "S2_by_skill_mechanism.csv", index=False
    )
    stats = pd.concat(
        [
            paired_vs_c1(df, m)
            for m in [
                "gain",
                "final_success",
                "auc_success",
                "control_steps",
                "attend_steps",
                "unnecessary_takeovers",
                "precision",
                "recall",
                "trust_sq_error",
                "gain_per_100_control",
            ]
        ],
        ignore_index=True,
    )
    stats.to_csv(TAB / "S2_paired_vs_C1.csv", index=False)
    mechs = [m for m in MECH_ORDER if m in set(df["mechanism"])]

    # overview: 6 metrics, mean +/- 95% CI
    panels = [
        ("gain", "Learning gain"),
        ("final_success", "Final autonomous success"),
        ("control_steps", "Supervisor control steps"),
        ("unnecessary_takeovers", "Unnecessary takeovers"),
        ("precision", "Takeover precision"),
        ("trust_sq_error", "Trust calibration error"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    for ax, (m, title) in zip(axes.flat, panels):
        for i, k in enumerate(mechs):
            v = df[df["mechanism"] == k][m].dropna().to_numpy()
            lo, hi = bootstrap_ci(v)
            ax.bar(i, v.mean(), color=COLORS[k], width=0.7)
            ax.errorbar(
                i,
                v.mean(),
                yerr=[[v.mean() - lo], [hi - v.mean()]],
                color="k",
                lw=0.8,
                capsize=2,
            )
        ax.set_xticks(range(len(mechs)), [SHORT[k] for k in mechs])
        ax.set_title(title, fontsize=8)
    save(fig, "fig_S2_overview")

    # efficiency frontier: control steps vs gain (per mechanism means over supervisor configs)
    agg = (
        df.groupby(["group", "mechanism"])[["gain", "control_steps", "attend_steps"]]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for k in mechs:
        a = agg[agg["mechanism"] == k]
        axes[0].scatter(
            a["control_steps"],
            a["gain"],
            s=10,
            color=COLORS[k],
            alpha=0.6,
            label=SHORT[k],
        )
        axes[1].scatter(
            a["attend_steps"],
            a["gain"],
            s=10,
            color=COLORS[k],
            alpha=0.6,
            label=SHORT[k],
        )
    axes[0].set_xlabel("Supervisor control steps (per run)")
    axes[1].set_xlabel("Supervisor monitoring steps (per run)")
    for ax in axes:
        ax.set_ylabel("Learning gain")
    axes[1].legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig_S2_cost_vs_gain")

    # by skill level
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, sk in zip(axes, ["worse", "okay", "better"]):
        sub = df[df["skill"] == sk]
        for i, k in enumerate(mechs):
            v = sub[sub["mechanism"] == k]["gain"].to_numpy()
            if len(v):
                lo, hi = bootstrap_ci(v)
                ax.bar(i, v.mean(), color=COLORS[k])
                ax.errorbar(
                    i,
                    v.mean(),
                    yerr=[[v.mean() - lo], [hi - v.mean()]],
                    color="k",
                    lw=0.8,
                    capsize=2,
                )
        ax.set_xticks(range(len(mechs)), [SHORT[k] for k in mechs])
        ax.set_title(f"'{sk}' supervisors", fontsize=8)
    axes[0].set_ylabel("Learning gain")
    save(fig, "fig_S2_gain_by_skill")

    # learning curves
    curves = learning_curves(df)
    curves.to_csv(PROC / "S2_learning_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k in mechs:
        c = (
            curves[curves["mechanism"] == k]
            .groupby("step")["autonomous_success"]
            .mean()
        )
        ax.plot(c.index, c.values, "-o", ms=2.5, color=COLORS[k], label=SHORT[k])
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Autonomous success")
    ax.legend(frameon=False, fontsize=6, ncol=3)
    save(fig, "fig_S2_learning_curves")

    # precision / recall
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k in mechs:
        a = df[df["mechanism"] == k]
        ax.scatter(a["recall"], a["precision"], s=5, alpha=0.35, color=COLORS[k])
        ax.scatter(
            a["recall"].mean(),
            a["precision"].mean(),
            s=50,
            color=COLORS[k],
            edgecolor="k",
            label=SHORT[k],
        )
    ax.set_xlabel("Takeover recall")
    ax.set_ylabel("Takeover precision")
    ax.legend(frameon=False, fontsize=6, ncol=3)
    save(fig, "fig_S2_precision_recall")

    headline(df, mechs)
    analyse_s5(df, mechs)
    return df


def headline(df, mechs):
    """Headline table + paired improvement figure: the angles where the mechanisms matter most."""
    per_cfg = df.groupby(["group", "mechanism"])[["gain", "auc_success", "control_steps", "attend_steps",
                                                  "unnecessary_takeovers", "precision", "trust_sq_error"]].mean()
    wide = per_cfg.reset_index().pivot(index="group", columns="mechanism")
    rows = []
    for k in mechs:
        g = wide[("gain", k)].dropna()
        worst = np.sort(g.to_numpy())[: max(1, int(np.ceil(0.1 * len(g))))]
        row = {"mechanism": k, "n_configs": len(g), "mean_gain": g.mean(), "cvar10_gain": worst.mean(),
               "configs_with_negative_gain": float((g < 0).mean()),
               "mean_unnecessary_takeovers": wide[("unnecessary_takeovers", k)].mean(),
               "mean_precision": wide[("precision", k)].mean(),
               "mean_control_steps": wide[("control_steps", k)].mean(),
               "mean_attend_steps": wide[("attend_steps", k)].mean(),
               "mean_auc": wide[("auc_success", k)].mean()}
        if k != "C1_none" and ("gain", "C1_none") in wide:
            d = (wide[("gain", k)] - wide[("gain", "C1_none")]).dropna()
            row["win_rate_vs_C1"] = float((d > 0).mean())
            row["mean_gain_diff_vs_C1"] = float(d.mean())
            lo, hi = bootstrap_ci(d.to_numpy())
            row["gain_diff_ci_low"], row["gain_diff_ci_high"] = lo, hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(TAB / "S2_headline.csv", index=False)

    best = "C6_full" if "C6_full" in mechs else ("C4_request" if "C4_request" in mechs else mechs[-1])
    if ("gain", "C1_none") not in wide or ("gain", best) not in wide:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    d = pd.DataFrame({"c1": wide[("gain", "C1_none")], "best": wide[("gain", best)]}).dropna().sort_values("c1")
    y = np.arange(len(d))
    for yi, (c1, b) in zip(y, d.to_numpy()):
        ax.annotate("", xy=(b, yi), xytext=(c1, yi),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS[best] if b >= c1 else "#B23A48", lw=0.9))
    ax.scatter(d["c1"], y, s=8, color=COLORS["C1_none"], zorder=3, label="C1 (no mechanism)")
    ax.scatter(d["best"], y, s=8, color=COLORS[best], zorder=3, label=f"{SHORT[best]} ({MECH_SHORT_DESC.get(best, '')})")
    ax.axvline(0, color="0.6", lw=0.7, ls="--")
    ax.set_yticks([])
    ax.set_xlabel("Learning gain")
    ax.set_ylabel("Supervisor configurations (sorted)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax = axes[1]
    base = wide[("gain", "C1_none")]
    for k in [m for m in ("C4_request", "C6_full") if ("gain", m) in wide]:
        d2 = (wide[("gain", k)] - base).dropna()
        b = base.reindex(d2.index)
        ax.scatter(b, d2, s=16, color=COLORS[k], alpha=0.75, label=f"{SHORT[k]} vs baseline")
        fit = np.polyfit(b, d2, 1)
        xs = np.linspace(b.min(), b.max(), 50)
        ax.plot(xs, np.polyval(fit, xs), color=COLORS[k], lw=1.2)
        r = np.corrcoef(b, d2)[0, 1]
        ax.text(0.02, 0.12 if k == "C4_request" else 0.05, f"{SHORT[k]}: slope {fit[0]:.2f}, r = {r:.2f}",
                transform=ax.transAxes, fontsize=6.5, color=COLORS[k])
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("Learning gain without any mechanism")
    ax.set_ylabel("Gain difference vs no transparency")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    save(fig, "fig_S2_headline")


def analyse_s5(df, mechs):
    """Robustness over the supervisor population: per-config means, worst-case, win rates, parameter sensitivity."""
    per_cfg = df.groupby(["group", "mechanism"])[
        ["gain", "control_steps", "unnecessary_takeovers"]
    ].mean()
    per_cfg = per_cfg.reset_index()
    params = df.drop_duplicates("group").set_index("group")[SUP_PARAMS]
    wide = per_cfg.pivot(index="group", columns="mechanism", values="gain")
    rows = []
    for k in mechs:
        v = wide[k].dropna().to_numpy()
        worst = np.sort(v)[: max(1, int(np.ceil(0.1 * len(v))))]
        row = {
            "mechanism": k,
            "n_configs": len(v),
            "mean_gain": v.mean(),
            "cvar10_gain": worst.mean(),
        }
        if k != "C1_none" and "C1_none" in wide:
            d = (wide[k] - wide["C1_none"]).dropna()
            row["win_rate_vs_C1"] = float((d > 0).mean())
            row["mean_diff_vs_C1"] = float(d.mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(TAB / "S5_robustness.csv", index=False)

    # standardized regression of (mechanism - C1) gain on supervisor parameters, bootstrap over configs
    coef_rows = []
    for k in [m for m in mechs if m != "C1_none"]:
        d = (wide[k] - wide["C1_none"]).dropna()
        X = params.loc[d.index]
        X = X.loc[:, X.std() > 0]
        Xs = (X - X.mean()) / X.std()
        A = np.column_stack([np.ones(len(Xs)), Xs.to_numpy()])
        beta = np.linalg.lstsq(A, d.to_numpy(), rcond=None)[0]
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(d), len(d))
            boots.append(np.linalg.lstsq(A[idx], d.to_numpy()[idx], rcond=None)[0])
        boots = np.array(boots)
        for j, name in enumerate(["intercept"] + list(Xs.columns)):
            coef_rows.append(
                {
                    "mechanism": k,
                    "term": name,
                    "beta": beta[j],
                    "ci_low": np.percentile(boots[:, j], 2.5),
                    "ci_high": np.percentile(boots[:, j], 97.5),
                }
            )
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(TAB / "S5_sensitivity_regression.csv", index=False)
    terms = [t for t in coefs["term"].unique() if t != "intercept"]
    show = [m for m in ("C2_uncertainty", "C4_request", "C6_full") if m in mechs]
    fig, axes = plt.subplots(1, len(show), figsize=(6.6, 2.6), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, show):
        c = coefs[(coefs["mechanism"] == k) & (coefs["term"] != "intercept")].set_index("term").reindex(terms)
        y = np.arange(len(terms))
        ax.errorbar(c["beta"], y, xerr=[c["beta"] - c["ci_low"], c["ci_high"] - c["beta"]], fmt="o", ms=3,
                    color=COLORS[k], lw=0.9)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_title(f"{SHORT[k]}\nvs no transparency", fontsize=8)
        ax.set_yticks(y, [PARAM_LABEL.get(t, t.replace("sup_", "").replace("_", " ")) for t in terms], fontsize=7)
    fig.supxlabel("Standardised effect of the supervisor parameter on the gain difference", fontsize=7.5)
    save(fig, "fig_S5_sensitivity")

    # interaction heat-maps: attention x delay for each mechanism vs no transparency
    fig, axes = plt.subplots(1, len(mechs) - 1, figsize=(7.0, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, [m for m in mechs if m != "C1_none"]):
        d = (wide[k] - wide["C1_none"]).dropna()
        p = params.loc[d.index]
        sc = ax.scatter(
            p["sup_p_attend"],
            p["sup_delay_s"],
            c=d,
            cmap="PuOr",
            vmin=-0.3,
            vmax=0.3,
            s=22,
            edgecolor="0.3",
            lw=0.3,
        )
        ax.set_title(f"{SHORT[k]}\nvs no transparency", fontsize=8)
        ax.set_xlabel("P(attend)", fontsize=7)
    axes[0].set_ylabel("Reaction delay (s)")
    fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, label="Gain difference")
    fig.savefig(FIG / "fig_S5_attention_delay_maps.png", bbox_inches="tight")
    fig.savefig(FIG / "fig_S5_attention_delay_maps.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- S3
def analyse_s3():
    df = collect("S3")
    if not len(df):
        return
    df["budget"] = df["variant"].str.extract(r"b([\d.]+)")[0].astype(float)
    df["reliance"] = df["variant"].str.extract(r"rel([\d.]+)")[0].astype(float)
    df.to_csv(PROC / "S3_runs.csv", index=False)
    tab = ci_table(
        df,
        ["group", "budget", "reliance"],
        [
            "gain",
            "control_steps",
            "requests",
            "request_precision",
            "unnecessary_takeovers",
        ],
    )
    tab.to_csv(TAB / "S3_budget_reliance.csv", index=False)
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(2, len(groups), figsize=(7.0, 4.0))
    axes = np.atleast_2d(axes)
    for j, g in enumerate(groups):
        for i, (m, cmap) in enumerate(
            [("gain", "viridis"), ("control_steps", "magma_r")]
        ):
            piv = df[df["group"] == g].pivot_table(
                index="reliance", columns="budget", values=m
            )
            ax = axes[i, j]
            ax.imshow(piv.to_numpy(), cmap=cmap, origin="lower", aspect="auto")
            ax.set_xticks(range(piv.shape[1]), piv.columns)
            ax.set_yticks(range(piv.shape[0]), piv.index)
            for (r, c), v in np.ndenumerate(piv.to_numpy()):
                ax.text(
                    c,
                    r,
                    f"{v:.2f}" if m == "gain" else f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="w",
                )
            ax.set_title(
                f"{g.replace('typical_', '')}: {m.replace('_', ' ')}", fontsize=7.5
            )
            ax.set_xlabel("Request budget b", fontsize=7)
            ax.set_ylabel("Reliance c", fontsize=7)
    save(fig, "fig_S3_budget_reliance")


# ---------------------------------------------------------------- S4
def analyse_s4():
    df = collect("S4")
    if not len(df):
        return
    df["w_f"] = df["variant"].str.extract(r"wf(\d+)")[0].astype(int)
    df["trust_gain"] = df["variant"].str.extract(r"gain([\d.]+)")[0].astype(float)
    df.to_csv(PROC / "S4_runs.csv", index=False)
    ci_table(
        df,
        ["mechanism", "w_f", "trust_gain"],
        [
            "trust_sq_error",
            "overtrust_fraction",
            "gain",
            "control_steps",
            "missed_failures",
        ],
    ).to_csv(TAB / "S4_trust.csv", index=False)
    # trust vs reliability trajectories for one setting per mechanism
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, mech in zip(axes, ["C1_none", "C2_uncertainty"]):
        sub = df[
            (df["mechanism"] == mech) & (df["w_f"] == 50) & (df["trust_gain"] == 0.5)
        ]
        for _, r in sub.iterrows():
            run = load_run(r["run"])
            eps, ev = run["episodes"], run["evals"]
            ax.plot(
                eps["end_step"], eps["trust"], color=COLORS[mech], lw=0.8, alpha=0.8
            )
            ax.step(
                ev["step"],
                ev["autonomous_success"],
                where="post",
                color="0.4",
                lw=0.8,
                ls="--",
            )
        ax.set_title(f"{SHORT[mech]} (w_f = 50, trust gain 0.5)", fontsize=8)
        ax.set_xlabel("Environment steps")
    axes[0].set_ylabel("Trust (solid) / reliability (dashed)")
    save(fig, "fig_S4_trust_trajectories")


# ---------------------------------------------------------------- S6
def analyse_s6():
    rows = []
    for done in sorted((ROOT / "results/S6").rglob("DONE")):
        tk = pd.read_csv(done.parent / "takeovers.csv")
        if not len(tk) or "necessary_full_horizon" not in tk:
            continue
        nec = tk["necessary"].astype(str).str.lower().eq("true")
        full = tk["necessary_full_horizon"].astype(str).str.lower().eq("true")
        sto = tk["stochastic_success_rate"] < 0.5
        rows.append(
            {
                "run": str(done.parent),
                "takeovers": len(tk),
                "agree_h50_full": float((nec == full).mean()),
                "agree_h50_stochastic": float((nec == sto).mean()),
                "agree_full_stochastic": float((full == sto).mean()),
                "necessary_rate_h50": float(nec.mean()),
                "necessary_rate_full": float(full.mean()),
                "necessary_rate_stochastic": float(sto.mean()),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(TAB / "S6_counterfactual_agreement.csv", index=False)


# ---------------------------------------------------------------- example trace
def fig_example_trace():
    runs = sorted((ROOT / "results/S2").glob("typical_okay/C4_request/seed0/DONE"))
    if not runs:
        return
    run = load_run(runs[0].parent)
    s, eps = run["steps"], run["episodes"]
    ep = eps[(eps["takeovers"] > 0) & (eps["requests"] > 0)]
    if not len(ep):
        return
    e = int(ep.iloc[len(ep) // 2]["episode"])
    t = s[s["episode"] == e]
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.plot(t["t"], t["u"], color="#2F6A9E", lw=1, label="normalised risk u")
    ax.fill_between(
        t["t"],
        0,
        1,
        where=t["is_intervention"] > 0,
        color="#F5E8D6",
        step="mid",
        label="supervisor control",
    )
    req = t[t["request"] > 0]
    ax.scatter(
        req["t"],
        np.ones(len(req)) * 1.03,
        marker="v",
        color="#96570F",
        s=20,
        label="request",
        zorder=3,
    )
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Time step in episode")
    ax.set_ylabel("u")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig_example_trace_C4")


def data_matrix():
    rows = []
    for exp in ["S1", "S2", "S3", "S4", "S6"]:
        man = ROOT / f"results/{exp}_manifest.json"
        planned = json.loads(man.read_text()) if man.exists() else []
        done = (
            {
                str(p.parent.relative_to(ROOT))
                for p in (ROOT / "results" / exp).rglob("DONE")
            }
            if (ROOT / "results" / exp).exists()
            else set()
        )
        rows.append(
            {
                "experiment": exp,
                "planned_in_manifest": len(planned),
                "finished": len(done),
                "finished_in_manifest": sum(r["out_dir"] in done for r in planned),
            }
        )
    pd.DataFrame(rows).to_csv(TAB / "data_matrix.csv", index=False)
    return rows


def main(exps):
    for d in (FIG, PROC, TAB):
        d.mkdir(parents=True, exist_ok=True)
    if "S1" in exps:
        analyse_s1()
    if "S2" in exps:
        analyse_s2()
        fig_example_trace()
    if "S3" in exps:
        analyse_s3()
    if "S4" in exps:
        analyse_s4()
    if "S6" in exps:
        analyse_s6()
    print(json.dumps(data_matrix()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["S1", "S2", "S3", "S4", "S6"])
