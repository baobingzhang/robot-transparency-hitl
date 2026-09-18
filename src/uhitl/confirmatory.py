"""Confirmatory analysis at the configuration level (n = 33), plus the checks raised by the review panel.

Seeds are nested inside supervisor configurations, so every confirmatory test averages the five seeds first and then
tests the 33 paired configuration means. Outputs: results/tables/confirmatory_*.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results/tables"
MECHS = [
    "C2_uncertainty",
    "C3_intent",
    "C4_request",
    "C5_thrifty",
    "C6_full",
    "HG_DAgger",
]


def load():
    d = pd.read_csv(ROOT / "results/processed/S2_runs.csv")
    d["skill"] = d["group"].str.extract(r"(worse|okay|better)")[0]
    d["context_switches"] = 2 * d["takeovers"]
    return d


def cfg_means(d, cols):
    return d.groupby(["group", "mechanism"])[cols].mean().reset_index()


def perm_paired(diff, n=20000, seed=0):
    """Two-sided sign-flip permutation test on paired differences."""
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    obs = diff.mean()
    signs = rng.choice([-1.0, 1.0], (n, len(diff)))
    p = float((np.abs((signs * diff).mean(1)) >= abs(obs) - 1e-12).mean())
    boot = rng.choice(diff, (10000, len(diff))).mean(1)
    return {
        "n": len(diff),
        "mean": float(obs),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_perm": p,
    }


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def tost(diff, margin=0.05):
    """Two one-sided tests for equivalence within +/- margin (paired configuration means)."""
    from scipy import stats

    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    t1 = stats.ttest_1samp(diff, -margin, alternative="greater")
    t2 = stats.ttest_1samp(diff, margin, alternative="less")
    return {
        "n": len(diff),
        "mean": float(diff.mean()),
        "p_lower": float(t1.pvalue),
        "p_upper": float(t2.pvalue),
        "p_tost": float(max(t1.pvalue, t2.pvalue)),
    }


def cvar(x, q):
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil(q * len(x))))
    return float(x[:k].mean())


def main():
    d = load()
    cols = [
        "gain",
        "final_success",
        "control_steps",
        "attend_steps",
        "takeovers",
        "context_switches",
        "unnecessary_takeovers",
        "precision",
        "recall",
        "trust_sq_error",
        "gain_per_100_control",
        "requests",
    ]
    cfg = cfg_means(d, cols)
    wide = cfg.pivot(index="group", columns="mechanism")
    out = {}

    # ---- 1. paired configuration-level tests vs C1, Holm within metric family
    rows = []
    for metric in cols:
        stats_m = []
        for m in MECHS:
            diff = (wide[(metric, m)] - wide[(metric, "C1_none")]).to_numpy()
            r = perm_paired(diff)
            r.update({"metric": metric, "mechanism": m})
            stats_m.append(r)
        ps = holm([r["p_perm"] for r in stats_m])
        for r, p in zip(stats_m, ps):
            r["p_holm_within_metric"] = float(p)
        rows += stats_m
    paired = pd.DataFrame(rows)
    paired.to_csv(TAB / "confirmatory_paired_config_level.csv", index=False)
    out["paired"] = paired

    # ---- 2. equivalence (TOST) for competent supervisors
    skill = d.drop_duplicates("group").set_index("group")["skill"]
    eq_rows = []
    for m in ["C2_uncertainty", "C4_request", "C6_full"]:
        diff = wide[("final_success", m)] - wide[("final_success", "C1_none")]
        for subset, idx in (
            ("okay+better", skill[skill != "worse"].index),
            ("okay", skill[skill == "okay"].index),
            ("better", skill[skill == "better"].index),
        ):
            r = tost(diff.reindex(idx).to_numpy())
            r.update({"mechanism": m, "subset": subset, "margin": 0.05})
            eq_rows.append(r)
    pd.DataFrame(eq_rows).to_csv(TAB / "confirmatory_equivalence_tost.csv", index=False)

    # ---- 3. tail risk: paired CVaR differences via bootstrap over configurations
    tail_rows = []
    rng = np.random.default_rng(1)
    for m in ["C4_request", "C6_full", "C2_uncertainty"]:
        a = wide[("gain", "C1_none")].to_numpy()
        b = wide[("gain", m)].to_numpy()
        for q in (0.10, 0.25):
            obs = cvar(b, q) - cvar(a, q)
            boots = []
            for _ in range(5000):
                i = rng.integers(0, len(a), len(a))
                boots.append(cvar(b[i], q) - cvar(a[i], q))
            tail_rows.append(
                {
                    "mechanism": m,
                    "quantile": q,
                    "cvar_C1": cvar(a, q),
                    "cvar_mech": cvar(b, q),
                    "delta": obs,
                    "ci_low": float(np.percentile(boots, 2.5)),
                    "ci_high": float(np.percentile(boots, 97.5)),
                }
            )
    pd.DataFrame(tail_rows).to_csv(TAB / "confirmatory_tail_risk.csv", index=False)

    # ---- 4. the attention-saving confound: reliance only lowers attention for request mechanisms
    rel = d.drop_duplicates("group").set_index("group")["sup_reliance"]
    conf_rows = []
    for m in ["C4_request", "C6_full"]:
        att = wide[("attend_steps", m)] - wide[("attend_steps", "C1_none")]
        r = np.corrcoef(rel.reindex(att.index), att)[0, 1]
        low = att[rel.reindex(att.index) < 0.2]
        conf_rows.append(
            {
                "mechanism": m,
                "metric": "attend_steps",
                "corr_with_reliance": float(r),
                "all_configs_mean": float(att.mean()),
                **{
                    f"low_reliance_{k}": v
                    for k, v in perm_paired(low.to_numpy()).items()
                },
            }
        )
        for metric in ("context_switches", "control_steps", "takeovers", "gain"):
            diff = wide[(metric, m)] - wide[(metric, "C1_none")]
            low = diff[rel.reindex(diff.index) < 0.2]
            conf_rows.append(
                {
                    "mechanism": m,
                    "metric": metric,
                    "corr_with_reliance": float(
                        np.corrcoef(rel.reindex(diff.index), diff)[0, 1]
                    ),
                    "all_configs_mean": float(diff.mean()),
                    **{
                        f"low_reliance_{k}": v
                        for k, v in perm_paired(low.to_numpy()).items()
                    },
                }
            )
    pd.DataFrame(conf_rows).to_csv(
        TAB / "confirmatory_attention_confound.csv", index=False
    )

    # ---- 4b. within-tier tests at the configuration level (the unit every confirmatory test uses)
    tier_rows = []
    for tier in ("worse", "okay", "better"):
        idx = skill[skill == tier].index
        stats_t = []
        for m in MECHS:
            diff = (wide[("gain", m)] - wide[("gain", "C1_none")]).reindex(idx).to_numpy()
            r = perm_paired(diff, n=20000, seed=3)
            r.update({"tier": tier, "mechanism": m})
            stats_t.append(r)
        ps = holm([r["p_perm"] for r in stats_t])
        for r, q in zip(stats_t, ps):
            r["p_holm_within_tier"] = float(q)
        tier_rows += stats_t
    pd.DataFrame(tier_rows).to_csv(TAB / "confirmatory_by_skill_config_level.csv", index=False)

    # ---- 5. supervisor-skill moderation at configuration level (exploratory)
    mod_rows = []
    for m in ["C2_uncertainty", "C4_request", "C6_full"]:
        diff = wide[("gain", m)] - wide[("gain", "C1_none")]
        s = skill.reindex(diff.index)
        worse, rest = diff[s == "worse"].to_numpy(), diff[s != "worse"].to_numpy()
        obs = worse.mean() - rest.mean()
        lab = np.array([1] * len(worse) + [0] * len(rest))
        vals = np.concatenate([worse, rest])
        rng2 = np.random.default_rng(2)
        null = []
        for _ in range(20000):
            p = rng2.permutation(lab)
            null.append(vals[p == 1].mean() - vals[p == 0].mean())
        mod_rows.append(
            {
                "mechanism": m,
                "worse_mean": float(worse.mean()),
                "rest_mean": float(rest.mean()),
                "contrast": float(obs),
                "p_label_perm": float((np.abs(null) >= abs(obs)).mean()),
                "n_worse": len(worse),
                "n_rest": len(rest),
            }
        )
    pd.DataFrame(mod_rows).to_csv(
        TAB / "confirmatory_skill_moderation.csv", index=False
    )

    # ---- 6. C5 gate diagnosis: when the robot asks, and whether the resulting take-overs were needed.
    # The risk at the request must come from the per-step log: takeovers.csv records the risk at the
    # moment control actually transfers, which is later by the supervisor's reaction delay.
    diag = []
    for mech in ("C4_request", "C5_thrifty", "C6_full"):
        nec_req, nec_self, u_req, u_acc = [], [], [], []
        for f in sorted((ROOT / "results/S2").glob(f"*/{mech}/seed*/steps.npz")):
            z = np.load(f, allow_pickle=True)
            st = pd.DataFrame(z["data"], columns=[str(c) for c in z["columns"]])
            req = st[st["request"] == 1]
            u_req += list(req["u"].dropna())
            u_acc += list(req[req["trigger"] == 1]["u"].dropna())
        for f in sorted((ROOT / "results/S2").glob(f"*/{mech}/seed*/takeovers.csv")):
            t = pd.read_csv(f)
            if not len(t) or "source" not in t:
                continue
            nec = t["necessary"].astype(str).str.lower().eq("true")
            nec_req += list(nec[t["source"] == "request"])
            nec_self += list(nec[t["source"] == "self"])
        diag.append(
            {
                "mechanism": mech,
                "n_request_takeovers": len(nec_req),
                "necessity_request": float(np.mean(nec_req)),
                "n_self_takeovers": len(nec_self),
                "necessity_self": float(np.mean(nec_self)),
                "n_requests": len(u_req),
                "mean_u_at_request": float(np.mean(u_req)) if u_req else np.nan,
                "mean_u_at_accepted_request": float(np.mean(u_acc)) if u_acc else np.nan,
            }
        )
    pd.DataFrame(diag).to_csv(TAB / "confirmatory_request_diagnosis.csv", index=False)

    print(
        json.dumps(
            {
                "paired_rows": len(paired),
                "files": sorted(p.name for p in TAB.glob("confirmatory_*.csv")),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
