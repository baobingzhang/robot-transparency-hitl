"""Per-run metrics computed from the files written by uhitl.train."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(run_dir):
    d = Path(run_dir)
    z = np.load(d / "steps.npz")
    steps = pd.DataFrame(z["data"], columns=[str(c) for c in z["columns"]])
    read = lambda name: (
        pd.read_csv(d / f"{name}.csv")
        if (d / f"{name}.csv").stat().st_size > 0
        else pd.DataFrame()
    )
    return {
        "cfg": json.loads((d / "config.json").read_text()),
        "steps": steps,
        "episodes": read("episodes"),
        "takeovers": read("takeovers"),
        "evals": pd.read_csv(d / "evals.csv"),
    }


def run_metrics(run_dir):
    r = load_run(run_dir)
    cfg, steps, eps, tk, ev = (
        r["cfg"],
        r["steps"],
        r["episodes"],
        r["takeovers"],
        r["evals"],
    )
    # start = 100-episode success of the shared pretrained checkpoint when available (10-episode evals are noisy)
    start = cfg.get("pretrained_success") or float(ev["autonomous_success"].iloc[0])
    final = float(ev["autonomous_success"].iloc[-1])
    control = float(steps["is_intervention"].sum())
    attend = float(steps["attend"].sum())
    n_tk = len(tk)
    necessary = (
        tk["necessary"].astype(str).str.lower().eq("true")
        if n_tk
        else pd.Series(dtype=bool)
    )
    n_nec = int(necessary.sum()) if n_tk else 0
    missed = (
        int(((~eps["success"].astype(bool)) & (eps["takeovers"] == 0)).sum())
        if len(eps)
        else 0
    )
    # trust calibration: trust at each episode end vs autonomous success of the latest evaluation checkpoint
    ev_steps, ev_vals = ev["step"].to_numpy(), ev["autonomous_success"].to_numpy()
    if len(eps):
        idx = np.searchsorted(ev_steps, eps["end_step"].to_numpy(), side="right") - 1
        rel = ev_vals[np.clip(idx, 0, len(ev_vals) - 1)]
        gap = eps["trust"].to_numpy() - rel
        trust_err, overtrust = float(np.mean(gap**2)), float(np.mean(gap > 0.2))
    else:
        trust_err = overtrust = np.nan
    # learning-curve metrics (evaluations during the run, excluding the initial checkpoint evaluation)
    curve = ev.iloc[1:-1] if len(ev) > 2 else ev.iloc[1:]
    auc = float(curve["autonomous_success"].mean()) if len(curve) else np.nan
    def steps_to(thr):
        hit = curve[curve["autonomous_success"] >= thr]
        return float(hit["step"].iloc[0]) if len(hit) else np.nan
    requests = int(eps["requests"].sum()) if len(eps) else 0
    from_request = (
        tk[tk["source"] == "request"] if n_tk and "source" in tk else pd.DataFrame()
    )
    return {
        "run": str(run_dir),
        "mechanism": (
            cfg["mechanism"] if cfg.get("learner", "sac") == "sac" else "HG_DAgger"
        ),
        "seed": cfg["seed"],
        "start_success": start,
        "final_success": final,
        "gain": final - start,
        "control_steps": control,
        "attend_steps": attend,
        "gain_per_100_control": (final - start) / max(control, 1) * 100,
        "gain_per_1000_attend": (final - start) / max(attend, 1) * 1000,
        "takeovers": n_tk,
        "context_switches": 2 * n_tk,
        "necessary_takeovers": n_nec,
        "unnecessary_takeovers": n_tk - n_nec,
        "precision": n_nec / n_tk if n_tk else np.nan,
        "missed_failures": missed,
        "recall": n_nec / (n_nec + missed) if (n_nec + missed) else np.nan,
        "requests": requests,
        "request_precision": (
            float(from_request["necessary"].astype(str).str.lower().eq("true").mean())
            if len(from_request)
            else np.nan
        ),
        "auc_success": auc,
        "steps_to_60": steps_to(0.6),
        "steps_to_80": steps_to(0.8),
        "trust_sq_error": trust_err,
        "overtrust_fraction": overtrust,
        "episodes": len(eps),
        "train_success_rate": float(eps["success"].mean()) if len(eps) else np.nan,
        **{f"sup_{k}": v for k, v in cfg["supervisor_resolved"].items()},
    }


def _auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def failure_labels(run_dir, horizon=10):
    """Robot-controlled steps of finished episodes, labelled 1 if the episode fails within `horizon` steps."""
    r = load_run(run_dir)
    eps = r["episodes"][["episode", "length", "success"]].copy()
    eps["success"] = eps["success"].astype(str).str.lower().eq("true")
    s = r["steps"]
    s = s[(s["is_intervention"] == 0) & s["u_raw"].notna()].copy()
    s["episode"] = s["episode"].astype(int)
    s = s.merge(eps, on="episode", how="inner")
    s["label"] = (~s["success"]) & (s["length"] - s["t"] <= horizon)
    return s


def uncertainty_quality(run_dir, horizon=10):
    s = failure_labels(run_dir, horizon)
    return {
        "run": str(run_dir),
        "auroc": _auroc(s["u_raw"], s["label"]),
        "n": len(s),
        "positives": int(s["label"].sum()),
    }
