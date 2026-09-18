"""Build experiment configurations (S1-S6) and run them in parallel, resumably (runs with a DONE file are skipped)."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ("worse", "okay", "better")


def skill_params(level):
    fits = json.loads((ROOT / "data/robomimic/skill_fit.json").read_text())["fits"]
    return {"skill_gain": fits[level]["gain"], "skill_noise": fits[level]["noise"]}


def supervisor_population(n_random=30, seed=2026):
    """Latin-hypercube sample of supervisor parameters plus three typical supervisors."""
    rng = np.random.default_rng(seed)
    dims = {
        "delay_s": (0.5, 2.7),
        "p_attend": (0.3, 0.9),
        "display_weight": (0.3, 0.7),
        "reliance": (0.0, 0.6),
        "theta": (0.45, 0.65),
        "trust_attention_gain": (0.0, 0.5),
    }
    strata = {k: rng.permutation(n_random) for k in dims}
    pop = []
    for i in range(n_random):
        cfg = {
            k: float(lo + (strata[k][i] + rng.random()) / n_random * (hi - lo))
            for k, (lo, hi) in dims.items()
        }
        cfg["w_f"] = float(rng.choice([20.0, 50.0]))
        level = SKILLS[i % 3]
        pop.append({"name": f"lhs{i:02d}_{level}", "skill": level, **cfg})
    for level, (delay, attend) in zip(SKILLS, ((2.7, 0.3), (1.5, 0.6), (0.5, 0.9))):
        pop.append(
            {
                "name": f"typical_{level}",
                "skill": level,
                "delay_s": delay,
                "p_attend": attend,
                "display_weight": 0.5,
                "reliance": 0.3,
                "theta": 0.55,
                "trust_attention_gain": 0.25,
                "w_f": 50.0,
            }
        )
    return pop


def build(experiment, pretrained, total_steps, seeds):
    runs = []
    pop = supervisor_population()
    base = {
        "pretrained": str(pretrained),
        "total_steps": total_steps,
        "eval_every": 200,
        "eval_episodes": 10,
        "final_eval_episodes": 50,
    }

    def sup(p):
        return {
            **skill_params(p["skill"]),
            **{k: v for k, v in p.items() if k not in ("name", "skill")},
        }

    if experiment == "S2":
        for p in pop:
            for mech in (
                "C1_none",
                "C2_uncertainty",
                "C3_intent",
                "C4_request",
                "C5_thrifty",
                "C6_full",
            ):
                for s in seeds:
                    runs.append(
                        {
                            **base,
                            "seed": s,
                            "mechanism": mech,
                            "supervisor": sup(p),
                            "out_dir": f"results/S2/{p['name']}/{mech}/seed{s}",
                        }
                    )
            for s in seeds:
                runs.append(
                    {
                        **base,
                        "seed": s,
                        "mechanism": "C1_none",
                        "learner": "bc",
                        "supervisor": sup(p),
                        "out_dir": f"results/S2/{p['name']}/HG_DAgger/seed{s}",
                    }
                )
    elif experiment == "S3":
        typical = [p for p in pop if p["name"].startswith("typical")]
        for p in typical:
            for b in (0.1, 0.2, 0.3):
                for rel in (0.0, 0.3, 0.6):
                    for s in seeds:
                        runs.append(
                            {
                                **base,
                                "seed": s,
                                "mechanism": "C4_request",
                                "budget": b,
                                "supervisor": {**sup(p), "reliance": rel},
                                "out_dir": f"results/S3/{p['name']}/b{b}_rel{rel}/seed{s}",
                            }
                        )
    elif experiment == "S1":
        # display-signal ablation under a typical supervisor: which signal should be shown / trigger requests?
        p = next(p for p in pop if p["name"] == "typical_okay")
        for s in seeds:  # reference condition without any signal
            runs.append({**base, "seed": s, "mechanism": "C1_none", "supervisor": sup(p),
                         "out_dir": f"results/S1/none/C1_none/seed{s}"})
        for sig in ("risk", "ensemble", "random"):
            for mech in ("C2_uncertainty", "C4_request", "C6_full"):
                for s in seeds:
                    runs.append({**base, "seed": s, "mechanism": mech, "unc_method": sig, "supervisor": sup(p),
                                 "out_dir": f"results/S1/{sig}/{mech}/seed{s}"})
    elif experiment == "S4":
        p = next(p for p in pop if p["name"] == "typical_okay")
        for mech in ("C1_none", "C2_uncertainty"):
            for wf in (20.0, 50.0):
                for gain in (0.0, 0.5):
                    for s in seeds:
                        runs.append(
                            {
                                **base,
                                "seed": s,
                                "mechanism": mech,
                                "supervisor": {
                                    **sup(p),
                                    "w_f": wf,
                                    "trust_attention_gain": gain,
                                },
                                "out_dir": f"results/S4/{mech}/wf{int(wf)}_gain{gain}/seed{s}",
                            }
                        )
    elif experiment == "S6":
        # counterfactual-judgement stability: horizon-capped deterministic vs full horizon vs stochastic rollouts
        for p in [p for p in pop if p["name"].startswith("typical")]:
            for mech in ("C1_none", "C4_request"):
                for s in seeds[:2]:
                    runs.append({**base, "seed": s, "mechanism": mech, "cf_variants": True, "supervisor": sup(p),
                                 "out_dir": f"results/S6/{p['name']}/{mech}/seed{s}"})
    else:
        raise ValueError(experiment)
    return runs


def launch(run):
    out = ROOT / run["out_dir"]
    if (out / "DONE").exists():
        return run["out_dir"], "skipped", 0.0
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config_in.json"
    cfg_path.write_text(json.dumps(run, indent=2))
    env = {
        **os.environ,
        "MUJOCO_GL": "egl",
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
    }
    tic = time.time()
    with open(out / "stdout.log", "w") as log:
        rc = subprocess.call(
            [
                sys.executable,
                "-W",
                "ignore",
                "-m",
                "uhitl.train",
                "--config",
                str(cfg_path),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return run["out_dir"], "ok" if rc == 0 else f"failed rc={rc}", time.time() - tic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["S1", "S2", "S3", "S4", "S6"])
    ap.add_argument("--pretrained", default="results/pretrain/checkpoint.pt")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument(
        "--limit", type=int, default=0, help="only the first N runs (for timing)"
    )
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    runs = build(args.experiment, ROOT / args.pretrained, args.steps, args.seeds)
    if args.limit:
        runs = runs[: args.limit]
    manifest = ROOT / f"results/{args.experiment}_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(runs, indent=2))
    print(f"{args.experiment}: {len(runs)} runs -> {manifest}", flush=True)
    if args.dry:
        return
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for out_dir, status, sec in pool.map(launch, runs):
            done += 1
            print(f"[{done}/{len(runs)}] {status} {sec:.0f}s {out_dir}", flush=True)


if __name__ == "__main__":
    main()
