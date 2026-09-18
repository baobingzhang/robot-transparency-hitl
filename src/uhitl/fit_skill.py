"""Fit ScriptedOperator (gain, noise) per skill level to robomimic Lift ratios relative to the 'better' operators."""

import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BETTER = (1.0, 0.05)
GAINS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NOISES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
EPISODES = 30


def rollout_stats(args):
    gain, noise = args
    from uhitl.envs import block_pos, flat_obs, make_env, tcp_pos
    from uhitl.supervisor import ScriptedOperator

    env = make_env(seed=0)
    rng = np.random.default_rng(1234)
    op = ScriptedOperator(gain, noise, rng)
    rows = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=10_000 + ep)
        o = flat_obs(obs)
        op.reset()
        start, target = tcp_pos(o).copy(), block_pos(o).copy()
        path, acts, info = 0.0, [], {"succeed": False}
        for t in range(100):
            a = op.act(o)
            acts.append(a[:3])
            prev = tcp_pos(o).copy()
            obs, _, term, _, info = env.step(a)
            o = flat_obs(obs)
            path += np.linalg.norm(tcp_pos(o) - prev)
            if term:
                break
        acts = np.array(acts)
        acts = acts[np.linalg.norm(acts, axis=1) > 1e-3]
        cos = np.sum(acts[1:] * acts[:-1], 1) / (
            np.linalg.norm(acts[1:], axis=1) * np.linalg.norm(acts[:-1], axis=1)
        )
        rows.append(
            {
                "success": bool(info["succeed"]),
                "length": t + 1,
                "path_ratio": path / max(np.linalg.norm(target - start), 1e-6),
                "dir_noise": float(1 - cos.mean()),
            }
        )
    succ = [r for r in rows if r["success"]] or rows
    return {
        "gain": gain,
        "noise": noise,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        **{
            k: float(np.mean([r[k] for r in succ]))
            for k in ("length", "path_ratio", "dir_noise")
        },
    }


def main():
    target = json.loads((ROOT / "data/robomimic/operator_stats.json").read_text())[
        "lift"
    ]
    grid = list(itertools.product(GAINS, NOISES))
    with Pool(6) as pool:
        results = pool.map(rollout_stats, grid)
    ref = next(r for r in results if (r["gain"], r["noise"]) == BETTER)
    fits = {"better": {"gain": BETTER[0], "noise": BETTER[1], "sim": ref}}
    for level in ("worse", "okay"):
        tr = target[level]["ratio_to_better"]
        best, best_loss = None, np.inf
        for r in results:
            if r["success_rate"] < 0.8:
                continue
            loss = sum(
                np.log(r[k] / ref[k] / tr[k]) ** 2
                for k in ("length", "path_ratio", "dir_noise")
            )
            if loss < best_loss:
                best, best_loss = r, loss
        fits[level] = {
            "gain": best["gain"],
            "noise": best["noise"],
            "loss": float(best_loss),
            "sim": best,
            "sim_ratio_to_better": {
                k: best[k] / ref[k] for k in ("length", "path_ratio", "dir_noise")
            },
            "robomimic_ratio_to_better": tr,
        }
    out = {"fits": fits, "grid": results, "episodes_per_config": EPISODES}
    (ROOT / "data/robomimic/skill_fit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
