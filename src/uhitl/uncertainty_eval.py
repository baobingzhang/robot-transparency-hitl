"""S1 uncertainty quality on autonomous rollouts: does u(s) predict failure within H steps? (no supervisor)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from uhitl.agent import SACAgent
from uhitl.envs import flat_obs, make_env
from uhitl.metrics import _auroc
from uhitl.train import MAX_T

ROOT = Path(__file__).resolve().parents[2]


def rollouts(agent, episodes=100, seed=0, dropout_samples=0):
    env = make_env(seed=seed)
    np.random.seed(seed)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset()
        o = flat_obs(obs)
        traj = []
        for t in range(MAX_T):
            q = agent.q_values(o)
            traj.append({"t": t, "ens_std": float(np.std(q)), "q_mean": float(np.mean(q)),
                         "mc_std": float(np.std(agent.mc_dropout_q(o))) if dropout_samples else np.nan})
            obs, _, term, _, info = env.step(agent.act(o, deterministic=True))
            o = flat_obs(obs)
            if term:
                break
        success = bool(info["succeed"])
        for r in traj:
            r.update({"episode": ep, "success": success, "length": len(traj)})
        rows += traj
    return rows


def quality(rows, horizon=10):
    lab = np.array([(not r["success"]) and (r["length"] - r["t"] <= horizon) for r in rows])
    ep_fail = np.array([not r["success"] for r in rows])
    out = {"steps": len(rows), "episodes_failed": int(len({r["episode"] for r in rows if not r["success"]})),
           "positives_h": int(lab.sum())}
    for key, sign in (("ens_std", 1), ("q_mean", -1), ("mc_std", 1)):
        v = np.array([r[key] for r in rows])
        if np.all(np.isnan(v)):
            continue
        out[f"auroc_{key}_fail_within_{horizon}"] = _auroc(sign * v, lab)
        out[f"auroc_{key}_episode_fails"] = _auroc(sign * v, ep_fail)
    return out


def main(ckpt, num_critics=5, dropout=0.0, tag="pretrain"):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    agent = SACAgent(num_critics=num_critics, critic_dropout=dropout, device="cuda")
    agent.load_state_dict(ck["agent"])
    rows = rollouts(agent, dropout_samples=10 if dropout > 0 else 0)
    res = {"checkpoint": str(ckpt), **{f"H{h}": quality(rows, h) for h in (5, 10, 20)}}
    out = ROOT / f"results/S1/quality_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], tag=sys.argv[2] if len(sys.argv) > 2 else "pretrain")
