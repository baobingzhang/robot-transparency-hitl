"""Create the shared starting checkpoint: 20 scripted demonstrations + autonomous BC-regularised SAC,
stopped at the first evaluation whose autonomous success (100 episodes) lies in the target band.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from uhitl.agent import SACAgent
from uhitl.envs import flat_obs, make_env
from uhitl.train import MAX_T, collect_demos, evaluate

ROOT = Path(__file__).resolve().parents[2]
BAND = (0.35, 0.65)


def main(seed=0, max_steps=12_000, eval_every=250):
    out = ROOT / "results/pretrain"
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env, ev = make_env(seed=seed), make_env(seed=seed + 1)
    agent = SACAgent(
        num_critics=5,
        bc_weight=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
    )
    collect_demos(agent, env, 20, rng)
    obs, _ = env.reset()
    o, t, tic, curve = flat_obs(obs), 0, time.time(), []
    for step in range(1, max_steps + 1):
        a = agent.act(o)
        obs, r, term, _, info = env.step(a)
        o2 = flat_obs(obs)
        agent.online.add(o, a, r + info.get("discrete_penalty", 0.0), o2, float(term))
        agent.update()
        o, t = o2, t + 1
        if term or t >= MAX_T:
            obs, _ = env.reset()
            o, t = flat_obs(obs), 0
        if step % eval_every == 0:
            quick = evaluate(agent, ev, 20, 600_000)
            curve.append({"step": step, "success_20": quick})
            print(
                f"step {step} quick eval {quick:.2f} {time.time() - tic:.0f}s",
                flush=True,
            )
            if BAND[0] <= quick <= BAND[1]:
                full = evaluate(agent, ev, 100, 650_000)
                curve[-1]["success_100"] = full
                print(f"  100-episode eval {full:.2f}", flush=True)
                if BAND[0] <= full <= BAND[1]:
                    torch.save(
                        {
                            "agent": agent.state_dict(),
                            "autonomous_success": full,
                            "step": step,
                            "online": {
                                k: getattr(agent.online, k)
                                for k in ("o", "a", "r", "o2", "d", "n", "i")
                            },
                            "interv": {
                                k: getattr(agent.interv, k)
                                for k in ("o", "a", "r", "o2", "d", "n", "i")
                            },
                        },
                        out / "checkpoint.pt",
                    )
                    break
    (out / "pretrain_curve.json").write_text(
        json.dumps({"band": BAND, "curve": curve}, indent=2)
    )


if __name__ == "__main__":
    main()
