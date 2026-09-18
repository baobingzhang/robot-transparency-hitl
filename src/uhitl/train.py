"""One human-in-the-loop RL run with a simulated supervisor and a transparency mechanism.

Outputs (in cfg['out_dir']): config.json, steps.npz, episodes.csv, evals.csv, takeovers.csv, final.pt
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from uhitl.agent import SACAgent
from uhitl.envs import Snapshot, flat_obs, make_env
from uhitl.mechanisms import (
    BudgetedRequester,
    IntentLabel,
    NoveltyRiskGate,
    SlidingNormalizer,
    UncertaintyEstimator,
)
from uhitl.supervisor import ScriptedOperator, SimulatedSupervisor, SupervisorParams

MAX_T = 100

DEFAULTS = {
    "seed": 0,
    "mechanism": "C1_none",
    "learner": "sac",  # "sac" or "bc" (HG-DAgger-style: actor trained only on intervention data)
    "total_steps": 6000,
    "pretrained": None,
    "num_critics": 5,
    "bc_weight": 1.0,  # auxiliary BC loss on intervention data (see results/diagnostics)
    "critic_dropout": 0.0,
    "unc_method": "risk",
    "budget": 0.2,
    "eval_every": 1000,
    "eval_episodes": 10,
    "final_eval_episodes": 50,
    "calib_episodes": 10,
    "n_demos": 0,
    "counterfactual": True,
    "cf_horizon": 50,  # max steps of the counterfactual rollout (S6 checks sensitivity)
    "cf_variants": False,  # S6: also full-horizon and 5 stochastic rollouts
    "supervisor": {},
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "out_dir": "results/debug",
}


def evaluate(agent, env, episodes, seed_base):
    succ = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        o = flat_obs(obs)
        for _ in range(MAX_T):
            obs, _, term, _, info = env.step(agent.act(o, deterministic=True))
            o = flat_obs(obs)
            if term:
                break
        succ += bool(info["succeed"])
    return succ / episodes


def counterfactual_success(agent, env, snap, t_now, horizon=None, deterministic=True):
    """Would the current policy have succeeded on its own from this exact state? (state restored afterwards)"""
    o = snap.restore(env)
    ok = False
    steps = MAX_T - t_now if horizon is None else min(MAX_T - t_now, horizon)
    for _ in range(steps):
        obs, _, term, _, info = env.step(agent.act(o, deterministic=deterministic))
        o = flat_obs(obs)
        if term:
            ok = bool(info["succeed"])
            break
    snap.restore(env)
    return ok


def collect_demos(agent, env, n, rng):
    op = ScriptedOperator(1.0, 0.05, rng)
    for ep in range(n):
        obs, _ = env.reset(seed=900_000 + ep)
        o = flat_obs(obs)
        op.reset()
        for _ in range(MAX_T):
            a = op.act(o)
            obs, r, term, _, info = env.step(a)
            o2 = flat_obs(obs)
            agent.interv.add(
                o, a, r + info.get("discrete_penalty", 0.0), o2, float(term)
            )
            o = o2
            if term:
                break


def run(cfg):
    cfg = {
        **DEFAULTS,
        **cfg,
        "supervisor": {**DEFAULTS["supervisor"], **cfg.get("supervisor", {})},
    }
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    np.random.seed(seed)  # gym-hil samples block positions with the global numpy RNG
    rng = np.random.default_rng(seed)
    env, eval_env = make_env(seed=seed), make_env(seed=seed + 1)
    agent = SACAgent(
        num_critics=cfg["num_critics"],
        critic_dropout=cfg["critic_dropout"],
        bc_weight=cfg["bc_weight"],
        device=cfg["device"],
        seed=seed,
    )
    if cfg["pretrained"]:
        ck = torch.load(
            cfg["pretrained"], map_location=cfg["device"], weights_only=False
        )
        agent.load_state_dict(ck["agent"])
        cfg["pretrained_success"] = ck.get("autonomous_success")
        for name in ("online", "interv"):
            if name not in ck:  # weights-only checkpoint: start from empty buffers
                continue
            buf = getattr(agent, name)
            for k, v in ck[name].items():
                setattr(buf, k, v.copy() if isinstance(v, np.ndarray) else v)
    if cfg["n_demos"]:
        collect_demos(agent, env, cfg["n_demos"], rng)

    sp = SupervisorParams(**cfg["supervisor"])
    sup = SimulatedSupervisor(sp, cfg["mechanism"], seed=seed + 7)
    unc = UncertaintyEstimator(agent, cfg["unc_method"], rng)
    intent = IntentLabel()

    # calibration rollouts (policy only, not stored, not counted): uncertainty scale, request threshold, gate
    calib_raw, calib_obs, calib_q = [], [], []
    for ep in range(cfg["calib_episodes"]):
        obs, _ = eval_env.reset(seed=500_000 + ep)
        o = flat_obs(obs)
        for _ in range(MAX_T):
            calib_raw.append(unc(o))
            calib_obs.append(o)
            calib_q.append(float(np.mean(agent.q_values(o))))
            obs, _, term, _, _ = eval_env.step(agent.act(o, deterministic=True))
            o = flat_obs(obs)
            if term:
                break
    norm = SlidingNormalizer(calib_raw)
    calib_u = np.array([norm(x) for x in calib_raw])
    requester = BudgetedRequester(calib_u, budget=cfg["budget"])
    gate = (
        NoveltyRiskGate(
            agent, np.array(calib_obs), np.array(calib_q), budget=cfg["budget"]
        )
        if cfg["mechanism"] == "C5_thrifty"
        else None
    )

    evals = [(0, evaluate(agent, eval_env, cfg["eval_episodes"], 700_000))]
    steps, episodes, takeovers = [], [], []
    ep, t, step = 0, 0, 0
    obs, _ = env.reset(seed=seed * 1000 + ep)
    o = flat_obs(obs)
    sup.reset_episode()
    requester.reset()
    intent.reset()
    ep_interv, ep_requests, ep_takeovers, prev_interv, ep_return = 0, 0, 0, False, 0.0
    tic = time.time()
    while step < cfg["total_steps"]:
        det = agent.act(o, deterministic=True)
        a_agent = agent.act(o) if cfg["learner"] == "sac" else det  # BC policy has no trained variance
        u_raw = unc(o) if cfg["mechanism"] != "C1_none" or step % 5 == 0 else np.nan
        u = norm(u_raw) if not np.isnan(u_raw) else np.nan
        signals = {"u": u, "intent": intent(det), "request": False}
        if cfg["mechanism"] in ("C4_request", "C6_full"):
            signals["request"] = requester(u)
        elif cfg["mechanism"] == "C5_thrifty":
            signals["request"] = gate(o)
        ep_requests += int(signals["request"])
        a_exec, is_interv, log = sup.step(o, a_agent, signals)

        if is_interv and not prev_interv:
            necessary = None
            extra = {}
            if cfg["counterfactual"]:
                snap = Snapshot(env)
                necessary = not counterfactual_success(agent, env, snap, t, cfg["cf_horizon"])
                if cfg["cf_variants"]:
                    extra["necessary_full_horizon"] = not counterfactual_success(agent, env, snap, t, None)
                    extra["stochastic_success_rate"] = float(np.mean(
                        [counterfactual_success(agent, env, snap, t, None, deterministic=False) for _ in range(5)]))
            takeovers.append(
                {
                    "step": step,
                    "episode": ep,
                    "t": t,
                    "u": u,
                    "trust": log.trust,
                    "necessary": necessary,
                    "source": log.source,
                    **extra,
                }
            )
            ep_takeovers += 1
        obs, r, term, _, info = env.step(a_exec)
        r = r + info.get("discrete_penalty", 0.0)
        o2 = flat_obs(obs)
        (agent.interv if is_interv else agent.online).add(o, a_exec, r, o2, float(term))
        if cfg["learner"] == "sac":
            agent.update()
        elif agent.interv.n >= agent.batch:
            agent.update_bc()
        steps.append(
            (
                step,
                ep,
                t,
                u_raw,
                u,
                signals["request"],
                log.attend,
                log.cue,
                log.trigger,
                is_interv,
                log.trust,
            )
        )
        ep_interv += int(is_interv)
        ep_return += r
        prev_interv = is_interv
        o, t, step = o2, t + 1, step + 1

        if term or t >= MAX_T:
            success = bool(info["succeed"])
            autonomous = success and ep_interv == 0
            episodes.append(
                {
                    "episode": ep,
                    "end_step": step,
                    "length": t,
                    "success": success,
                    "autonomous_success": autonomous,
                    "intervention_steps": ep_interv,
                    "takeovers": ep_takeovers,
                    "requests": ep_requests,
                    "trust": sup.trust,
                    "return": ep_return,
                }
            )
            sup.end_episode(autonomous)
            if gate is not None:
                gate.add_reference(
                    agent.online.o[max(0, agent.online.n - t) : agent.online.n]
                )
            ep += 1
            t, ep_interv, ep_requests, ep_takeovers, prev_interv, ep_return = (
                0,
                0,
                0,
                0,
                False,
                0.0,
            )
            obs, _ = env.reset(seed=seed * 1000 + ep)
            o = flat_obs(obs)
            sup.reset_episode()
            requester.reset()
            intent.reset()
        if step % cfg["eval_every"] == 0:
            evals.append(
                (step, evaluate(agent, eval_env, cfg["eval_episodes"], 700_000))
            )
            print(
                f"[{out.name}] step {step} eval {evals[-1][1]:.2f} episodes {ep} "
                f"interv_steps {sum(s[9] for s in steps)} {time.time() - tic:.0f}s",
                flush=True,
            )

    final = evaluate(agent, eval_env, cfg["final_eval_episodes"], 800_000)
    (out / "config.json").write_text(
        json.dumps({**cfg, "supervisor_resolved": asdict(sp)}, indent=2)
    )
    np.savez_compressed(
        out / "steps.npz",
        data=np.array(steps, dtype=np.float64),
        columns=np.array(
            [
                "step",
                "episode",
                "t",
                "u_raw",
                "u",
                "request",
                "attend",
                "cue",
                "trigger",
                "is_intervention",
                "trust",
            ]
        ),
    )
    for name, rows in (("episodes", episodes), ("takeovers", takeovers)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    with open(out / "evals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "autonomous_success"])
        w.writerows(evals + [(step, final)])
    torch.save({"agent": agent.state_dict()}, out / "final.pt")  # weights only (buffers are ~80 MB)
    (out / "DONE").write_text(f"{time.time() - tic:.1f}\n")
    return {"final_success": final, "seconds": time.time() - tic}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON file with run configuration")
    args = ap.parse_args()
    print(run(json.loads(Path(args.config).read_text())))
