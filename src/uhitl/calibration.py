"""Operator-skill statistics from the robomimic multi-human (MH) datasets.

Per operator group (worse / okay / better) we compute, for every demonstration:
  length       number of control steps
  path_ratio   end-effector path length / straight-line distance from start to the object's initial position
  dir_noise    1 - mean cosine similarity between consecutive translational actions (jerkiness of commands)
The simulated supervisor is later fitted to the ratios of these statistics relative to the "better" group.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def demo_stats(g):
    eef = g["obs"]["robot0_eef_pos"][:]
    obj = g["obs"]["object"][:][0, :3]
    act = g["actions"][:][:, :3]
    path = np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()
    straight = max(np.linalg.norm(obj - eef[0]), 1e-6)
    a = act[np.linalg.norm(act, axis=1) > 1e-3]
    cos = np.sum(a[1:] * a[:-1], 1) / (
        np.linalg.norm(a[1:], axis=1) * np.linalg.norm(a[:-1], axis=1)
    )
    return {
        "length": len(eef),
        "path_ratio": path / straight,
        "dir_noise": float(1 - cos.mean()),
    }


def task_stats(path):
    out = {}
    with h5py.File(path, "r") as f:
        for level in ("worse", "okay", "better"):
            names = [n.decode() for n in f["mask"][level][:]]
            rows = [demo_stats(f["data"][n]) for n in names]
            out[level] = {
                k: {
                    "mean": float(np.mean([r[k] for r in rows])),
                    "sd": float(np.std([r[k] for r in rows])),
                    "n": len(rows),
                }
                for k in rows[0]
            }
    for level in ("worse", "okay", "better"):
        out[level]["ratio_to_better"] = {
            k: out[level][k]["mean"] / out["better"][k]["mean"]
            for k in ("length", "path_ratio", "dir_noise")
        }
    return out


if __name__ == "__main__":
    res = {
        t: task_stats(ROOT / f"data/robomimic/{t}_mh_low_dim_v15.hdf5")
        for t in ("lift", "can")
    }
    dst = ROOT / "data/robomimic/operator_stats.json"
    dst.write_text(json.dumps(res, indent=2))
    json.dump(res, sys.stdout, indent=2)
