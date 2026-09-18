# Robot transparency in human-in-the-loop robot learning

Code and derived data for the study *Robot Transparency Reduces Supervision Cost and Worst-Case
Risk Rather than Average Performance*.

The study asks what a learning robot gains by communicating its own uncertainty to the human who
supervises it. A soft actor--critic learner is trained online on a simulated Franka Panda
pick-and-place task while a parameterised simulated supervisor watches it, decides when to take
over, and corrects it. Six transparency conditions and one interactive-imitation baseline are
compared across a population of 33 supervisor profiles, for a total of 1302 runs. Every take-over
is labelled necessary or unnecessary by restoring the exact simulator state and rolling the policy
forward alone.

No human participants were involved at any stage.

## Repository contents

```
src/uhitl/            implementation and analysis package
  envs.py             environment construction and exact MuJoCo state snapshots
  agent.py            soft actor--critic with a critic ensemble and an auxiliary behaviour-cloning loss
  mechanisms.py       uncertainty estimation, percentile normalisation, request rules, novelty gate
  supervisor.py       the simulated supervisor: attention, decision, delay, control, trust
  train.py            one run: interaction, counterfactual judging, logging
  sweep.py            construction and resumable parallel execution of the experiment grid
  pretrain.py         produces the checkpoint that every run starts from
  fit_skill.py        fits the three operator skill tiers to the robomimic proficiency tiers
  calibration.py      calibration helpers
  metrics.py          per-run outcome metrics
  analyze.py          aggregation and paired statistics
  analysis_full.py    per-experiment analysis and supplementary figures
  confirmatory.py     configuration-level confirmatory tests reported in the paper
  style.py            shared figure style
  figures*.py         publication figures
  visuals.py          illustrative renders of the task and of the counterfactual judge
  uncertainty_eval.py open-loop predictive evaluation of the candidate uncertainty signals

calibration inputs
  data/robomimic/skill_fit.json          fitted skill tiers and the statistics they were matched to
  data/robomimic/operator_stats.json     per-operator statistics extracted from the source dataset
  data/literature/supervisor_parameters.yaml   every supervisor parameter with its source and status

results/processed/    per-run outcome metrics for every finished run (the input to the analysis)
results/tables/       the derived tables behind every number reported in the paper
results/pretrain/     policy weights of the shared starting checkpoint
scripts/reproduce.sh  the full pipeline in the order it was run
```

## Installation

Python 3.10 is required. MuJoCo renders offscreen through EGL; on a headless machine set
`MUJOCO_GL=egl`, which the code does by default.

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

The confirmatory analysis reported in the paper can be reproduced from the shipped per-run metrics,
without running any simulation:

```
PYTHONPATH=src python -m uhitl.confirmatory
```

This recomputes the configuration-level paired tests, the equivalence tests, the tail statistics,
the disclosed attention confound and the diagnosis of the novelty gate, and rewrites
`results/tables/confirmatory_*.csv`.

Reproducing the experiments themselves requires the starting checkpoint and roughly one GPU-day:

```
PYTHONPATH=src python -m uhitl.pretrain                       # produces results/pretrain/checkpoint.pt
PYTHONPATH=src python -m uhitl.sweep S2 --workers 8
PYTHONPATH=src python -m uhitl.analysis_full S1 S2 S3 S4 S6
PYTHONPATH=src python -m uhitl.confirmatory
```

`scripts/reproduce.sh` runs the same sequence for all five experiments.

## What is and is not shipped

`results/pretrain/policy_weights.pt` contains the network parameters of the checkpoint that every
run starts from, at an autonomous success rate of 0.37. It is sufficient for the illustrative
figures and for inspecting the policy. It does **not** contain the pre-training replay buffers,
which are part of the full checkpoint and are loaded into the learner at the start of every run.
Runs started from the weights alone therefore begin with empty buffers and will not reproduce the
reported numbers exactly; `uhitl.pretrain` regenerates the full checkpoint.

The robomimic multi-human datasets used to fit the operator skill tiers are not redistributed here.
`data/robomimic/skill_fit.json` and `operator_stats.json` record the fitted values and the
statistics they were matched against, so the calibration can be checked without the raw data.
`fit_skill.py` regenerates the fit once `lift_mh_low_dim_v15.hdf5` and `can_mh_low_dim_v15.hdf5`
have been obtained from the robomimic project.

Per-run logs for the 1302 runs are approximately 12 GB and are not included; the aggregated tables
in `results/tables/` contain every quantity reported in the paper.

## Provenance of the supervisor model

Four of the ten supervisor parameters are anchored to published measurements and the remaining
six are modelling choices that are swept across the population. The distinction is recorded
parameter by parameter in `data/literature/supervisor_parameters.yaml` and is reproduced in the
paper. The skill tiers are fitted to the three proficiency tiers of the robomimic multi-human
datasets, the take-over delay distribution is anchored to the pooled mean of a meta-analysis of 129
take-over studies, and the trust dynamics follow a published Bayesian Beta model.

## Determinism

Rollouts are bit-reproducible only when PyTorch runs single-threaded; `uhitl.visuals` sets
`torch.set_num_threads(1)` for this reason. Without it, repeated runs of the same seed diverge.

## Licence

The code is released under the MIT Licence; see `LICENSE`. The robomimic datasets and the
gym-hil environment are distributed by their own authors under their own terms.
