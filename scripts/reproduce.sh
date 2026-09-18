#!/usr/bin/env bash
# Full pipeline in the order it was run. Expects the dependencies in requirements.txt
# and a CUDA device; the main experiment is 1155 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export MUJOCO_GL=egl

WORKERS="${1:-8}"

python -m uhitl.fit_skill                       # operator skill tiers from the robomimic datasets
python -m uhitl.pretrain                        # shared starting checkpoint (autonomous success 0.37)
python -m uhitl.uncertainty_eval                # open-loop quality of the candidate uncertainty signals

for exp in S1 S2 S3 S4 S6; do
  python -m uhitl.sweep "$exp" --workers "$WORKERS"
done

python -m uhitl.analysis_full S1 S2 S3 S4 S6    # aggregation, per-experiment tables and figures
python -m uhitl.confirmatory                    # configuration-level tests reported in the paper
python -m uhitl.figures_paper                   # publication figures
python -m uhitl.visuals                         # illustrative renders
