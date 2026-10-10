#!/bin/bash --login
# Crystal-X band-gap (MODNet MP_2018.6) — 6 conv layers, AdamW lr 1e-4, LOSS=l2|l1|smoothl1.
# Override for a smoke run:  EPOCHS=3 DATA=.../mp2018_bandgap_smoke/seed123 OUT=... sbatch ...
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=cx_crystalx_bg
#SBATCH --output=logs/crystalx_bg_%j.out
#SBATCH --error=logs/crystalx_bg_%j.err

set -euo pipefail
mkdir -p logs
module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

BASE=/ibex/user/shoaibsa/Crystal-X/baselines/training
DATA="${DATA:-$BASE/data/mp2018_bandgap/seed123}"   # dir with train.pt/val.pt/test.pt
OUT="${OUT:-$BASE/results/crystalx_run}"
LOSS="${LOSS:-l2}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-123}"
BS="${BS:-256}"

echo "Node: $(hostname)  Start: $(date -Iseconds)"
echo "DATA=$DATA  OUT=$OUT  EPOCHS=$EPOCHS  SEED=$SEED  BS=$BS  LOSS=$LOSS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$BASE/../.."
python train.py \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BS" \
  --n-conv 6 --atom-fea-len 64 --h-fea-len 256 \
  --loss "$LOSS" --lr 1e-4 --weight-decay 1e-5 --lr-milestones 100 \
  --seed "$SEED" --workers 8

echo "End: $(date -Iseconds)"
echo "Test CSV: $OUT/test_results.csv"
