#!/bin/bash --login
# Crystal-X band-gap (MODNet MP_2018.6) — 8 conv layers, natural recipe, best-on-val.
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
OUT="${OUT:-$BASE/results/crystalx_bandgap}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-123}"
BS="${BS:-256}"

echo "Node: $(hostname)  Start: $(date -Iseconds)"
echo "DATA=$DATA  OUT=$OUT  EPOCHS=$EPOCHS  SEED=$SEED  BS=$BS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$BASE/crystalx_run"
python train_crystalx.py \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BS" \
  --n-conv 8 --atom-fea-len 64 --h-fea-len 256 \
  --lr 0.001 --weight-decay 1e-5 --lr-milestones 100 \
  --seed "$SEED" --workers 8

echo "End: $(date -Iseconds)"
echo "Test CSV: $OUT/crystalx_mp2018_bandgap_test_results.csv"
