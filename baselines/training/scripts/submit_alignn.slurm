#!/bin/bash --login
# ALIGNN band-gap (MODNet MP_2018.6) — canonical Matbench arch (4+4, hidden 256),
# full model (no subspace), explicit split matching the tensors, best-checkpoint fix.
# Override for a smoke run:  EPOCHS=3 ROOT=.../alignn_mp2018_bandgap_smoke OUT=... sbatch ...
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name=cx_alignn_bg
#SBATCH --output=logs/alignn_bg_%j.out
#SBATCH --error=logs/alignn_bg_%j.err

set -euo pipefail
mkdir -p logs
module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

BASE=/ibex/user/shoaibsa/Crystal-X/baselines/training
ROOT="${ROOT:-$BASE/data/alignn_mp2018_bandgap}"    # dir containing id_prop.json
CONFIG="${CONFIG:-../configs/config_bandgap_matbench.json}"
OUT="${OUT:-$BASE/results/alignn_bandgap}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-123}"

echo "Node: $(hostname)  Start: $(date -Iseconds)"
echo "ROOT=$ROOT  CONFIG=$CONFIG  OUT=$OUT  EPOCHS=$EPOCHS  SEED=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"

cd "$BASE/alignn_run/alignn"
python train_alignn.py \
  --root_dir "$ROOT" \
  --config_name "$CONFIG" \
  --target_key band_gap \
  --id_key material_id \
  --subspace_method none \
  --id_enable False \
  --id_dim 1.0 \
  --epochs "$EPOCHS" \
  --output_dir "$OUT" \
  --random_seed "$SEED" \
  --split_seed "$SEED"

echo "End: $(date -Iseconds)"
echo "Test CSV: $OUT/prediction_results_test_set.csv"
