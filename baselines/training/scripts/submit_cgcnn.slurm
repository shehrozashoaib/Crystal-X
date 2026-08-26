#!/bin/bash --login
# CGCNN band-gap (MODNet MP_2018.6) — canonical arch, full model, best-on-val selection.
# Override for a smoke run:  EPOCHS=3 DATA=.../mp2018_bandgap_smoke OUT=... sbatch ... submit_cgcnn.slurm
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=cx_cgcnn_bg
#SBATCH --output=logs/cgcnn_bg_%j.out
#SBATCH --error=logs/cgcnn_bg_%j.err

set -euo pipefail
mkdir -p logs
module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

BASE=/ibex/user/shoaibsa/Crystal-X/baselines/training
DATA="${DATA:-$BASE/data/mp2018_bandgap}"          # dir containing seed<seed>/{train,val,test}.pt
OUT="${OUT:-$BASE/results/cgcnn_bandgap}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-123}"
BS="${BS:-256}"

echo "Node: $(hostname)  Start: $(date -Iseconds)"
echo "DATA=$DATA  OUT=$OUT  EPOCHS=$EPOCHS  SEED=$SEED  BS=$BS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$BASE/cgcnn_run"
python main.py \
  --cached-data-dir "$DATA" \
  --matbench-task mp2018_bandgap \
  --data-seed "$SEED" --random-seed "$SEED" \
  --task regression --optim Adam \
  --atom-fea-len 64 --h-fea-len 128 --n-conv 4 --n-h 1 \
  --epochs "$EPOCHS" --batch-size "$BS" --lr 0.001 --lr-milestones 100 \
  --workers 8 \
  --output-dir "$OUT"

echo "End: $(date -Iseconds)"
echo "Test CSV: $OUT/mp2018_bandgap_test_results.csv"
