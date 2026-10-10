#!/bin/bash --login
# ONE-GPU smoke: run all three models for a few epochs on the tiny subset datasets,
# on a real A100. Validates GPU execution, memory (esp. ALIGNN 4+4/hidden256 @ bs64),
# and gives a real per-epoch time so we can size the full-run walltimes. ~<1h.
#   sbatch scripts/submit_smoke_all.slurm
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=cx_smoke_all
#SBATCH --output=logs/smoke_all_%j.out
#SBATCH --error=logs/smoke_all_%j.err

set -uo pipefail
mkdir -p logs
module purge
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export ALIGNN_DETERMINISTIC=0

BASE=/ibex/user/shoaibsa/Crystal-X/baselines/training
SM=$BASE/results/smoke
mkdir -p "$SM"
echo "Node: $(hostname)  Start: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==================== [1/3] CGCNN smoke (3 ep) ===================="
cd "$BASE/cgcnn_run"
python main.py \
  --cached-data-dir "$BASE/data/mp2018_bandgap_smoke" \
  --matbench-task mp2018_bandgap --data-seed 123 --random-seed 123 \
  --task regression --optim Adam \
  --atom-fea-len 64 --h-fea-len 128 --n-conv 4 --n-h 1 \
  --epochs 3 --batch-size 256 --workers 8 \
  --output-dir "$SM/cgcnn" && echo "[CGCNN smoke OK]" || echo "[CGCNN smoke FAILED]"

echo "==================== [2/3] Crystal-X smoke (3 ep) ===================="
cd "$BASE/../.."
python train.py \
  --data-dir "$BASE/data/mp2018_bandgap_smoke/seed123" \
  --output-dir "$SM/crystalx" \
  --epochs 3 --batch-size 256 --n-conv 6 --atom-fea-len 64 --h-fea-len 256 --loss l2 --lr 1e-4 \
  --seed 123 --workers 8 && echo "[Crystal-X smoke OK]" || echo "[Crystal-X smoke FAILED]"

echo "==================== [3/3] ALIGNN smoke (3 ep) ===================="
cd "$BASE/alignn_run/alignn"
python train_alignn.py \
  --root_dir "$BASE/data/alignn_mp2018_bandgap_smoke" \
  --config_name ../configs/config_bandgap_matbench_smoke.json \
  --target_key band_gap --id_key material_id \
  --subspace_method none --id_enable False --id_dim 1.0 \
  --epochs 3 --output_dir "$SM/alignn" \
  --random_seed 123 --split_seed 123 && echo "[ALIGNN smoke OK]" || echo "[ALIGNN smoke FAILED]"

echo "==================== smoke summary ===================="
for m in cgcnn crystalx alignn; do
  echo "--- $m ---"; ls "$SM/$m"/*.csv 2>/dev/null || echo "  (no CSV)"
done
echo "End: $(date -Iseconds)"
