# Matched-protocol benchmark: Crystal-X vs. CGCNN vs. ALIGNN

Crystal-X, CGCNN and ALIGNN trained **from scratch on the same data, the same split, and
evaluated on the same 6,848 test crystals**, so the three MAEs can be compared directly.

Dataset: MODNet **MP_2018.6** band gap, 68,488 crystals.
Split: **61,640 train / 6,848 test**, seed 123.
Hardware: one NVIDIA A100-SXM4-80GB per model (KAUST Ibex).

## Results

| model | MAE (eV) | RMSE (eV) | R² | wallclock | best epoch |
|---|---|---|---|---|---|
| **Crystal-X** | **0.2511** | **0.4983** | **0.9079** | 1.9 h | 129 |
| CGCNN | 0.2681 | 0.5105 | 0.9034 | 1.3 h | — |
| ALIGNN | 0.2632 | 0.5464 | 0.8893 | 11.5 h | — |

The three test CSVs are keyed by `mp_id` and are row-for-row aligned: same materials, same
order, same targets. Verified, not assumed.

### MAE by band-gap range

| range | n | Crystal-X | CGCNN | ALIGNN |
|---|---|---|---|---|
| metal (Eg = 0) | 2,263 | **0.1256** | 0.1370 | 0.1519 |
| 0 < Eg ≤ 1 | 1,376 | **0.2855** | 0.2965 | 0.3131 |
| 1 < Eg ≤ 3 | 1,948 | 0.3336 | 0.3605 | **0.3175** |
| Eg > 3 | 1,261 | **0.3113** | 0.3297 | 0.3248 |

### MAE by crystal system

| system | space groups | n | Crystal-X | CGCNN | ALIGNN |
|---|---|---|---|---|---|
| triclinic | 1–2 | 1,062 | **0.2982** | 0.3101 | 0.3235 |
| monoclinic | 3–15 | 1,766 | **0.2845** | 0.2932 | 0.2855 |
| orthorhombic | 16–74 | 1,317 | 0.2848 | 0.3002 | **0.2782** |
| tetragonal | 75–142 | 707 | **0.1935** | 0.2028 | 0.2070 |
| trigonal | 143–167 | 628 | **0.2664** | 0.2958 | 0.2932 |
| hexagonal | 168–194 | 459 | **0.1583** | 0.1849 | 0.1837 |
| cubic | 195–230 | 909 | **0.1634** | 0.1977 | 0.1911 |

Space groups resolved for all 6,848 test crystals from the JARVIS `megnet` source
(`scripts/spacegroups_mp2018_test.py`). Crystal-X leads in six of seven systems.

Machine-readable copies: `results/summary_metrics.csv`,
`results/metrics_by_gap_range.csv`, `results/metrics_by_crystal_system.csv`.

## Important caveat: validation split is the test split

`data/alignn_mp2018_bandgap/split_counts.json` records it explicitly — the 6,848 test
crystals are reused as the validation set, and all three models select their best
checkpoint on them. This is applied **identically to all three models**, so the ranking
between them is fair, but the absolute MAEs are optimistically biased. For Crystal-X the
size of that bias is measurable: best-on-val MAE 0.2511 vs. last-epoch val MAE 0.2564,
i.e. roughly 0.005 eV. Treat these as matched-protocol comparison numbers, not as
clean held-out generalization estimates.

## Training configuration

| | epochs | batch | seed | architecture | optimizer |
|---|---|---|---|---|---|
| Crystal-X | 200 | 256 | 123 | n_conv=8, atom_fea=64, h_fea=256 | AdamW 1e-3, wd 1e-5, SmoothL1, MultiStepLR[100] |
| CGCNN | 200 | 256 | 123 | n_conv=4, atom_fea=64, h_fea=128, n_h=1 | Adam 1e-3, MSE, MultiStepLR[100] |
| ALIGNN | 200 | 64 | 123 | alignn=4, gcn=4, hidden=256, emb=64 | AdamW 1e-3, MSE, OneCycle, 2000 warmup |

CGCNN and ALIGNN use their canonical/Matbench settings. All three are full models — no
Fastfood/subspace reduction (`--subspace_method none`).

### ALIGNN best-vs-last checkpoint fix

Upstream ALIGNN assigns `best_model = net`, a live reference that keeps training, so the
written test predictions reflect **last-epoch** weights rather than the best checkpoint.
`alignn_run/alignn/train.py` reloads `best_model.pt` before the final evaluation; the run
log shows the `[bestfix] reloaded BEST checkpoint` line confirming it took effect.

## Layout

```
crystalx_run/     train_crystalx.py + model.py (identical to the repo-root model.py)
cgcnn_run/        main.py + cgcnn package
alignn_run/       the three locally-shadowed ALIGNN modules + the config used
scripts/          slurm submission, dataset builders, space-group + plotting scripts
data/             split_counts.json, per-test-crystal space groups (bulk tensors excluded)
results/          test CSVs, metric summaries, trained weights, figure
logs/             stdout from the three production jobs (48967875/76/77)
```

## Reproducing

Dataset tensors and ALIGNN `id_prop.json` are tens of GB and are **not** in the repo.
Rebuild them first:

```bash
# CGCNN / Crystal-X tensors: un-collate the CGCNN-format saved tensors into
# per-crystal samples ((atom_fea, nbr_fea, nbr_fea_idx), target, mp_id)
python scripts/uncollate_saved_tensors.py

# ALIGNN id_prop.json — ordered [train 61640 | val 6848 | test 6848], val block == test block
python scripts/build_alignn_idprop.py
```

Then submit (GPU jobs must be submitted from a login node):

```bash
sbatch scripts/submit_smoke_all.slurm   # 3 epochs, tiny subset, all three models — run this first
sbatch scripts/submit_cgcnn.slurm
sbatch scripts/submit_crystalx.slurm
sbatch scripts/submit_alignn.slurm
```

Regenerate the figure and the space-group table:

```bash
python scripts/spacegroups_mp2018_test.py
python scripts/plot_picture1_new.py
```

### Environment

Python 3.12, PyTorch + CUDA, `dgl` (pytorch backend), `alignn==2026.5.20`,
`jarvis-tools`, `pymatgen`. All Python must run with
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (jarvis/scipy need the conda `libstdc++`); the sbatch
scripts set this.

ALIGNN note: `train_alignn.py` uses bare imports (`from data import ...`,
`from train import ...`), so the copies of `data.py` and `train.py` in `alignn_run/alignn/`
shadow the installed package when you run from that directory. Everything else
(`alignn.config`, `alignn.models.*`) comes from the pip install. Run ALIGNN from inside
`alignn_run/alignn/` or the fix above will not be picked up.

## A note on the earlier, superseded results

An earlier round of this comparison used **different test sets per model** and is not
included here. In it, ALIGNN's 0.218 eV came from ALIGNN's own published 4,239-material
JARVIS `megnet` gap-pbe split, which shares only ~11% of its materials with the 6,848-crystal
Crystal-X test set. Numbers from that round are not comparable across models and should not
be quoted; this directory supersedes them.
