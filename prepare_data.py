"""Featurize MP 2018.6 or MP 2022 into train/val/test tensor files for train.py.

    python prepare_data.py mp2018 --moddata MP_2018.6 --out data/mp2018
    MP_API_KEY=... python prepare_data.py mp2022 --out data/mp2022
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import warnings
from pathlib import Path

import torch

from crystalx.data import CrystalFeaturizer, load_mp2018, load_mp2022

_featurizer, _records = None, None


def _featurize(i):
    warnings.filterwarnings("ignore")
    cif_id, target, structure = _records[i]
    return _featurizer(structure, target, cif_id)


def make_split(ids, args):
    """Benchmark split from a file (MP 2018) or a seeded random split (MP 2022)."""
    rng = random.Random(args.seed)
    if args.split:
        with open(args.split) as f:
            split = json.load(f)
        train, test = list(split["train"]), list(split["test"])
    else:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_test = int(args.test_ratio * len(shuffled))
        train, test = shuffled[:len(shuffled) - n_test], shuffled[len(shuffled) - n_test:]
    if args.val_ratio > 0:
        train = list(train)
        rng.shuffle(train)
        n_val = int(args.val_ratio * (len(train) + len(test)))
        val, train = train[:n_val], train[n_val:]
    else:
        val = test
    return {"train": train, "val": val, "test": test}


def main():
    global _featurizer, _records
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["mp2018", "mp2022"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--moddata", help="path to the MODNet MP_2018.6 file (mp2018)")
    ap.add_argument("--api-key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key (mp2022)")
    ap.add_argument("--split", help="JSON with train/test ids; default for mp2018: splits/mp2018_benchmark.json")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="fraction of the dataset taken from train for validation; 0 = validate on test")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="test fraction for a random split (mp2022)")
    ap.add_argument("--atom-init", default=str(Path(__file__).parent / "atom_init.json"))
    ap.add_argument("--max-num-nbr", type=int, default=15)
    ap.add_argument("--radius", type=float, default=20)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    metadata = {"dataset": args.dataset}
    if args.dataset == "mp2018":
        assert args.moddata, "--moddata is required for mp2018"
        args.split = args.split or str(Path(__file__).parent / "splits" / "mp2018_benchmark.json")
        ids, targets, structures = load_mp2018(args.moddata)
    else:
        assert args.api_key, "set MP_API_KEY or pass --api-key for mp2022"
        ids, targets, structures, metadata["mp_database_version"] = load_mp2022(args.api_key)
    print(f"[data] {args.dataset}: {len(ids)} crystals", flush=True)

    split = make_split(ids, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split.json", "w") as f:
        json.dump(split, f)

    _featurizer = CrystalFeaturizer(args.atom_init, args.max_num_nbr, args.radius, step=args.step)
    by_id = {i: (i, t, s) for i, t, s in zip(ids, targets, structures)}
    for name in ["train", "val", "test"]:
        if name == "val" and split["val"] is split["test"]:
            continue
        _records = [by_id[i] for i in split[name]]
        with mp.get_context("fork").Pool(args.workers) as pool:
            samples = pool.map(_featurize, range(len(_records)), chunksize=16)
        meta = {**metadata, "split": name, "n_samples": len(samples),
                "orig_atom_fea_len": int(samples[0][0][0].shape[1]),
                "nbr_fea_len": int(samples[0][0][1].shape[-1])}
        torch.save({"samples": samples, "metadata": meta}, out / f"{name}.pt")
        print(f"[data] wrote {out / f'{name}.pt'} ({len(samples)} crystals)", flush=True)
        if name == "test" and split["val"] is split["test"]:
            torch.save({"samples": samples, "metadata": {**meta, "split": "val(=test)"}}, out / "val.pt")
            print(f"[data] wrote {out / 'val.pt'} (= test)", flush=True)


if __name__ == "__main__":
    main()
