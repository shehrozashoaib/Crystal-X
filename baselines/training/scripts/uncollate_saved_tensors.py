"""Un-collate the pre-batched saved_tensors.zip back into per-crystal CGCNN samples.

Each shard in the zip is a collate_pool OUTPUT (a whole batch). We invert that:
for every crystal, slice its atom rows out of the batched tensors and de-offset its
neighbor indices, recovering the original per-crystal sample
    ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
which is exactly what Crystal-X and (patched) CGCNN TensorCIFData consume.

Writes, for the single available split:
    <out>/seed<SEED>/train.pt   (61,640 samples)
    <out>/seed<SEED>/test.pt    ( 6,848 samples)
    <out>/seed<SEED>/val.pt     ( == test.pt; user asked to reuse test as val)
each as {"samples": [...], "metadata": {...}} (TensorCIFData dict format).

Memory: processed shard-by-shard; only the accumulated per-split sample list is held.
"""
import argparse
import glob
import io
import json
import re
import time
import zipfile
from pathlib import Path

import torch


def shard_indices(names, split, field):
    pat = re.compile(rf"saved_tensors/{split}_{field}_(\d+)\.(pt|txt)$")
    out = {}
    for n in names:
        m = pat.search(n)
        if m:
            out[int(m.group(1))] = n
    return out


def load_pt(zf, name):
    with zf.open(name) as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu")


def load_txt(zf, name):
    with zf.open(name) as f:
        return [l.strip() for l in f.read().decode().splitlines() if l.strip()]


def uncollate_split(zf, names, split):
    fields = ["atom_fea", "nbr_fea", "nbr_fea_idx", "crystal_atom_idx", "target", "cif_ids"]
    idx_maps = {fld: shard_indices(names, split, fld) for fld in fields}
    shard_ids = sorted(idx_maps["atom_fea"].keys())
    samples = []
    t0 = time.time()
    for si in shard_ids:
        atom_fea = load_pt(zf, idx_maps["atom_fea"][si])
        nbr_fea = load_pt(zf, idx_maps["nbr_fea"][si])
        nbr_fea_idx = load_pt(zf, idx_maps["nbr_fea_idx"][si])
        crystal_atom_idx = load_pt(zf, idx_maps["crystal_atom_idx"][si])
        target = load_pt(zf, idx_maps["target"][si])
        cif_ids = load_txt(zf, idx_maps["cif_ids"][si])
        assert len(crystal_atom_idx) == len(cif_ids) == target.shape[0], (
            f"{split} shard {si}: n_crys mismatch "
            f"{len(crystal_atom_idx)}/{len(cif_ids)}/{target.shape[0]}"
        )
        for k, idxmap in enumerate(crystal_atom_idx):
            base = int(idxmap[0])
            n_i = idxmap.shape[0]
            af = atom_fea[base:base + n_i].clone()
            nf = nbr_fea[base:base + n_i].clone()
            nfi = (nbr_fea_idx[base:base + n_i] - base).clone().long()
            tg = target[k].clone().view(1).float()
            samples.append(((af, nf, nfi), tg, cif_ids[k]))
        if si % 40 == 0:
            print(f"  [{split}] shard {si}/{shard_ids[-1]}  "
                  f"{len(samples)} crystals  {time.time()-t0:.0f}s", flush=True)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="/ibex/user/shoaibsa/Crystal-X/baselines/saved_tensors.zip")
    ap.add_argument("--out", default="/ibex/user/shoaibsa/Crystal-X/baselines/training/data/mp2018_bandgap")
    ap.add_argument("--seed", type=int, default=123,
                    help="label for the seed<> subdir; this is the DATA split label")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="smoke mode: only use the first N shards of each split")
    args = ap.parse_args()

    out_dir = Path(args.out) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        result = {}
        for split in ["train", "test"]:
            print(f"[{split}] un-collating ...", flush=True)
            if args.limit_shards:
                # keep only first N shards for a fast smoke build
                keep = set()
                for fld in ["atom_fea", "nbr_fea", "nbr_fea_idx",
                            "crystal_atom_idx", "target", "cif_ids"]:
                    si = shard_indices(names, split, fld)
                    for k in sorted(si)[:args.limit_shards]:
                        keep.add(si[k])
                sub = [n for n in names if n in keep]
            else:
                sub = names
            samples = uncollate_split(zf, sub, split)
            result[split] = samples
            print(f"[{split}] {len(samples)} crystals", flush=True)

    orig_len = torch.tensor([s[0][0].shape[1] for s in result["train"][:1]])
    meta_common = {
        "source": "saved_tensors.zip (MODNet MP_2018.6 band gap)",
        "orig_atom_fea_len": int(result["train"][0][0][0].shape[1]),
        "nbr_fea_len": int(result["train"][0][0][1].shape[-1]),
        "max_num_nbr": int(result["train"][0][0][1].shape[1]),
    }
    for split in ["train", "test"]:
        payload = {"samples": result[split],
                   "metadata": {**meta_common, "split": split, "seed": args.seed,
                                "n_samples": len(result[split])}}
        torch.save(payload, out_dir / f"{split}.pt")
        print(f"wrote {out_dir/f'{split}.pt'}  ({len(result[split])} samples)", flush=True)
    # val := test  (user: reuse test split as validation)
    payload = {"samples": result["test"],
               "metadata": {**meta_common, "split": "val(=test)", "seed": args.seed,
                            "n_samples": len(result["test"])}}
    torch.save(payload, out_dir / "val.pt")
    print(f"wrote {out_dir/'val.pt'}  (== test, {len(result['test'])} samples)", flush=True)

    with (out_dir / "metadata.json").open("w") as f:
        json.dump({"task": "mp2018_bandgap", "seed": args.seed,
                   "counts": {"train": len(result["train"]),
                              "val": len(result["test"]),
                              "test": len(result["test"])},
                   **meta_common}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
