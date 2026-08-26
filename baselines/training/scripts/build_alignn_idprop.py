"""Build ALIGNN id_prop.json for the exact saved-tensor train/test split.

Structures come from the MP_2018.6 MODData, simplified EXACTLY as Crystal-X's
data_MP_2018.py does (first element per site, frac coords) so ALIGNN sees the same
crystals that produced the tensors. Targets + split come from the saved tensors
(st_meta cif_ids/targets), guaranteeing an identical split to CGCNN/Crystal-X.

Output list ordering (for ALIGNN keep_data_order=true + explicit n_train/n_val/n_test):
    [ train (61,640) , val==test (6,848) , test (6,848) ]   total 75,336
so get_id_train_val_test yields id_train=[:61640], id_val=[61640:68488], id_test=[68488:].
"""
import glob
import json
import re
import sys
import types
from pathlib import Path

import torch

MODD = "/tmp/claude-224900/-ibex-user-shoaibsa/eede28e9-0580-4747-b6d5-ba7ee386036a/scratchpad/modd/MP_2018.6"
STMETA = "/tmp/claude-224900/-ibex-user-shoaibsa/eede28e9-0580-4747-b6d5-ba7ee386036a/scratchpad/st_meta/saved_tensors"
OUT = Path("/ibex/user/shoaibsa/Crystal-X/baselines/training/data/alignn_mp2018_bandgap")


def ordered_split(prefix):
    """Return (ids, targets) in shard order for a split."""
    idf = sorted(glob.glob(f"{STMETA}/{prefix}_cif_ids_*.txt"),
                 key=lambda x: int(re.findall(r"(\d+)\.txt", x)[0]))
    tgf = sorted(glob.glob(f"{STMETA}/{prefix}_target_*.pt"),
                 key=lambda x: int(re.findall(r"(\d+)\.pt", x)[0]))
    ids, tg = [], []
    for f in idf:
        ids += [l.strip() for l in open(f) if l.strip()]
    for f in tgf:
        tg += torch.load(f, map_location="cpu").view(-1).tolist()
    assert len(ids) == len(tg), f"{prefix}: {len(ids)} ids vs {len(tg)} targets"
    return ids, tg


def main():
    # ---- stub modnet, load MODData structures ----
    m = types.ModuleType("modnet")
    p = types.ModuleType("modnet.preprocessing")
    class MODData:  # noqa
        pass
    p.MODData = MODData
    m.preprocessing = p
    sys.modules["modnet"] = m
    sys.modules["modnet.preprocessing"] = p

    import pickle
    from pymatgen.core import Structure, Lattice, Element
    from jarvis.core.atoms import pmg_to_atoms

    print("loading MODData ...", flush=True)
    with open(MODD, "rb") as f:
        md = pickle.load(f)
    mpids = list(md.mpids)
    structures = md.structures
    print(f"MODData: {len(mpids)} structures", flush=True)

    # mpid -> JARVIS atoms dict (same simplification as data_MP_2018.py)
    print("converting structures -> JARVIS atoms ...", flush=True)
    atoms_by_id = {}
    for i, (mid, s) in enumerate(zip(mpids, structures)):
        simp = Structure(
            Lattice(s.lattice.matrix),
            [Element(str(site.species.elements[0])) for site in s.sites],
            [site.frac_coords for site in s.sites],
        )
        atoms_by_id[str(mid)] = pmg_to_atoms(simp).to_dict()
        if i % 10000 == 0:
            print(f"  {i}/{len(mpids)}", flush=True)

    tr_ids, tr_tg = ordered_split("train")
    te_ids, te_tg = ordered_split("test")
    print(f"split: train={len(tr_ids)} test={len(te_ids)}", flush=True)

    missing = [i for i in tr_ids + te_ids if i not in atoms_by_id]
    assert not missing, f"{len(missing)} ids missing structures, e.g. {missing[:5]}"

    def rec(mid, tgt):
        return {"material_id": mid, "band_gap": float(tgt), "atoms": atoms_by_id[mid]}

    records = (
        [rec(i, t) for i, t in zip(tr_ids, tr_tg)]      # train block
        + [rec(i, t) for i, t in zip(te_ids, te_tg)]    # val block  (== test)
        + [rec(i, t) for i, t in zip(te_ids, te_tg)]    # test block
    )
    n_train, n_val, n_test = len(tr_ids), len(te_ids), len(te_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "id_prop.json").open("w") as f:
        json.dump(records, f)
    with (OUT / "split_counts.json").open("w") as f:
        json.dump({"n_train": n_train, "n_val": n_val, "n_test": n_test,
                   "total": len(records),
                   "note": "val block == test block (reused); order train|val|test"},
                  f, indent=2)
    print(f"wrote {OUT/'id_prop.json'}  records={len(records)} "
          f"(n_train={n_train} n_val={n_val} n_test={n_test})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
