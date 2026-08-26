"""Space group for each of the 6,848 MODNet MP_2018.6 test crystals.

Structures are rebuilt cleanly from the ALIGNN id_prop.json (lattice_mat / coords /
elements) -- the same crystals the models trained on. We must NOT analyze the raw
MODData pickle objects directly: they were pickled by a 2020-era pymatgen and their
internal state is stale under the current version (site.is_ordered raises), which makes
SpacegroupAnalyzer throw for most entries.

No symmetry fallback: if spglib genuinely cannot resolve a structure we record None and
drop it, rather than mislabelling it P1.
"""
import csv
import json
from pathlib import Path

BASE = Path("/ibex/user/shoaibsa/Crystal-X/baselines/training")
IDPROP = BASE / "data/alignn_mp2018_bandgap/id_prop.json"
OUT = BASE / "data/mp2018_test_spacegroups.csv"
N_TRAIN, N_VAL = 61640, 6848


def crystal_system(n):
    if n <= 2: return "Triclinic"
    if n <= 15: return "Monoclinic"
    if n <= 74: return "Orthorhombic"
    if n <= 142: return "Tetragonal"
    if n <= 167: return "Trigonal"
    if n <= 194: return "Hexagonal"
    return "Cubic"


def main():
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    recs = json.load(open(IDPROP))
    test = recs[N_TRAIN + N_VAL:]          # the test block
    print(f"test records: {len(test)}", flush=True)

    rows, unresolved = [], 0
    for k, r in enumerate(test):
        a = r["atoms"]
        st = Structure(
            Lattice(a["lattice_mat"]),
            a["elements"],
            a["coords"],
            coords_are_cartesian=bool(a.get("cartesian", False)),
        )
        sg = None
        for prec in (0.01, 0.1, 0.3):
            try:
                sg = SpacegroupAnalyzer(st, symprec=prec).get_space_group_number()
                if sg:
                    break
            except Exception:
                continue
        if not sg:
            unresolved += 1
            continue                        # drop, never mislabel as P1
        rows.append((r["material_id"], sg, crystal_system(sg)))
        if k % 1000 == 0:
            print(f"  {k}/{len(test)}", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("id", "space_group", "crystal_system"))
        w.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)}/{len(test)} resolved, {unresolved} unresolved (dropped)",
          flush=True)
    from collections import Counter
    for k, v in Counter(r[2] for r in rows).most_common():
        print(f"  {k:<14} {v:>5}  ({100*v/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
