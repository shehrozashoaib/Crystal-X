import json
import pickle
import sys
import types
import warnings

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure
from torch.utils.data import Dataset


class GaussianDistance(object):
    """Expands interatomic distances (angstrom) on a Gaussian basis."""

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = step if var is None else var

    def expand(self, distances):
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomCustomJSONInitializer(object):
    """Element feature lookup from a JSON file mapping atomic number -> feature vector."""

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        self._embedding = {int(k): np.array(v, dtype=float) for k, v in elem_embedding.items()}

    def get_atom_fea(self, atom_type):
        assert atom_type in self._embedding, f"element Z={atom_type} not in atom_init.json"
        return self._embedding[atom_type]


class CrystalFeaturizer(object):
    """Builds the crystal graph ((atom_fea, nbr_fea, nbr_fea_idx), target, id) for one structure."""

    def __init__(self, atom_init_file, max_num_nbr=15, radius=20, dmin=0, step=0.2):
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self.max_num_nbr, self.radius = max_num_nbr, radius

    def __call__(self, crystal, target, cif_id):
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f"{cif_id} not find enough neighbors to build graph. "
                              "If it happens frequently, consider increase radius.")
                nbr_fea_idx.append([x[2] for x in nbr] + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append([x[1] for x in nbr] +
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append([x[2] for x in nbr[:self.max_num_nbr]])
                nbr_fea.append([x[1] for x in nbr[:self.max_num_nbr]])
        nbr_fea = self.gdf.expand(np.array(nbr_fea))
        return ((torch.Tensor(atom_fea), torch.Tensor(nbr_fea), torch.LongTensor(np.array(nbr_fea_idx))),
                torch.Tensor([float(target)]), cif_id)


def load_mp2018(moddata_file):
    """Read the MODNet MP_2018.6 band-gap file. Returns (ids, band_gaps, structures).

    The file is a pickled MODData object; a stub class is enough to unpickle it, so the
    modnet package (and its tensorflow dependency) is not required.
    """
    if "modnet.preprocessing" not in sys.modules:
        modnet, preprocessing = types.ModuleType("modnet"), types.ModuleType("modnet.preprocessing")
        preprocessing.MODData = type("MODData", (), {})
        modnet.preprocessing = preprocessing
        sys.modules["modnet"], sys.modules["modnet.preprocessing"] = modnet, preprocessing
    with open(moddata_file, "rb") as f:
        data = pickle.load(f)
    structures = [
        Structure(Lattice(s.lattice.matrix),
                  [Element(str(site.species.elements[0])) for site in s.sites],
                  [site.frac_coords for site in s.sites])
        for s in data.structures
    ]
    return [str(i) for i in data.mpids], data.df_targets["egap"].tolist(), structures


def load_mp2022(api_key):
    """Download all Materials Project entries with a band gap. Returns (ids, band_gaps, structures, db_version)."""
    from mp_api.client import MPRester

    with MPRester(api_key, use_document_model=False) as mpr:
        db_version = mpr.get_database_version()
        docs = mpr.materials.summary.search(band_gap=(0, 1e21),
                                            fields=["material_id", "structure", "band_gap"])
    docs = [d for d in docs if d.get("band_gap") is not None]
    return ([str(d["material_id"]) for d in docs], [d["band_gap"] for d in docs],
            [d["structure"] if isinstance(d["structure"], Structure) else Structure.from_dict(d["structure"])
             for d in docs], db_version)


class TensorDataset(Dataset):
    """Pre-featurized samples written by prepare_data.py."""

    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.samples = payload["samples"] if isinstance(payload, dict) else payload
        self.metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pool(dataset_list):
    """Batch crystal graphs: concatenate atoms, offset neighbor indices, keep per-crystal atom maps."""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target, batch_cif_ids = [], [], []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return ((torch.cat(batch_atom_fea, dim=0),
             torch.cat(batch_nbr_fea, dim=0),
             torch.cat(batch_nbr_fea_idx, dim=0),
             crystal_atom_idx),
            torch.stack(batch_target, dim=0),
            batch_cif_ids)
