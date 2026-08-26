"""Train Crystal-X on the MODNet MP_2018.6 band-gap tensors (natural settings).

Loads the un-collated per-crystal samples ((atom_fea,nbr_fea,nbr_fea_idx), target, cif_id),
selects the BEST model on the (=test) validation split, and writes a per-material test CSV
with mp_ids. Natural Crystal-X recipe from the paper notebook: SmoothL1Loss, AdamW
(lr 1e-3, wd 1e-5), MultiStepLR(milestones=[100], gamma=0.1), batch 256, n_conv=8,
atom_fea_len=64, h_fea_len=256.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

from model import CrystalGraphConvNet


# --------------------------------------------------------------------------- data
class TensorDataset(Dataset):
    def __init__(self, path):
        payload = torch.load(path, map_location="cpu")
        self.samples = payload["samples"] if isinstance(payload, dict) else payload
        self.metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pool(dataset_list):
    """Crystal-X collate: samples are ((atom_fea,nbr_fea,nbr_fea_idx), target, cif_id)."""
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
    return (
        (torch.cat(batch_atom_fea, dim=0),
         torch.cat(batch_nbr_fea, dim=0),
         torch.cat(batch_nbr_fea_idx, dim=0),
         crystal_atom_idx),
        torch.stack(batch_target, dim=0),
        batch_cif_ids,
    )


# ----------------------------------------------------------------------- normalizer
class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, t):
        return (t - self.mean) / self.std

    def denorm(self, t):
        return t * self.std + self.mean


def mae(pred, target):
    return torch.mean(torch.abs(target - pred))


# ---------------------------------------------------------------------------- loops
def move_batch(inp, target, device):
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inp
    return (
        atom_fea.to(device), nbr_fea.to(device), nbr_fea_idx.to(device),
        [c.to(device) for c in crystal_atom_idx],
    ), target.to(device)


def run_epoch(loader, model, criterion, optimizer, normalizer, device, train):
    model.train() if train else model.eval()
    tot_mae, n = 0.0, 0
    torch.set_grad_enabled(train)
    for inp, target, _ in loader:
        inp, target = move_batch(inp, target, device)
        target_normed = normalizer.norm(target)
        output = model(*inp)
        loss = criterion(output, target_normed)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = normalizer.denorm(output.detach())
        bs = target.size(0)
        tot_mae += mae(pred, target).item() * bs
        n += bs
    torch.set_grad_enabled(True)
    return tot_mae / n


@torch.no_grad()
def predict(loader, model, normalizer, device):
    model.eval()
    ids, targets, preds = [], [], []
    for inp, target, cif_ids in loader:
        inp, target = move_batch(inp, target, device)
        output = model(*inp)
        pred = normalizer.denorm(output).cpu().view(-1).tolist()
        ids += list(cif_ids)
        targets += target.cpu().view(-1).tolist()
        preds += pred
    return ids, targets, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train.pt/val.pt/test.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-conv", type=int, default=8)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--h-fea-len", type=int, default=256)
    ap.add_argument("--lr-milestones", type=int, nargs="+", default=[100])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    train_ds = TensorDataset(data_dir / "train.pt")
    val_ds = TensorDataset(data_dir / "val.pt")
    test_ds = TensorDataset(data_dir / "test.pt")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.workers, collate_fn=collate_pool,
                                   pin_memory=True)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    # normalizer from a 500-sample draw of train targets (as in the notebook)
    idx = random.sample(range(len(train_ds)), min(500, len(train_ds)))
    _, sample_target, _ = collate_pool([train_ds[i] for i in idx])
    normalizer = Normalizer(sample_target)
    print(f"[normalizer] mean={normalizer.mean:.4f} std={normalizer.std:.4f}", flush=True)

    (af, nf, _), _, _ = train_ds[0]
    orig_atom_fea_len, nbr_fea_len = af.shape[-1], nf.shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] orig_atom_fea_len={orig_atom_fea_len} nbr_fea_len={nbr_fea_len} "
          f"n_conv={args.n_conv} params={n_params}", flush=True)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    best_val = float("inf")
    best_path = out / "model_best.pth"
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, model, criterion, optimizer, normalizer, device, True)
        va = run_epoch(val_loader, model, criterion, optimizer, normalizer, device, False)
        scheduler.step()
        if va != va:  # NaN guard
            print("Exit due to NaN", flush=True)
            raise SystemExit(1)
        is_best = va < best_val
        if is_best:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                        "epoch": epoch, "val_mae": va}, best_path)
        print(f"epoch {epoch:3d}  train_MAE={tr:.4f}  val_MAE={va:.4f}"
              f"{'  *best' if is_best else ''}", flush=True)

    # ---- load BEST checkpoint, evaluate on test ----
    print(f"[test] loading best (val_MAE={best_val:.4f}) from {best_path}", flush=True)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    normalizer.mean = ckpt["normalizer"]["mean"]
    normalizer.std = ckpt["normalizer"]["std"]

    ids, targets, preds = predict(test_loader, model, normalizer, device)
    test_mae = float(np.mean(np.abs(np.array(targets) - np.array(preds))))
    print(f"[test] Crystal-X test MAE = {test_mae:.4f} eV  (best epoch {ckpt['epoch']})", flush=True)

    csv_path = out / "crystalx_mp2018_bandgap_test_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("id", "target", "prediction"))
        for i, t, p in zip(ids, targets, preds):
            w.writerow((i, t, p))
    print(f"[test] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
