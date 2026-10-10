"""Train Crystal-X on tensors written by prepare_data.py, or evaluate a saved checkpoint.

    python train.py --data-dir data/mp2018 --output-dir runs/mp2018 --loss l2
    python train.py --data-dir data/mp2018 --output-dir runs/eval --eval-only path/to/model_best.pth
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from crystalx.data import TensorDataset, collate_pool
from crystalx.model import CrystalGraphConvNet

LOSSES = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smoothl1": nn.SmoothL1Loss}


class Normalizer:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def norm(self, t):
        return (t - self.mean) / self.std

    def denorm(self, t):
        return t * self.std + self.mean


def move_batch(inp, target, device):
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inp
    return (atom_fea.to(device), nbr_fea.to(device), nbr_fea_idx.to(device),
            [c.to(device) for c in crystal_atom_idx]), target.to(device)


def run_epoch(loader, model, criterion, optimizer, normalizer, device, train):
    model.train(train)
    tot_mae, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for inp, target, _ in loader:
            inp, target = move_batch(inp, target, device)
            output = model(*inp)
            loss = criterion(output, normalizer.norm(target))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pred = normalizer.denorm(output.detach())
            tot_mae += torch.mean(torch.abs(target - pred)).item() * target.size(0)
            n += target.size(0)
    return tot_mae / n


@torch.no_grad()
def predict(loader, model, normalizer, device):
    model.eval()
    ids, targets, preds = [], [], []
    for inp, target, cif_ids in loader:
        inp, target = move_batch(inp, target, device)
        preds += normalizer.denorm(model(*inp)).cpu().view(-1).tolist()
        targets += target.cpu().view(-1).tolist()
        ids += list(cif_ids)
    return ids, targets, preds


def build_model(state_dict):
    """Model config from the checkpoint when available (released weights carry none)."""
    n_conv = len({k.split(".")[1] for k in state_dict if k.startswith("convs.")})
    orig_atom_fea_len = state_dict["embedding.weight"].shape[1]
    nbr_fea_len = state_dict["convs.0.edge_mlp.0.weight"].shape[1]
    atom_fea_len = state_dict["embedding.weight"].shape[0]
    h_fea_len = state_dict["conv_to_fc.weight"].shape[0]
    return CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=atom_fea_len,
                               n_conv=n_conv, h_fea_len=h_fea_len)


def write_predictions(path, ids, targets, preds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("id", "target", "prediction"))
        w.writerows(zip(ids, targets, preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train.pt/val.pt/test.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--loss", choices=sorted(LOSSES), default="l2")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lr-milestones", type=int, nargs="+", default=[100])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-conv", type=int, default=6)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--h-fea-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-only", metavar="CHECKPOINT", help="skip training; evaluate this checkpoint on test.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                            num_workers=args.workers, collate_fn=collate_pool,
                                            pin_memory=device.type == "cuda")
    test_ds = TensorDataset(data_dir / "test.pt")
    test_loader = loader(test_ds, False)

    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        model = build_model(ckpt["state_dict"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        normalizer = Normalizer(ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"])
        ids, targets, preds = predict(test_loader, model, normalizer, device)
        print(f"[test] MAE = {np.mean(np.abs(np.array(targets) - np.array(preds))):.4f} eV "
              f"({len(ids)} crystals)", flush=True)
        write_predictions(out / "test_results.csv", ids, targets, preds)
        return

    train_ds = TensorDataset(data_dir / "train.pt")
    val_ds = TensorDataset(data_dir / "val.pt")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    train_loader, val_loader = loader(train_ds, True), loader(val_ds, False)

    # target normalizer from 500 random training crystals
    idx = random.sample(range(len(train_ds)), min(500, len(train_ds)))
    sample_target = torch.stack([train_ds[i][1] for i in idx])
    normalizer = Normalizer(torch.mean(sample_target), torch.std(sample_target))

    (af, nf, _), _, _ = train_ds[0]
    model = CrystalGraphConvNet(af.shape[-1], nf.shape[-1], atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv, h_fea_len=args.h_fea_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] orig_atom_fea_len={af.shape[-1]} nbr_fea_len={nf.shape[-1]} n_conv={args.n_conv} "
          f"loss={args.loss} lr={args.lr} params={n_params}", flush=True)

    criterion = LOSSES[args.loss]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    best_val, best_path = float("inf"), out / "model_best.pth"
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, model, criterion, optimizer, normalizer, device, True)
        va = run_epoch(val_loader, model, criterion, optimizer, normalizer, device, False)
        scheduler.step()
        if va != va:
            raise SystemExit("Exit due to NaN")
        is_best = va < best_val
        if is_best:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                        "epoch": epoch, "val_mae": va, "args": vars(args)}, best_path)
        print(f"epoch {epoch:3d}  train_MAE={tr:.4f}  val_MAE={va:.4f}{'  *best' if is_best else ''}", flush=True)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ids, targets, preds = predict(test_loader, model, normalizer, device)
    print(f"[test] MAE = {np.mean(np.abs(np.array(targets) - np.array(preds))):.4f} eV "
          f"(best epoch {ckpt['epoch']})", flush=True)
    write_predictions(out / "test_results.csv", ids, targets, preds)


if __name__ == "__main__":
    main()
