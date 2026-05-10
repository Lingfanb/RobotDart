"""Train an action classifier on G1 motion primitives for unified evaluation.

Inputs: (T, 29) dof_angle time series (T variable; primitives are T=10).
Output: class logits over top-22 BABEL act_cats + OTHER (=23 classes total).

Usage:
    cd ~/Gitcode/DART
    CUDA_VISIBLE_DEVICES=1 python -m mld.train_action_classifier \\
        --exp_name action_clf_v1 \\
        --total_steps 30000 \\
        --batch_size 512 \\
        --no-wandb
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_PKL = Path("data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/train.pkl")
DEFAULT_VAL_PKL = Path("data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/val.pkl")
DEFAULT_SAVE_DIR = Path("outputs/checkpoints/action_classifier")

DOF_DIM = 29
DOF_SLICE = (11, 40)  # features_69[:, 11:40] = dof_angle
NUM_TOP_CLASSES = 22
OTHER_LABEL = "OTHER"


# ── Class taxonomy discovery ─────────────────────────────────────────────────

def discover_top_classes(samples, top_k: int = NUM_TOP_CLASSES) -> list[str]:
    c = Counter()
    for x in samples:
        for cat in (x.get("act_cats") or []):
            c[cat] += 1
    classes = [name for name, _ in c.most_common(top_k)]
    return classes


def pick_label(act_cats: list[str], class_to_idx: dict[str, int], other_idx: int) -> int:
    """Pick the most informative (rarest) act_cat label that is in our taxonomy.

    BABEL clips often carry both a coarse tag ("transition", "hand movements", "stand")
    and a specific tag ("wave", "kick", "jump"). The specific tag is much more useful
    for our action-classifier supervision, so prefer it. Within our top-K taxonomy,
    larger idx = rarer (since classes are sorted by descending frequency).
    """
    if not act_cats:
        return other_idx
    best_idx = None
    for cat in act_cats:
        idx = class_to_idx.get(cat)
        if idx is None:
            continue
        # Prefer the rarest (largest idx) match — more specific.
        if best_idx is None or idx > best_idx:
            best_idx = idx
    if best_idx is None:
        return other_idx
    return best_idx


# ── Dataset ──────────────────────────────────────────────────────────────────

class PrimitiveActionDataset(Dataset):
    def __init__(self, samples, class_to_idx: dict[str, int], other_idx: int):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.other_idx = other_idx
        self.labels = np.array([
            pick_label(s.get("act_cats") or [], class_to_idx, other_idx)
            for s in samples
        ], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        feats = s["features_69"]  # (T, 69)
        dof = feats[:, DOF_SLICE[0]:DOF_SLICE[1]].astype(np.float32)  # (T, 29)
        return torch.from_numpy(dof), int(self.labels[i])


def collate_pad(batch):
    """Pad variable-T sequences. Returns (x, lengths, mask, y)."""
    seqs, ys = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    T_max = int(lengths.max().item())
    B = len(seqs)
    x = torch.zeros(B, T_max, DOF_DIM, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)  # True = valid
    for i, s in enumerate(seqs):
        T = s.shape[0]
        x[i, :T] = s
        mask[i, :T] = True
    y = torch.tensor(ys, dtype=torch.long)
    return x, lengths, mask, y


# ── Model ────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        # x: (B, T, d)
        T = x.size(1)
        return x + self.pe[:, :T]


class ActionClassifier(nn.Module):
    def __init__(self, in_dim: int = DOF_DIM, h_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 4, ff_dim: int = 256,
                 num_classes: int = 23, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, h_dim)
        self.posenc = PositionalEncoding(h_dim, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=h_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(h_dim)
        self.head = nn.Linear(h_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, in_dim); mask: (B, T) True=valid
        h = self.in_proj(x)
        h = self.posenc(h)
        # transformer src_key_padding_mask wants True = padding (ignore)
        kpm = ~mask if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=kpm)
        h = self.norm(h)
        # masked mean pool
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)
        return self.head(pooled)


# ── Training ─────────────────────────────────────────────────────────────────

@dataclass
class CLFConfig:
    h_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    max_len: int = 512


def make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = inv[labels]
    return WeightedRandomSampler(weights=torch.from_numpy(weights).double(),
                                 num_samples=len(labels), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    per_class_correct = Counter()
    per_class_total = Counter()
    for x, lengths, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, mask=mask)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.item())
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
        for yi, pi in zip(y.cpu().tolist(), pred.cpu().tolist()):
            per_class_total[yi] += 1
            if yi == pi:
                per_class_correct[yi] += 1
    model.train()
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
        "per_class_total": dict(per_class_total),
        "per_class_correct": dict(per_class_correct),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="action_clf_v1")
    ap.add_argument("--train_pkl", type=str, default=str(DEFAULT_TRAIN_PKL))
    ap.add_argument("--val_pkl", type=str, default=str(DEFAULT_VAL_PKL))
    ap.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    ap.add_argument("--total_steps", type=int, default=30000)
    ap.add_argument("--val_interval", type=int, default=5000)
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--h_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--top_k", type=int, default=NUM_TOP_CLASSES)
    ap.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    ap.add_argument("--wandb_project", type=str, default="g1_action_classifier")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)

    print(f"[load] train pkl: {args.train_pkl}")
    with open(args.train_pkl, "rb") as f:
        train_samples = pickle.load(f)
    print(f"[load] val pkl: {args.val_pkl}")
    with open(args.val_pkl, "rb") as f:
        val_samples = pickle.load(f)
    print(f"  train n={len(train_samples)}  val n={len(val_samples)}")

    # Build taxonomy from train data
    top_classes = discover_top_classes(train_samples, top_k=args.top_k)
    class_names = top_classes + [OTHER_LABEL]
    class_to_idx = {n: i for i, n in enumerate(top_classes)}
    other_idx = len(top_classes)
    num_classes = len(class_names)
    print(f"[taxonomy] {num_classes} classes: {class_names}")

    # Save class names early
    with open(save_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    train_ds = PrimitiveActionDataset(train_samples, class_to_idx, other_idx)
    val_ds = PrimitiveActionDataset(val_samples, class_to_idx, other_idx)

    train_label_counts = Counter(train_ds.labels.tolist())
    val_label_counts = Counter(val_ds.labels.tolist())
    print("[label-dist] train:")
    for i, n in enumerate(class_names):
        print(f"  {i:2d} {n:<28}  train={train_label_counts.get(i, 0):>6}  val={val_label_counts.get(i, 0):>6}")

    sampler = make_weighted_sampler(train_ds.labels, num_classes)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_pad, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    cfg = CLFConfig(h_dim=args.h_dim, num_layers=args.num_layers,
                    num_heads=args.num_heads, ff_dim=args.ff_dim,
                    dropout=args.dropout, max_len=512)

    model = ActionClassifier(
        in_dim=DOF_DIM, h_dim=cfg.h_dim, num_layers=cfg.num_layers,
        num_heads=cfg.num_heads, ff_dim=cfg.ff_dim, num_classes=num_classes,
        max_len=cfg.max_len, dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.total_steps, 1), eta_min=args.learning_rate * 0.05,
    )

    use_wandb = args.use_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.exp_name,
                                   config={**vars(args), **vars(cfg), "num_classes": num_classes})
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            use_wandb = False

    model.train()
    step = 0
    best_val_acc = -1.0
    train_iter = iter(train_loader)
    while step < args.total_steps:
        try:
            x, lengths, mask, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, lengths, mask, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, mask=mask)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        if step % args.log_interval == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[step {step:>6}/{args.total_steps}] loss={loss.item():.4f} acc={acc:.3f} lr={lr_now:.2e}")
            if use_wandb and wandb_run is not None:
                wandb_run.log({"train/loss": float(loss.item()),
                               "train/acc": float(acc),
                               "lr": float(lr_now),
                               "step": step})

        if step % args.val_interval == 0 or step == args.total_steps:
            stats = evaluate(model, val_loader, device)
            print(f"[val   {step:>6}] loss={stats['val_loss']:.4f} acc={stats['val_acc']:.3f}")
            if use_wandb and wandb_run is not None:
                wandb_run.log({"val/loss": stats["val_loss"],
                               "val/acc": stats["val_acc"],
                               "step": step})
            if stats["val_acc"] > best_val_acc:
                best_val_acc = stats["val_acc"]
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": {
                        "h_dim": cfg.h_dim,
                        "num_layers": cfg.num_layers,
                        "num_heads": cfg.num_heads,
                        "ff_dim": cfg.ff_dim,
                        "dropout": cfg.dropout,
                        "max_len": cfg.max_len,
                        "in_dim": DOF_DIM,
                        "num_classes": num_classes,
                    },
                    "step": step,
                    "val_acc": float(stats["val_acc"]),
                }
                torch.save(ckpt, save_dir / "best.pt")
                print(f"  -> saved best.pt (val_acc={best_val_acc:.3f})")

    # Final dump
    ckpt = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "config": {
            "h_dim": cfg.h_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "ff_dim": cfg.ff_dim,
            "dropout": cfg.dropout,
            "max_len": cfg.max_len,
            "in_dim": DOF_DIM,
            "num_classes": num_classes,
        },
        "step": step,
        "val_acc": float(best_val_acc),
    }
    torch.save(ckpt, save_dir / "last.pt")
    print(f"[done] best_val_acc={best_val_acc:.3f}  saved best.pt + last.pt → {save_dir}")
    if use_wandb and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
