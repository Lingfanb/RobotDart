"""500-step sanity training — verify FlowDART-HOI pipeline runs and loss decreases.

Trains a small ``HOIDenoiser`` on the 194-NPZ val mini-dataset under a flow
matching objective:

    x_t = (1 - t) * x_0 + t * noise
    v_target = noise - x_0          # constant-velocity FM ground truth
    loss = MSE(model(x_t, t, obj, cat), v_target)

Outcome we look for: loss curve drops meaningfully from initial (≈ var of
motion) to ≪1 within 500 steps.  This is *not* a quality benchmark — just
proof the training loop works end-to-end.

Run:
    python -m ManipAgent.train.g1_hoi_sanity --steps 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ManipAgent.data.g1_hoi import G1HOIDataset, collate
from ManipAgent.model.denoiser_hoi import HOIDenoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/home/lingfanb/Gitcode/DART/data/processed/g1_hoi_npz/val"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/lingfanb/Gitcode/DART/outputs/runs/vadmanip_sanity"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads",  type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ── data ───────────────────────────────────────────────────────────────
    ds = G1HOIDataset(args.data)
    print(f"dataset: {len(ds)} clips from {args.data}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=0, drop_last=True, collate_fn=collate)
    n_batches_per_epoch = len(dl)
    print(f"batches per epoch: {n_batches_per_epoch}")

    # quick stats for normalisation
    motions = torch.stack([ds[i]["motion"] for i in range(len(ds))])  # (N, T, 43)
    motion_mean = motions.mean(dim=(0, 1))
    motion_std = motions.std(dim=(0, 1)).clamp(min=1e-4)
    print(f"motion mean range: [{motion_mean.min():.3f}, {motion_mean.max():.3f}]")
    print(f"motion std  range: [{motion_std.min():.3f}, {motion_std.max():.3f}]")

    # ── model ──────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    model = HOIDenoiser(motion_dim=43, obj_dim=9, num_categories=13,
                         hidden=args.hidden, num_layers=args.layers,
                         num_heads=args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: HOIDenoiser, {n_params/1e6:.2f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── train loop ─────────────────────────────────────────────────────────
    loss_hist: list[tuple[int, float]] = []
    mean = motion_mean.to(device); std = motion_std.to(device)
    model.train()

    data_iter = iter(dl)
    t0 = time.time()
    running_loss = 0.0
    running_n = 0

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl); batch = next(data_iter)

        motion = batch["motion"].to(device)                   # (B, T, 43)
        obj = batch["object"].to(device)                       # (B, T, 9)
        cat = batch["object_cat"].to(device)                   # (B,)
        B, T, D = motion.shape

        # normalise motion
        x0 = (motion - mean) / std

        # flow-matching: sample t ~ U(0,1), x_t = (1-t)x0 + t*noise, v=noise - x0
        t_diff = torch.rand(B, device=device)
        t_b = t_diff[:, None, None]
        noise = torch.randn_like(x0)
        x_t = (1 - t_b) * x0 + t_b * noise
        v_target = noise - x0

        v_pred = model(x_t, t_diff, obj, cat)
        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item() * B; running_n += B

        if step % args.log_every == 0 or step == 1:
            avg = running_loss / max(running_n, 1)
            dt = time.time() - t0
            print(f"  step {step:4d}/{args.steps}   loss {avg:.5f}   ({dt:.1f}s, "
                  f"{step / dt:.1f} it/s)")
            loss_hist.append((step, avg))
            running_loss = 0.0; running_n = 0

    # ── save ───────────────────────────────────────────────────────────────
    ckpt_path = args.out / "model_sanity.pt"
    torch.save({
        "model": model.state_dict(),
        "motion_mean": mean.cpu(),
        "motion_std": std.cpu(),
        "args": vars(args) | {"data": str(args.data), "out": str(args.out)},
        "loss_hist": loss_hist,
    }, ckpt_path)
    print(f"\nsaved ckpt → {ckpt_path}")

    # plot loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [s for s, _ in loss_hist]; ls = [l for _, l in loss_hist]
        plt.figure(figsize=(7, 4))
        plt.plot(steps, ls, "o-")
        plt.xlabel("step"); plt.ylabel("MSE(velocity, noise-x0)")
        plt.title(f"ManipAgent HOI sanity ({len(ds)} clips, {n_params/1e6:.2f} M params)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plot_path = args.out / "loss_curve.png"
        plt.savefig(plot_path); plt.close()
        print(f"saved loss curve → {plot_path}")
    except Exception as e:
        print(f"could not save plot: {e}")

    # one-line summary
    initial = loss_hist[0][1]; final = loss_hist[-1][1]
    drop = (initial - final) / initial * 100
    print(f"\nSummary: loss {initial:.4f} → {final:.4f}  ({drop:.1f}% drop)")
    print("PASS" if final < initial * 0.7 else "WARNING — loss didn't drop ≥ 30%")


if __name__ == "__main__":
    main()
