# 2026-04-09 — DDP support + dataset prefetch optimization

> Added DDP path to denoiser training and rewrote `G1PrimitiveSequenceDataset.get_batch` to eliminate
> the per-step Python loop bottleneck. Should speed up both single-GPU and multi-GPU training.

## Files changed
- [mld/train_g1_mld.py](../mld/train_g1_mld.py) — DDP support
- [data_loaders/humanml/data/dataset_g1.py](../data_loaders/humanml/data/dataset_g1.py) — precomputed batch tensors + pre-encoded text
- [run_denoiser_single_gpu.sh](../run_denoiser_single_gpu.sh) — new launch script (single-GPU)

## 1. DDP path in `mld/train_g1_mld.py`

Goal: launch with `torchrun --nproc_per_node=2` and split the global batch across GPUs.

Key changes:
- New `setup_ddp()` / `cleanup_ddp()` helpers reading `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from env. Pass `device_id` to `init_process_group` to silence the NCCL device-guess warning. Falls back to `(0, 1, 0)` for non-distributed runs so the old single-GPU path still works.
- Trainer `__init__` takes `rank, world_size, local_rank`. Per-rank seed = `args.seed + rank` so the two ranks sample independent data.
- `train_args.batch_size` is treated as **global** batch — per-rank = `batch_size // world_size`, asserted divisible.
- Dataset init uses rank-0-first + barrier so only rank 0 writes the `mean_std.pkl` cache.
- VAE / checkpoint loads pass `map_location=device` (otherwise everything piles onto cuda:0).
- EMA copy happens **before** DDP wrap (otherwise the EMA module would carry the `module.` parameter prefix). EMA update iterates over `denoiser_model_module.parameters()`, not the wrapper.
- `common_step()` takes an optional `denoiser_model` arg so validation can call the unwrapped module (only rank 0 runs validation, calling the DDP wrapper alone would deadlock).
- Stage-2+ rollout `p_sample_loop` always uses the unwrapped module — under `no_grad`, the DDP wrapper just adds overhead.
- `validate()`: rank 0 runs the loop, all other ranks call `dist.barrier()` and return. Final barrier syncs everyone before the next training step.
- `save()` / wandb / SummaryWriter / tqdm / config dump all guarded by `is_main`.

### Bugs hit along the way
- **`tyro` import error**: ran `torchrun` from `(base)` env instead of `(DART)`. Fix: `conda activate DART`.
- **VAE checkpoint path**: `./mvae/g1_vae_v1/...` doesn't exist — the actual dir is `./mvae/g1_mvae/`.
- **DDP `broadcast_coalesced` "single memory location" error**: `model/mld_denoiser.py` registers the same `PositionalEncoding` instance under both `self.sequence_pos_encoder` and `self.embed_timestep.sequence_pos_encoder`. Its `pe` buffer is therefore registered twice in the model under aliased storage, which DDP can't broadcast. Fix: pass `broadcast_buffers=False` when wrapping. The buffer is deterministic sin/cos, identical on every rank, so broadcasting it is unnecessary.

### DDP launch
```bash
torchrun --nproc_per_node=2 -m mld.train_g1_mld \
    --exp_name g1_mld_v3_ddp \
    --denoiser_args.mvae_path ./mvae/g1_mvae/checkpoint_300000.pt \
    --train_args.batch_size 1024
```

## 2. Dataset prefetch optimization (`data_loaders/humanml/data/dataset_g1.py`)

### Why
First DDP run hit ~3 it/s with asymmetric GPU utilization (cuda:0 73%, cuda:1 26%). Root cause was **CPU data prep on the critical path**, not GPU heterogeneity. Per `get_batch(512)` × `num_primitive=4` = 2048 inner Python iterations, each doing:
- `_data_to_tensor(data)` — 6× `torch.tensor(numpy_array)` + `cat`
- `.to(self.device)` — small H2D copy with PCIe launch latency
- `_get_text_embedding(text)` — dict lookup (or cold CLIP encode)
- + `torch.stack` of B small tensors at the end

Estimated ~30 ms of CPU work per `_build_primitive_batch` call → ~120 ms per outer step pinned on CPU, GPU starves waiting for the next batch.

### Changes
Three precompute steps added at the end of `__init__`:
1. **All motion → single GPU tensor**. `(N, T, D)` float32 (~960 MB for the 66k-primitive train split) moved to `self.device` once. Replaces all per-call `torch.tensor` + `.to(device)`.
2. **All unique texts pre-encoded via CLIP**. ~184 unique texts → `(184, 512)` table. Stored alongside a `text_to_idx` map.
3. **Per-primitive text-index list**. Each dataset entry stores its text indices into the table; `_get_text_embedding` is no longer called at runtime.

`_build_primitive_batch` rewritten:
- Takes a 1-D index array instead of a list of dicts.
- `motion_tensor = self.all_motion_tensor.index_select(0, indices_tensor)` — one GPU fancy-index op replaces 512 small H2D copies.
- Random-text choice is a tiny Python loop over the batch (just `random.randrange` + list lookup, no torch).
- `text_embeddings = self.all_text_embeddings.index_select(0, ...)` — one GPU lookup replaces 512 dict accesses.

`get_batch` updated to pass numpy index arrays (1D for `num_primitive=1`, 2D `(num_primitive, B)` otherwise) instead of building dict lists.

### Expected effect
Per-call CPU work ~30 ms → ~0.5 ms (~60×). Across `num_primitive=4`, that's ~120 ms saved per outer step, on top of removing the per-step PCIe traffic. Both single-GPU and DDP benefit. Memory cost: ~960 MB train + ~340 MB val per rank on the GPU.

Both `train_g1_mvae.py` and `train_g1_mld.py` use this dataset, so VAE training also benefits.

## 3. New launch script
[run_denoiser_single_gpu.sh](../run_denoiser_single_gpu.sh) — single-GPU launch matching the v2 hyperparameters (batch=1024, transformer denoiser, 100k×3 stages). Picks GPU via `CUDA_VISIBLE_DEVICES`.

## Status
- DDP training kicked off but not yet verified at steady state (the early-step throughput included CUDA/cuDNN warmup + cold CLIP cache, so it wasn't representative).
- Need to re-run after the dataset optimization and check `nvidia-smi` utilization + `it/s`. If DDP is still slower than single-GPU on the powerful card, the answer is "use single-GPU on cuda:0" since the model is small enough that 2× DDP probably can't beat one fast card on this workload.
