"""Auto-evaluate a trained FM checkpoint and emit a JSON decision.

Workflow:
1. Render 8 standard prompts at K=1 (optionally also K=5 for Pareto)
2. Read data.npz per prompt, compute numerical metrics
3. Apply decision rules → output JSON

Decision categories:
  - "success"    : metrics passed, stop training
  - "retrain"    : adjust specified params and train more
  - "halt"       : serious problem (collapse), need human review

Usage:
    python -m scripts.auto_eval \
        --ckpt ./mld_denoiser/g1_fm_reflow_v1based/checkpoint_100000.pt \
        --render_dir ./mld_denoiser/g1_fm_reflow_v1based/auto_eval_100k_k1 \
        --inference_steps 1

Output:
    JSON to stdout + written to <render_dir>/decision.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


STANDARD_PROMPTS = (
    "stand", "walk forward", "run", "kick",
    "wave right hand", "punch", "jump", "turn left",
)


@dataclass
class Args:
    ckpt: str
    render_dir: str = ""
    inference_steps: int = 1
    prompts: tuple[str, ...] = STANDARD_PROMPTS
    render_script: str = "mld.render_g1_rollout_fm"
    """Which render module to run. Use 'mld.render_g1_rollout_fm_latent' for latent FM."""
    skip_render: bool = False
    """If True, assume render_dir already has data.npz files."""
    cuda: int = 1
    """Which GPU to render on."""


# ── Metric thresholds ───────────────────────────────────────────────────────
# Tune these to change what "success" means.

THRESHOLDS = {
    # Per-prompt thresholds
    'sign_flip_rate_max': 0.30,     # target < 30%
    'max_vel_max':        0.80,     # target < 0.80 rad/frame
    'joint_abs_max':      2.70,     # < 2.7 rad ≈ 155°
    'min_root_z':         0.50,     # robot should not sink below 0.5m
    'max_root_z':         1.20,     # robot should not levitate above 1.2m

    # Global thresholds
    'min_prompts_passing': 6,       # at least 6/8 prompts must pass
    'collapse_max_prompt_diff_std': 0.02,  # stdev of diff across prompts — below = collapse
    'halt_sign_flip_rate': 0.55,    # > 55% → severe jitter / collapse
}


def compute_prompt_metrics(npz_path: Path) -> dict:
    """Per-prompt numerical metrics from rendered rollout npz."""
    d = np.load(npz_path)
    dof = d['dof_pos']              # (T, 29) joint angles in radians
    world = d['world_pos']          # (T, 3)  root xyz
    contact = d['foot_contact']     # (T, 2)

    dq = np.diff(dof, axis=0)       # (T-1, 29)
    max_vel = float(np.abs(dq).max())
    mean_vel = float(np.abs(dq).mean())

    # Sign-flip rate (most direct jitter indicator)
    if dq.shape[0] >= 2:
        sign_flip = float(((np.sign(dq[1:]) * np.sign(dq[:-1])) < 0).mean())
    else:
        sign_flip = 0.0

    max_joint_abs = float(np.abs(dof).max())
    max_joint_idx = int(np.abs(dof).max(axis=0).argmax())

    root_z_min = float(world[:, 2].min())
    root_z_max = float(world[:, 2].max())
    xy_drift = float(np.linalg.norm(world[-1, :2] - world[0, :2]))
    contact_pct = float((contact > 0.5).mean())

    return {
        'max_vel':       max_vel,
        'mean_vel':      mean_vel,
        'sign_flip_rate': sign_flip,
        'joint_abs_max': max_joint_abs,
        'joint_idx':     max_joint_idx,
        'root_z_min':    root_z_min,
        'root_z_max':    root_z_max,
        'xy_drift':      xy_drift,
        'foot_contact_pct': contact_pct,
        'mean_dof_last_frame': float(dof[-1].mean()),  # for collapse detection
    }


def check_prompt_passes(m: dict) -> tuple[bool, list[str]]:
    """Return (passes, list_of_failed_criteria)."""
    fails = []
    if m['sign_flip_rate'] > THRESHOLDS['sign_flip_rate_max']:
        fails.append(f"sign_flip={m['sign_flip_rate']:.3f}>0.30")
    if m['max_vel'] > THRESHOLDS['max_vel_max']:
        fails.append(f"max_vel={m['max_vel']:.2f}>0.80")
    if m['joint_abs_max'] > THRESHOLDS['joint_abs_max']:
        fails.append(f"joint_abs={m['joint_abs_max']:.2f}>2.70")
    if m['root_z_min'] < THRESHOLDS['min_root_z']:
        fails.append(f"root_z_min={m['root_z_min']:.2f}<0.50")
    if m['root_z_max'] > THRESHOLDS['max_root_z']:
        fails.append(f"root_z_max={m['root_z_max']:.2f}>1.20")
    return (len(fails) == 0, fails)


def detect_collapse(all_metrics: dict[str, dict]) -> bool:
    """Mode collapse: all prompts produce near-identical final DoF values."""
    finals = np.array([m['mean_dof_last_frame'] for m in all_metrics.values()])
    std = float(finals.std())
    return std < THRESHOLDS['collapse_max_prompt_diff_std']


def render_all(args: Args) -> Path:
    """Run the render script. Returns the output directory path."""
    render_dir = args.render_dir
    if not render_dir:
        ckpt_dir = Path(args.ckpt).parent
        render_dir = str(ckpt_dir / f"auto_eval_k{args.inference_steps}")
    render_path = Path(render_dir)
    render_path.mkdir(parents=True, exist_ok=True)

    if args.skip_render:
        return render_path

    cmd = [
        "/home/lingfanb/miniforge3/envs/DART/bin/python",
        "-m", args.render_script,
        "--denoiser_checkpoint", args.ckpt,
        "--inference_steps", str(args.inference_steps),
        "--output_dir", str(render_path),
        "--prompts", *args.prompts,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    env["MUJOCO_GL"] = "egl"
    print(f"[auto_eval] Rendering → {render_path}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[auto_eval] Render failed:", file=sys.stderr)
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("render failed")
    return render_path


def decide(all_metrics: dict[str, dict], is_collapsed: bool) -> dict:
    """Apply decision rules.

    Returns dict with keys:
      action: 'success' | 'retrain' | 'halt'
      adjust: dict of suggested param changes (for 'retrain')
      reason: human-readable
    """
    if is_collapsed:
        return {
            'action': 'halt',
            'adjust': {},
            'reason': 'Mode collapse detected — all prompts converge to identical output.',
        }

    sign_flips = np.array([m['sign_flip_rate'] for m in all_metrics.values()])
    mean_sign_flip = float(sign_flips.mean())
    if mean_sign_flip > THRESHOLDS['halt_sign_flip_rate']:
        return {
            'action': 'halt',
            'adjust': {},
            'reason': f'Severe jitter: avg sign_flip={mean_sign_flip:.3f} > 0.55.',
        }

    # Count passing prompts
    per_prompt_pass = {p: check_prompt_passes(m) for p, m in all_metrics.items()}
    num_pass = sum(1 for (ok, _) in per_prompt_pass.values() if ok)

    if num_pass >= THRESHOLDS['min_prompts_passing']:
        return {
            'action': 'success',
            'adjust': {},
            'reason': f'{num_pass}/{len(all_metrics)} prompts passing.',
            'per_prompt_pass': {p: ok for p, (ok, _) in per_prompt_pass.items()},
        }

    # Retrain — figure out what to adjust based on dominant failure mode
    dominant_fails = {'jitter': 0, 'joints': 0, 'heights': 0, 'vel': 0}
    for p, (ok, reasons) in per_prompt_pass.items():
        if ok:
            continue
        for r in reasons:
            if r.startswith('sign_flip'):
                dominant_fails['jitter'] += 1
            elif r.startswith('joint_abs'):
                dominant_fails['joints'] += 1
            elif r.startswith('root_z'):
                dominant_fails['heights'] += 1
            elif r.startswith('max_vel'):
                dominant_fails['vel'] += 1

    adjust = {}
    if dominant_fails['jitter'] >= 2 or dominant_fails['vel'] >= 2:
        adjust['weight_acc_match_gt'] = 1.0        # from 0.5 → 1.0
        adjust['weight_vel_match_gt'] = 1.5        # from 1.0 → 1.5
    if dominant_fails['joints'] >= 2:
        adjust['weight_joint_limit'] = 0.10        # from 0.05 → 0.10
    if dominant_fails['heights'] >= 2:
        adjust['note'] = 'root_z off — may need fresh retrain (not resumable).'
    if not adjust:
        adjust['weight_acc_match_gt'] = 0.75       # mild default

    return {
        'action': 'retrain',
        'adjust': adjust,
        'reason': f'Only {num_pass}/{len(all_metrics)} prompts passing. '
                  f'Dominant fails: {dominant_fails}.',
        'per_prompt_pass': {p: ok for p, (ok, _) in per_prompt_pass.items()},
        'per_prompt_fails': {p: reasons for p, (_, reasons) in per_prompt_pass.items()},
    }


def main():
    args = tyro.cli(Args)

    render_path = render_all(args)

    # Collect per-prompt metrics
    all_metrics = {}
    for prompt in args.prompts:
        safe = prompt.replace(' ', '_').replace('/', '_')[:50]
        npz = render_path / safe / "data.npz"
        if not npz.exists():
            print(f"[auto_eval] WARNING: missing {npz}", file=sys.stderr)
            continue
        all_metrics[prompt] = compute_prompt_metrics(npz)

    is_collapsed = detect_collapse(all_metrics)
    decision = decide(all_metrics, is_collapsed)

    report = {
        'ckpt': args.ckpt,
        'render_dir': str(render_path),
        'inference_steps': args.inference_steps,
        'is_collapsed': is_collapsed,
        'metrics': all_metrics,
        'decision': decision,
        'thresholds': THRESHOLDS,
    }

    out_json = render_path / "decision.json"
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    # Pretty console summary
    print("\n" + "=" * 72)
    print(f"CHECKPOINT: {args.ckpt}")
    print(f"K = {args.inference_steps}")
    print(f"COLLAPSED: {is_collapsed}")
    print("=" * 72)
    print(f"{'prompt':<20} {'sign_flip':>10} {'max_vel':>9} {'joint°':>8} {'pass':>6}")
    for p, m in all_metrics.items():
        ok, fails = check_prompt_passes(m)
        deg = np.degrees(m['joint_abs_max'])
        mark = '✓' if ok else '✗'
        print(f"  {p:<18} {m['sign_flip_rate']:>10.3f} {m['max_vel']:>9.2f} "
              f"{deg:>7.0f}° {mark:>6}")
    print("=" * 72)
    print(f"DECISION: {decision['action']}")
    print(f"REASON:   {decision['reason']}")
    if decision.get('adjust'):
        print(f"ADJUST:   {decision['adjust']}")
    print("=" * 72)
    print(f"JSON saved: {out_json}")

    # Exit code encodes action for shell scripts
    code = {'success': 0, 'retrain': 2, 'halt': 3}.get(decision['action'], 1)
    sys.exit(code)


if __name__ == "__main__":
    main()
