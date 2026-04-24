---
title: v7 FM Baseline (Locked Recipe)
tags: [experiment, fm, baseline, v7, locked]
related: [v12_velocity_snr_rejected.md, ablation_cheatsheet.md, ../methods/flow_matching.md]
last_updated: 2026-04-23
status: draft
---

# v7 FM Baseline — Locked Recipe

## TL;DR

(待填写)

## 为什么 v7 是 M1A baseline

## v7 配置表

| 参数 | 值 |
|---|---|
| parameterization | x0 |
| t sampling | uniform |
| σ_min | 0.001 |
| stage1 steps | 80k |
| stage2 steps | 100k |
| stage3 steps | 100k |
| weight_vel_match_gt | 2.0 |
| weight_acc_match_gt | 1.0 |
| weight_jerk | 0.3 |
| max_rollout_prob | 0.8 |

## Evaluation (4/8 pass)

## 7-Row Ablation（v6-v11 对比表）

## 为什么单变量偏离都更差

## 和 v12 的对比（velocity SNR 实验）

## Related Files
