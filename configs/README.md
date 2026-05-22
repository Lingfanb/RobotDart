# configs/

按主题组织,不按 agent 消费者(因 act_classes / VAD anchor 等多 agent 共享)。

## 结构

| dir | 用途 |
|---|---|
| `VAD/` | VAD framework 全套:22-class taxonomy + 24-primitive manifest + per-action μ/σ + 20 anchors |
| `ACP/` | ACP framework(Tier 3 决策层)— 待填 |
| `MoGen/` | MoGenAgent 专属:训练数据 whitelist + augmentation preset |
| `Manip/` | ManipAgent 专属 — 待填 |
| `Loco/` | LocoAgent 专属 — 待填 |

## VAD/ 文件清单

| 文件 | 谁读谁 |
|---|---|
| `act_classes.yaml` | `MoGenAgent/data_pipeline/vad/action_taxonomy.py` (22-class 入口) |
| `motion_lib.yaml` | scripts/render_motion_lib_exemplars.py · scripts/scan_all_primitives_babel.py 等 |
| `norm_params_by_action.yaml` | `MoGenAgent/data_pipeline/vad/regressor_3x3.py::load_per_action_norm_params()` |
| `anchors/<primitive>.yaml` | scripts/auto_pick_zero_anchors*.py · scripts/calibrate_from_anchors.py · scripts/verify_anchor_quality.py |

## MoGen/ 文件清单

| 文件 | 谁读谁 |
|---|---|
| `data/amass_whitelist.txt` | SONIC filter pipeline — 通过 filter 的 BABEL/AMASS clips |
| `data/bones_whitelist.txt` | 同上,BONES |
| `aug/preset_*.yaml` | (待重建,5/12 design-only 版本遗失) |
