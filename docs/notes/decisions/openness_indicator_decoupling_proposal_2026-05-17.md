*Date: 2026-05-17 · Owner: Lingfan · Type: PROPOSAL · Status: deferred*

## body_openness 指标解耦提案

### 问题

`_body_openness_5pt_yz_distsum` 用 5 个 keypoint(L_wrist + R_wrist + L_elbow + R_elbow + Chest)的 YZ 平面 pairwise 距离之和。

但 **wrist 位置同时被 opt 1 (amplitude amplifier) 放大**。所以即使 opt 3 (openness primitive) 只调 shoulder_roll,opt 1 一启动 → wrist y 一变 → body_openness 跟着变 → V[2] 与 V[0] 永远耦合,无法在指标层证明 axis 独立。

### 提案 A(用户拍板,但当前暂不动)

改为 **3-keypoint 版**:{L_elbow, R_elbow, Chest},3 对距离:
- L_elbow ↔ R_elbow(肘部 lateral spread)
- L_elbow ↔ Chest(左侧 shoulder roll)
- R_elbow ↔ Chest(右侧 shoulder roll)

**优**:
- 与 wrist 解耦 — opt 1 改 wrist 不再污染 V[2]
- 与 D[0] reach_extension 仍然解耦(reach 用 X-forward 通道,openness 用 YZ)
- 几何上"上身侧展度"由 elbow 位置就够了,wrist 是 reach 该管的

**劣**:
- raw signal 量纲下降(10 对→3 对、且少最远的 wrist 对)
- tanh scale 需重新拟合 — P3 normalizer 顺手处理

### 当前决策

**deferred** — 先把 opt 3 primitive 在现有 5pt 指标下做出来,visual 验证 OK 后再回头改指标 + 重算 V。理由:
- opt 3 设计本身比指标改重要,先跑通主线
- 改指标会让现存 144 batch + 75 LHS 的 V 值失效,需重算
- 现有耦合不影响 opt 3 是否能用,只影响"axis 独立性"的论证强度

### 实施触发条件

opt 3 跑通后,如果 V[0] / V[2] 耦合分析显示 r > 0.5(强相关),则启动此提案 → 改指标 → 重算 V → 再次验证解耦。

### 实施步骤(预留)

1. 改 `src/data_pipeline/vad/regressor_3x3.py:_body_openness_5pt_yz_distsum` → 改名 `_3pt_` + 改 keypoint set + 改 docstring
2. 改 call site `compute_va_torch`(line 520)
3. 写 `scripts/recompute_v_only.py` — 读现有 NPZ + 重算 V/A/D,~30 秒
4. 更新 `src/data_pipeline/vad/regressor_3x3.py` 头部 v1.5 → v1.6 note
5. P3 normalizer 重新拟合
