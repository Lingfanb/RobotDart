*Date: 2026-05-13 · Owner: Lingfan · Type: LIVE · Status: v1.4*

VAD regressor v1.4 闭式计算参数表。Fused VAD ∈ [−1, +1]³,每维 indicator 权重和 = 1。

## Definitions

| 维 | 心理学含义 | 典型运动线索 |
|---|---|---|
| V (Valence) | 愉悦 vs 悲伤 / 喜欢 vs 厌恶 | 张开 · 大幅度 · 流畅 · upright |
| A (Arousal) | 兴奋 vs 平静 / 激活水平 | 速度高 · 加速度峰 · 急促 · 节奏快 |
| D (Dominance) | 主导 vs 顺从 / 控制感 | 向外伸 · 前倾 · 占空间 · 接近 target |

## Fusion weights

```text
V = 0.40·V1 + 0.35·V2 + 0.25·V3
A = 1.00·A1
D = 0.40·D1 + 0.60·D2
```

## Indicators

| 指标 ID | 名称 | 权重 | 公式 (raw) | (μ, σ) | 物理 channel |
|---|---|---|---|---|---|
| V1 | Motion Amplitude | 0.40 | per EE: median over sliding 15-frame windows of local BBox 对角线;取 4 EE 中 top-2 mean | (0.08, 0.06) TBD | 4 EE sliding-window BBox (pelvis-local) |
| V2 | Root Height | 0.35 | mean_t z_root(t) | (0.65, 0.15) TBD | pelvis 世界 z 坐标 |
| V3 | Body Openness | 0.25 | mean_t Σ_{i<j ∈ 5pts} ‖p_i(t).[y,z] − p_j(t).[y,z]‖, 5pts = {L_wrist, R_wrist, L_elbow, R_elbow, chest} | (4.5, 1.5) TBD | 上身 5 点 frontal-plane (yz) 10-pair 距离和 |
| A1 | Energy / frame | 1.00 | mean_t Σ_j v_j(t)², j ∈ 29 DOFs | (0.010, 0.020) TBD | DOF kinetic-energy proxy |
| D1 | Reach Extension | 0.40 | top-25% mean (sign-aware) of max(0, ½·(x_L_wrist + x_R_wrist)) in pelvis-local | (0.20, 0.15) TBD | wrist forward (sustained peak) |
| D2 | Forward Lean | 0.60 | top-25% mean (sign-aware) of sin(pitch_root(t)) | (0.00, 0.25) TBD | root pitch 旋转 (sustained peak) |

时间聚合 (Time aggregation):
- V1: sliding-window median (window 内 bbox span,filter raise/lower transient)
- V2 / V3 / A1: mean over t (持续状态 / 累积能量,transient 也算)
- D1 / D2: top-25% mean (top quartile of frames, sign-preserving) — 抓 sustained action peak (e.g. bow bottom, handshake contact),过滤 setup/return transient

TBD = 新公式 / 新指标,等 calibrate 后定。其它 (μ, σ) 沿用 v1.4 calibration。

## Internal / legacy (not in fusion)

| 变量 | 公式 | 用途 |
|---|---|---|
| jerk_l1 | peak-centered Gaussian weighted ⟨\|d³q/dt³\|⟩ | v1.3 V1 smoothness 内部用,v1.4 不进 fusion |
| smoothness | 1 − clip(jerk_l1 / (mean_speed + ε), 0, 1) | v1.3 V1,v1.4 保留计算但移出 FUSION_WEIGHTS |

## Constants

| 常量 | 值 | 说明 |
|---|---|---|
| PEAK_CENTERED_SIGMA | 15.0 frames | A1 Gaussian σ,~0.5s @ 30fps,Flash & Hogan 1985 minimum-jerk |
| MOTION_GATE_THRESHOLD | 0.02 rad/frame | smoothness motion gate (legacy 用) |
| EPS | 1e-3 | div-by-zero 防护 |
| UPPER_BODY_DOF_IDX | [12:29] | V1 amplitude 用的 DOF 子集 |

Upper-body DOF 范围 = waist [12:15] + L_arm [15:22] + R_arm [22:29] = **17 DOFs**。排除腿 (0:12) 避免 V 与 A locomotion 通道重叠。

## Normalization

每个 raw indicator 独立 tanh-normalized 到 [−1, +1]:

```text
f_norm = tanh((raw − μ) / σ)
V = Σ w_Vi · f_norm_Vi          (A / D 同理)
```

权重内部和 = 1 + 每输入 ∈ [−1, +1] → fused 输出保证 ∈ [−1, +1]。

## Per-action calibration

YAML: src/data_pipeline/vad/norm_params_by_action.yaml

15 个 action class (BONES 粗粒度 taxonomy):

- standing_idle · sitting · kneeling · crawling · walking · jogging · jumping
- climbing · gesture · dancing · action_dynamic · action_misc · transition
- other · _global

每类有自己的 (μ, σ) 覆盖 global default → 测 "动作内部的风格偏离" 而非 "绝对运动量"。

⚠️ wave_hand 当前 fallback 到 `other` bucket — canonicalize() 不识别 22-class taxonomy 里的 wave_one_arm / wave_two_arms。后续可在 action_taxonomy.\_PATTERNS 加 wave 模式或保持现状。

## Decoupling check

| 物理通道 | V | A | D |
|---|---|---|---|
| DOF angle range (positional) | V1 | | |
| DOF velocity / accel (temporal) | | A1 + A3 | |
| All-link distance (spatial) | V2 | | |
| Shoulder z (postural) | V3 | | |
| Wrist x (manipulation reach) | | | D1 |
| Root pitch (body lean) | | | D2 |

已知耦合点:V1 amplitude vs A1 mean_speed 都涉及 "动作大小",但前者 positional span (max−min),后者 temporal rate (\|dq/dt\|)。实测 wave_hand pool V-A r = **+0.024** (110 clip,v1.3 是 +0.257) → 改善 ✓。

## Version history

- v1.5 (2026-05-13, design + impl lock):
  - **V1** sliding-window median BBox (4 EE,top-2 mean) — 过滤 raise/lower transient
  - **V2** root_height (mean of pelvis world z) — 整体姿态高度
  - **V3** body_openness (5-pt yz pairwise distance sum, mean over t) — 上身 frontal-plane 张开
  - **A1** energy_per_frame (mean of Σ v² over t,replaces v1.4 mean_speed+accel_peak)
  - **D1/D2** 从 mean over t 切到 top-25% mean (sign-aware) — 抓 sustained action peak
- v1.5 V1 design notes: 试过 global bbox / linear+quadratic detrend / sliding max / drop-edges 5 种,仅 sliding median 通过临界对 (R1 A+1.000 raise+small wave vs R15 A+0.911 pure wave) 视觉验证。V-A Spearman ρ = +0.064 (near-orthogonal)。Median 等价于 Motion GPT 思路的 kinematic 轻量版 — 把 raise transient 当 outlier 过滤,保留 sustained wave 振幅。
- v1.4 (2026-05-12): V1 smoothness → motion_amplitude (DOF range)
- v1.3 (2026-05-12): V3 chest_height · A 去 jerk + peak-centered Gaussian · D2 forward_lean (signed) · D 改 2 indicator (reach + lean, Option C 0.40/0.60)
- v1.2 / 1.1 / 1.0 / 0: 见 src/data_pipeline/vad/regressor_3x3.py 头部 docstring

## References

| 指标 | 主要文献 |
|---|---|
| V1 BBox amplitude | Hartmann et al. 2005 (SPC) · Wallbott 1998 (BHIS) · Glowinski et al. 2008 · De Meijer 1989 · Pollick et al. 2001 · Crenn et al. 2017 |
| V2 Root height | Tracy & Robins 2004 · Coulson 2004 |
| V3 Body openness (5-pt yz distsum) | Wallbott 1998 · Tracy & Robins 2004 · Crenn et al. 2017 (distances feature group) |
| A1 Energy / frame | Camurri et al. 2003 (QoM) · Karg et al. 2013 · LaMoGen 2025 · Pollick et al. 2001 |
| D1 Reach extension | Tracy & Robins 2004 · Ekman & Friesen 1972 |
| D2 Forward lean | Burgoon et al. 1995 · Hall 1966 · Mehrabian 1972 |

完整引用 (bibtex-ready):

- Hartmann B., Mancini M., Pelachaud C. (2005). Implementing Expressive Gesture Synthesis for Embodied Conversational Agents. Gesture in Human-Computer Interaction. (SPC = Spatial extent parameter)
- Wallbott H.G. (1998). Bodily expression of emotion. European Journal of Social Psychology 28(6): 879-896.
- Glowinski D., Camurri A., Volpe G., Dael N., Scherer K.R. (2008). Technique for automatic emotion recognition by body gesture analysis. CVPR Workshops.
- De Meijer M. (1989). The contribution of general features of body movement to the attribution of emotions. Journal of Nonverbal Behavior 13: 247-268.
- Pollick F.E., Paterson H.M., Bruderlin A., Sanford A.J. (2001). Perceiving affect from arm movement. Cognition 82(2): B51-B61.
- Crenn A., Khan R.A., Meyer A., Bouakaz S. (2017). Body Motion Analysis for Emotion Recognition in Serious Games. Springer LNCS 9747: 33-42.
- Tracy J.L., Robins R.W. (2004). Show your pride: Evidence for a discrete emotion expression. Psychological Science 15: 194-197.
- Coulson M. (2004). Attributing emotion to static body postures: Recognition accuracy, confusions, and viewpoint dependence. Journal of Nonverbal Behavior 28: 117-139.
- Boone R.T., Cunningham J.G. (2001). Children's expression of emotional meaning in music through expressive body movement. Journal of Nonverbal Behavior 25: 21-41.
- Camurri A., Lagerlöf I., Volpe G. (2003). Recognizing emotion from dance movement: comparison of spectator recognition and automated techniques. International Journal of Human-Computer Studies 59(1-2): 213-225.
- Karg M., Samadani A.A., Gorbet R., Kühnlenz K., Hoey J., Kulic D. (2013). Body movements for affective expression: A survey of automatic recognition and generation. IEEE Trans. Affective Computing 4(4): 341-359.
- Ekman P., Friesen W.V. (1972). Hand movements. Journal of Communication 22(4): 353-374.
- Burgoon J.K., Buller D.B., Woodall W.G. (1995). Nonverbal Communication: The Unspoken Dialogue. McGraw-Hill.
- Hall E.T. (1966). The Hidden Dimension. Doubleday.
- Mehrabian A. (1972). Nonverbal Communication. Aldine-Atherton.
- LaMoGen 2025 (LaBan-effort Motion Generation, exact venue/auth TBD as of 2026-05-13).
