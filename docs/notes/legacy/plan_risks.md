# Risk Register (NMI 13-week plan)

**Last updated**: 2026-04-23

每个 risk 有：**probability × impact = severity**。定期 review + 更新状态。

## 🔴 High severity (积极监控)

### R1 · IRB 4/26 未批 → 用户研究堵塞
- **Probability**: medium (30%)
- **Impact**: critical (没 user study 就没 C4 contribution)
- **Mitigation**:
  - 本周五提交完，周一催伦理委员会状态
  - Fallback A: 先用 sim-only data 做 "perceived VAD by regressor"，把 user study 作为补充而非核心
  - Fallback B: 找外部 crowdsourcing (Prolific/MTurk) 做 video-based rating，不需要现场 IRB
- **Owner**: me + co-author
- **Trigger**: 5/3 仍无批复 → 切 Fallback A

### R2 · M7 handover 训练数据不够
- **Probability**: high (70%, BONES 真正 handover 只 <100 条)
- **Impact**: high (C2 立不住)
- **Mitigation**:
  - Week 3 下载 HandoverSim (~10k 双人递物)
  - Week 5 自录 200-300 条 G1 handover (hand-teleop)
  - Fallback: scope reduce，只做 "receive" 单向（给 robot 递物），不做 bidirectional
- **Owner**: me
- **Trigger**: Week 4 末 HandoverSim 没 retarget 通

### R3 · 真机 G1 不稳定（OmniH2O policy 不够鲁棒）
- **Probability**: medium (40%, motion tracker 是未知数)
- **Impact**: high (没 real demo = NMI 立意丢一半)
- **Mitigation**:
  - Week 5 提早开始 G1 SDK 联调
  - Backup: 租用 Unitree 官方 demo G1 做录制（付费但可得）
  - Fallback: 只做 sim demo + anonymized sim→real video mixing
- **Owner**: me
- **Trigger**: Week 7 末 real G1 没跑通任何 demo

## 🟡 Medium severity (定期 review)

### R4 · VAD regressor 精度不够 (r < 0.5 on ABEE)
- **Probability**: medium (40%)
- **Impact**: medium (可以换 M1B 的 VAD 源)
- **Mitigation**:
  - 9-feature set + closed-loop augmentation 已设计好
  - Fallback: 纯用 BONES style_prior 做 categorical VAD (丢弃连续性)
- **Owner**: me
- **Trigger**: Week 3 末 ABEE eval r < 0.4 for any dim

### R5 · bones_fm_v1 M1A baseline 不够好 (pass rate < 4/8)
- **Probability**: low (25%, 已有 v7 同 recipe 经验)
- **Impact**: medium (要 debug 训练 schedule)
- **Status**: 🟡 测试中 (auto_eval running 4/23)
- **Mitigation**:
  - 如果 < 4/8: 扩训 stage 到 500k 看趋势
  - 如果 sign_flip 是主因: Savitzky-Golay 预滤波数据
- **Owner**: me
- **Trigger**: 今天 auto_eval 结果

### R6 · 13 周时间不够，deadline miss
- **Probability**: medium (50%)
- **Impact**: medium (fallback 10/15 已在 milestones 里)
- **Mitigation**:
  - Milestones 文件里每周有 exit criteria
  - Week 5 + Week 8 + Week 10 三次正式 check-in
  - 切换 fallback 的明确条件
- **Owner**: me

## 🟢 Low severity (log 一下即可)

### R7 · 模型训练 GPU 被占用
- **Probability**: medium (室友/同事偶尔用)
- **Impact**: low (延迟几小时)
- **Mitigation**: tmux + 训练脚本快速 resume，Blackwell 96GB 训练也快

### R8 · BONES license 问题
- **Probability**: low (已 ack license)
- **Impact**: low (最多重新填表)
- **Mitigation**: 保留所有 email 通信

### R9 · Paper 超长
- **Probability**: low (NMI 4500 word main + supp)
- **Impact**: low (砍方法描述到 supp)
- **Mitigation**: 先写 supp 再写 main，main 只留故事主线

## Risk review cadence

- **每周日 weekly retro 时**更新 R1-R3（critical path）
- **每阶段 milestone exit 时**重新评估所有 R
- **出现新 risk 时**立刻 append 到文件底部

## 已关闭的 risk

### R0 (closed) · BONES 下载失败/gated access
- **Date closed**: 2026-04-23
- **Outcome**: 142k 全量下载完成，单位转换 + parser 跑通
