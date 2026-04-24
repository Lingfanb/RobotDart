# Long-Term Goals (6 months – 3 years)

**Last updated**: 2026-04-23

## Horizon 1 · NMI 提交后 (6-12 个月)

### 主线 · VADBridge paper sequence

```
NMI 主文 (2026-07-19 submit)
   ↓ accept 或 reject-with-feedback
T-RO Extended 版 (rolling, 按 reviewer 意见补 experiment)
   ↓
ICLR / CVPR / Robotics: Science and Systems 续作
   (不同社区露出，找工业界合作可能)
```

### 副产物 · 可独立发表

| Paper | 潜在 venue | 何时 |
|---|---|---|
| VAD-conditioned motion augmentation (anchor+ΔVAD) | CVPR workshop | 2026 秋 |
| 开源 G1 social handover benchmark | NeurIPS D&B | 2026 冬 |
| Affect-aware LLM agent architecture (M-Brain) | HRI | 2027 春 |

## Horizon 2 · 博士阶段总体方向 (1-3 年)

### 研究主线

> **在真实人机交互场景中，让具身智能 (embodied AI) 具备情感感知与情感表达能力**

三条支线：
1. **Affective motion generation** — 当前 NMI 的延伸，向更细粒度 (finger/face) + 更长时程扩展
2. **Multi-modal affective perception** — 目前 M2/M8 还是 off-the-shelf 拼装，长期要做端到端 (audio+video+pose → VAD) pretrained model
3. **Social coordination beyond handover** — handover 是最小单元，长期扩到 collaborative manipulation, service robotics, caregiving

### 里程碑 (按学位时间线)

```
Y1 (2026)  ✅ G1 adapt + FM baseline + NMI submit
Y2 (2027)  真实部署 + user study N≥100 + 另一篇 top-tier
Y3 (2028)  系统化 benchmark + 工业合作 + 毕业论文框架
```

### 不做什么 (Focus discipline)

- ❌ 不追 LLM 本身能力的改进（用 Claude/GPT API 即可）
- ❌ 不做无 embodied context 的 motion generation（HumanML3D 那类纯 benchmark）
- ❌ 不做 pure face / voice 情感识别（只用作 perception input）
- ❌ 不做 vision-language alignment（太拥挤的方向）

## Horizon 3 · 长期愿景 (3-5 年)

形成一个可复现的 **"LLM agent + affective perception/action"** 架构标准，被社区采纳为 baseline。

影响目标：
- 代码 + 数据 + 模型公开，至少 1k GitHub star
- 至少 3 个工业界实验室 (e.g., Unitree, Booster, Figure) 用或对比
- 毕业后继续做 embodied affective AI，学界或业界 R&D

## 风险 / 触发调整条件

如果 NMI 被 reject 且 T-RO 也无法 salvage → **Horizon 2 重新权衡**：
- 方向 A：继续 affective HRI 方向但换场景（caregiving robot for elderly）
- 方向 B：转向 manipulation 主线（放下 affect 这块）
- 方向 C：转向 LLM agent for robotics 方向（放弃 VAD 具体化）

每年 Q1 review 一次这个文件。
