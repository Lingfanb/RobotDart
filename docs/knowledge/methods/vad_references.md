---
title: VAD Indicators — Literature Support Audit
tags: [vad, reference, citation, audit]
related: [vad_indicators_definition.md, affect_features.yaml]
last_updated: 2026-04-24
status: stable
---

# VAD 9-Indicator · 文献支持度审计

> 每个指标的**方向性支持**（文献说这个方向对不对）和**具体公式支持**（文献是否用过这个 form）分别评分。
> Paper 写作参考。

## 总表

| 指标 | 方向 | 公式 | 备注 |
|---|---|---|---|
| V1 · smoothness $\phi$ | ✅✅✅ | 🟡 我的 form | $J/(\bar s+\epsilon)$ 相对化是 operational choice |
| V2 · body_contraction $\kappa$ | ✅✅✅ | ✅ Camurri 原创 | 最硬的一个 |
| V3 · spine_uprightness $u$ | ✅✅ | 🟡 我的 form | 非对称惩罚是 operational |
| A1 · mean_speed $\bar s$ | ✅✅✅ | ✅ 标准 | universally used |
| A2 · jerk $J$ | ✅✅✅ | ✅ 标准 | multiple papers use |
| A3 · accel_peak $a_{\max}$ | ✅✅ | 🟡 max 是选择 | mean 更常见 |
| D1 · reach_extension $r$ | ⚠️ 弱 | ❌ 无直接 | **paper 时需说明为 project-specific framing** |
| D2 · forward_approach $v_{\text{fwd}}$ | ✅ | 🟡 我的 form | 概念有支持 |
| D3 · directness $\delta$ | ✅✅✅ | ✅ Laban tradition | 稳 |

---

## 按指标详细 · 可引文献

### V1 · Relative Smoothness $\phi$

**方向支持（high）**：
- **Pollick et al. 2001** "Perceiving affect from arm movement" *Cognition* 82:B51-B61
- **Flash & Hogan 1985** "The coordination of arm movements" *J. Neurosci.* — minimum-jerk principle
- Mancini et al. 2012 — normalized jerk as clinical motor-quality metric
- Karg 2013 §4.4 — "valence related to smoothness"
- Laban Effort · Flow (Free/Bound)

**公式 $J / (\bar s + \epsilon)$**：未见 prior art，project-specific form to prevent static-pose bias.

### V2 · Body Contraction $\kappa$

**方向 + 公式都硬**：
- **Camurri, Lagerlöf, Volpe 2003** "Recognizing emotion from dance movement" *Int. J. HCI* 59:213-225 — ⭐ 首创 "contraction index"
- Glowinski et al. 2011 "Toward a minimal representation of affective gestures" *IEEE Affective Computing*
- Kleinsmith & Bianchi-Berthouze 2013 综述 §4.2

### V3 · Spine Uprightness $u$

**方向支持（high）**：
- **Boone & Cunningham 2001** "Children's expression of emotional meaning in music through expressive body movement" *Dev. Psychology* 37:21-41 — ⭐ "duration of leaning forward" 识别为 sadness cue
- Wallbott 1998 "Bodily expression of emotion" *Eur. J. Soc. Psych.* 28:879-896 — trunk posture
- Coulson 2004 "Attributing emotion to static body postures" *J. Nonverbal Behavior* 28:117-139
- Gross et al. 2012 "Effort-Shape and kinematic assessment of bodily expression of emotion during gait"

**公式（非对称惩罚 $\max(0, -\sin(\text{pitch}))$）**：operational choice；文献里通常对前倾-后仰对称建模，我们强调只惩罚前倾更干净。

### A1 · Mean Speed $\bar s$

**方向 + 公式都硬**：
- Camurri 2003 — Quantity of Motion
- **Pollick et al. 2001** — arm speed ↔ affect
- Wallbott 1998, Atkinson et al. 2004 "Emotion perception from dynamic and static body expressions" *Perception* 33:717-746
- Paterson et al. 2001 "The carrier of emotion" — gait speed
- Karg 2013 §7.3 "speed is the most commonly selected feature"

### A2 · Jerk $J$

**方向 + 公式都硬**：
- **Karg et al. 2009** "A Two-Fold PCA-Approach for Inter-Individual Recognition of Emotions in Natural Walking" *MVA* — 直接用 jerk
- Crenn et al. 2016 "Body expression recognition from animated 3D skeleton" *SPIE*
- Wallbott & Scherer 1986

### A3 · Acceleration Peak $a_{\max}$

**方向支持（medium）**：
- de Meijer 1989 "The contribution of general features of body movement to the attribution of emotions" *J. Nonverbal Behavior* 13:247-268
- Truong & Weber 2006 "Gesture recognition using motion capture and peak acceleration"
- Laban Effort · Weight (Light/Strong)
- Delsarte · Force
- Wallbott 1998 power/energy

**Max vs Mean of 2nd diff**：operational choice，捕捉 peak events 比 mean 更能区分"平稳 vs 猛烈"。

### D1 · Reach Extension $r$ ⚠️

**这是最弱的一个。老实讲。**

标准 Karg 综述里 reach → D 不是强 link。更常见映射：
- reach → manipulation (neutral)
- reach → V (Boone & Cunningham "arms away from torso" → happiness)

**间接支持**：
- **Tracy & Robins 2004** "Show your pride: Evidence for a discrete emotion expression" *Psych Science* 15:194-197 — pride display includes "arms outstretched" → pride 有 dominance 成分
- **Witkower & Tracy 2019** "Bodily communication of emotion: Evidence for extra-facial behavioral expressions and available coding systems" *Emotion Review* 11:184-193 — systematic review
- Dael, Mortillaro, Scherer 2012 BAP — encodes arm extensions
- de Meijer 1989 — dominance kinematic features
- ⚠️ Carney, Cuddy, Yap 2010 "Power posing" — expansive postures + power（replication controversy，引用慎）

**Paper 建议写法**：

> "We define reach extension as a D-cue specifically in the handover context,
> where extending hands toward a partner signals an active dominant role (the
> agent 'does' rather than 'receives'). While this specific mapping is not
> directly established in the motion-affect literature, it aligns with the
> broader power/approach framework in social psychology
> (Burgoon et al. 1995, Mehrabian 1972)."

### D2 · Forward Approach $v_{\text{fwd}}$

**方向支持（medium-high）**：
- **Mehrabian 1972** "Nonverbal Communication" — immediacy behaviors (approach) as power signal
- **Burgoon, Buller, Woodall 1995** *Nonverbal Communication: The Unspoken Dialogue*
- **Hall 1966** *The Hidden Dimension* — proxemics (about distance, not speed, but foundational)
- Andersen 2008 *Nonverbal Communication* — approach/avoidance
- Gross et al. 2012 — gait in emotional walk

**具体 form（character-frame forward velocity）**：operational choice。保留符号（不取 abs）反映"退缩 = 负 D"。

### D3 · Directness $\delta$

**方向 + 公式都硬**：
- **Laban & Lawrence 1947** *Effort: Economy of Human Movement* — Space (Direct/Indirect)
- **Chi, Costa, Zhao, Badler 2000** "The EMOTE model for effort and shape" *SIGGRAPH*
- **Larboulette & Gibet 2015** "A review of computable expressive descriptors of human motion" *MOCO*
- Bishko 2007, Zhao & Badler 2005
- Camurri EyesWeb — 同样用 path_net / path_total 公式

---

## Paper Bibliography · 第一梯队（必进）

```
V 维度 (3 核心):
  Camurri, A., Lagerlöf, I., & Volpe, G. (2003). Recognizing emotion from
    dance movement: comparison of spectator recognition and automated
    techniques. Int. J. Human-Computer Studies, 59(1-2), 213-225.
  Boone, R. T., & Cunningham, J. G. (2001). Children's expression of emotional
    meaning in music through expressive body movement. Dev. Psychology, 37, 21-41.
  Pollick, F. E., Paterson, H. M., Bruderlin, A., & Sanford, A. J. (2001).
    Perceiving affect from arm movement. Cognition, 82(2), B51-B61.

A 维度 (3 核心):
  Karg, M., Kühnlenz, K., & Buss, M. (2009). A Two-Fold PCA-Approach for
    Inter-Individual Recognition of Emotions in Natural Walking. MVA.
  Wallbott, H. G. (1998). Bodily expression of emotion. Eur. J. Soc. Psych., 28, 879-896.
  Laban, R., & Lawrence, F. C. (1947). Effort: Economy of Human Movement.

D 维度 (3 核心):
  Mehrabian, A. (1972). Nonverbal Communication. Aldine-Atherton.
  Tracy, J. L., & Robins, R. W. (2004). Show your pride: Evidence for a discrete
    emotion expression. Psych Science, 15(3), 194-197.
  Chi, D. M., Costa, M., Zhao, L., & Badler, N. I. (2000). The EMOTE model for
    effort and shape. SIGGRAPH.

综述 (必进):
  Karg, M., Samadani, A. A., Gorbet, R., Kühnlenz, K., Hoey, J., & Kulić, D.
    (2013). Body Movements for Affective Expression: A Survey of Automatic
    Recognition and Generation. IEEE TAC, 4(4), 341-359.
  Kleinsmith, A., & Bianchi-Berthouze, N. (2013). Affective Body Expression
    Perception and Recognition: A Survey. IEEE TAC.
```

## Paper Bibliography · 第二梯队（支持/辅助）

```
V 辅助:
  Coulson 2004 "Attributing emotion to static body postures"
  Flash & Hogan 1985 "The coordination of arm movements"
  Glowinski et al. 2011 "Toward a minimal representation of affective gestures"

A 辅助:
  Pollick et al. 2001 (V, A 都用)
  Atkinson et al. 2004 "Emotion perception from dynamic and static body expressions"
  Paterson et al. 2001 "The carrier of emotion"
  de Meijer 1989 "The contribution of general features of body movement..."
  Truong & Weber 2006 (peak acc)
  Crenn et al. 2016 (3D skeleton)

D 辅助:
  Witkower & Tracy 2019 "Bodily communication of emotion"
  Hall 1966 "The Hidden Dimension" (proxemics)
  Burgoon, Buller, Woodall 1995 "Nonverbal Communication"
  Andersen 2008 "Nonverbal Communication"
  Larboulette & Gibet 2015 (computable Laban)
  Zhao & Badler 2005 (motion qualities)
```

---

## Paper 写作建议（3 处需要说明 operational choice）

1. **V1** `φ = 1 - J/(s̄+ε)`：相对 jerk-to-speed ratio 是我们为 fix "静止姿势误判为 smooth" 而引入的 operational form。
2. **V3** 非对称前倾惩罚（$\max(0, -\sin)$ 不 $|\sin|$）：基于心理学观察，前倾是负价信号，后仰不是对称的正价信号。
3. **A3** `max` 而非 `mean` of 2nd diff：为捕捉 peak events（impact, snap）而非 average force。

## 对 paper reviewer 的"最可能挑刺点"

**D1 reach_extension**：最容易被 reviewer 问"你凭什么说 reach = D？"

**防御**：
1. 在 Methods 里明确 scope："for handover context"
2. 引 Tracy & Robins pride display + Burgoon et al. power display
3. 同时提供 ablation：去掉 D1 后 handover task performance 下降多少
