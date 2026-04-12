# 三线并行执行计划

创建于 2026-04-12

## 总览

三条线同时推进，按资源类型分配：

| Track | 内容 | 资源 | 产出 |
|---|---|---|---|
| A | 完善 DART v8 + Flow Matching | GPU | 更好的 motion generator |
| B | V-A 数据集准备（优先 Audio） | CPU + 网络 | 新模态训练数据 |
| C | 论文写作 + 文献调研 | 人 | 论文初稿 |

## Track A：GPU 密集（完善 + FM）

### Phase 1: 完善 v8（2-3 天）

前置工作——不管走不走 FM 都要做：

| 任务 | 时间 | 状态 |
|---|---|---|
| 加 joint limit loss/clamp（消除 223 度超限） | 2h | 待做 |
| retarget 全 AMASS SMPLX_N (18270 npz → 预计 15k 过滤后) | 14h 计算 | 待做 |
| 过滤 tpose / apose / transition to stand（减 13% 垃圾数据） | 30min | 待做 |
| 试 diffusion_steps=5 推理（零成本） | 5min | 待做 |
| 重新提取 mp_data_g1_69 + 重训 VAE v4 | 3h | 待做 |
| 重训 denoiser v8 (全 AMASS + joint limit) | 3h | 待做 |
| 渲染 v8 rollout 对比 v7 | 10min | 待做 |

### Phase 2: Flow Matching（1-2 周）

| 任务 | 时间 | 说明 |
|---|---|---|
| 读论文: CFM (Lipman 2023), MotionFlow, FlowMDM | 2天 | 理解 FM 在 motion 上的应用 |
| 实现 flow_matching.py 模块 | 1天 | 替换 diffusion/gaussian_diffusion.py |
| 修改 train_g1_mld.py 用 FM loss | 0.5天 | velocity prediction 替代 x0 prediction |
| 修改 render_g1_rollout_69.py 用 FM sampling | 0.5天 | ODE solver / 1-step Euler |
| 训练 FM 版本 + 多步 ablation (K=1,2,5,10) | 2天 | 对比 DDPM |
| 渲染 FM vs DDPM 对比视频 | 0.5天 | 定性评估 |
| 量化 FID / R@K / MM-Dist 指标 | 1天 | 定量评估 |

FM 核心代码改动量估算：~200 行新 + ~50 行改。VAE / 69-dim feature / dataset 全部不动。

### Phase 3: Transition Demo（2 天）

| 任务 | 时间 | 说明 |
|---|---|---|
| 给 render 脚本加 prompt_schedule 模式 | 1h | 按 step 切换 text_embedding |
| 渲染 multi-prompt transition 视频 | 0.5h | stand → walk → wave → kick → stand |
| 对比 69-dim vs 360-dim transition smoothness | 1天 | PJ / AUJ 指标 |
| 对比 FM vs DDPM transition smoothness | 1天 | PJ / AUJ 指标 |

## Track B：CPU 密集（V-A 数据准备）

在 GPU 训练时同步做，不抢资源。

### Phase 1: Audio 数据（1-2 周）

优先做 AIST++（最干净、最可行）：

| 任务 | 时间 | 说明 |
|---|---|---|
| 下载 AIST++ (~3GB) | 1h | https://google.github.io/aistplusplus/ |
| SMPL → G1 retarget (GMR pipeline) | 4h 计算 | 复用现有 GMR 脚本 |
| 提取 audio feature (EnCodec / Jukebox) | 2h | CPU |
| 对齐 audio-motion 时间戳 | 1h | AIST++ 自带 metadata |
| 写 audio-conditioned dataset class | 4h | 继承 G1PrimitiveSequenceDataset |
| 写 audio encoder 模块 | 4h | MLP / Transformer 映射到 512-dim |
| 初步训练 audio-conditioned denoiser | 5h GPU | 在 AIST++ G1 数据上 |

### Phase 2: Vision 数据（可选，Week 3-4）

| 方案 | 可行性 | 说明 |
|---|---|---|
| A. 从 motion 渲染 MuJoCo 视角作为 vision input (self-supervised) | 最可行 | V 和 motion 天然对齐 |
| B. BABEL text → VLM 生成 image → image-motion pairs | 中 | 不够真实 |
| C. 实验室采集 | 最难 | 需要设备+标注 |

建议先走 A，有了 FM + audio 之后再考虑真实 vision。

## Track C：论文写作

| 周 | 任务 |
|---|---|
| Week 1 | 读 FM + V-A 相关论文 10 篇，写 related work outline |
| Week 2 | 写 method section (69-dim feature + FM + transition) |
| Week 3 | 跑完所有 ablation，写 experiments section |
| Week 4 | introduction + conclusion + 内部 review |

## 里程碑

| 日期 | 里程碑 | 验收标准 |
|---|---|---|
| Week 1 末 | v8 训完 (全 AMASS + joint limit) | walk > 1 m/s, 无关节超限 |
| Week 2 末 | FM v1 训完 + 1-step 推理可用 | 推理 < 5ms，FID 不劣于 DDPM |
| Week 2 末 | AIST++ retarget 完成 | 1000+ G1 dance clips |
| Week 3 末 | FM vs DDPM 全 ablation 完成 | 表格数据齐全 |
| Week 3 末 | audio-conditioned v1 可 demo | 放音乐 → G1 跳舞 |
| Week 4 末 | 论文初稿 | 可内部 review |
