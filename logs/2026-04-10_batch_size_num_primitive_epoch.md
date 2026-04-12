# Batch size / num_primitive / epoch — 关系详解

> Reference doc — 训练超参之间的关系，专门解释 G1 denoiser 训练里 `batch_size`、`num_primitive`、`epoch`、`step` 怎么相互影响。
> 写于 2026-04-10，对应 G1-DART 数据集（66,496 个 train primitive）和 04-08 之后的训练配置。

## 1. Primitive 是什么

一个 **primitive** = 一段 10 帧的动作 = `history(2 帧) + future(8 帧)`，30fps 下 ≈ 0.33 秒。

**训练数据集** = 把所有 motion clip 切成 primitive 后，得到 **66,496 个 primitive**（train split）。

```
Sequence A (4 秒):  [P0 P1 P2 P3 P4 P5 P6 P7 P8 P9 ...] ← 切成多个 primitive
Sequence B (3 秒):  [P0 P1 P2 P3 P4 P5 P6 ...]
...
∑ = 66,496 个 primitive
```

每个 primitive 就是模型的"一个训练样本"。Primitive 之间的切片 stride = `future_length = 8`，所以同一 sequence 内相邻两个 primitive 重叠 2 帧（即前一个的最后 2 帧 = 后一个的 history）。

## 2. Batch size 是什么

**batch_size = 一次反向传播之前喂给模型多少个 primitive**。这是所有深度学习的标准定义。

如果 `batch_size=128`，那一个 forward+backward 里：
1. 随机抽 128 个 primitive
2. 一起 forward → 算 loss
3. backward + optimizer.step

## 3. Epoch 是什么

**1 个 epoch = 把整个数据集完整过一遍**。

```
steps_per_epoch = dataset_size / batch_size
```

| 配置 | dataset | batch | steps/epoch |
|---|---|---|---|
| `batch=128` | 66,496 | 128 | 519 |
| `batch=4096` | 66,496 | 4096 | 16 |

如果训练总共跑 300,000 step：

| 配置 | total_steps | steps/epoch | 跑了多少 epoch |
|---|---|---|---|
| `batch=128` | 300k | 519 | **~578 epoch** |
| `batch=1024` | 300k | 65 | **~4,615 epoch** |
| `batch=4096` | 300k | 16 | **~18,750 epoch** |

> 注意：04-08 work log 把"18,479 epochs"判断为过拟合是**错的**。原版 DART denoiser 实际用的就是 `batch=1024, num_primitive=4`，等效 ~18,479 epoch（数学上跟 `batch=4096, num_primitive=1` 完全相同）。我们之前 v2/v3/v4 飘的真正原因是 **`num_primitive=1`** 导致没训练自回归 rollout 能力，而不是 batch 太大。

## 4. num_primitive=4 是什么

这里就稍微特殊了。原本的标准训练：
> 一个 step = 抽 batch 个样本 → 一次 forward+backward

但 DART 想训练**自回归 rollout**（让模型能连续生成长序列），所以引入了 `num_primitive`：
> 一个 step = 抽 batch 个**起点**，每个起点取 **num_primitive** 个**连续的** primitive，**串联**地依次过 forward+backward

```
batch_size=128, num_primitive=4 时一个 outer step 的内部：

抽 128 个 sequence 的起点：
  Seq_a 起点 P5  → P5,  P6,  P7,  P8       ← 4 个连续 primitive
  Seq_b 起点 P12 → P12, P13, P14, P15
  Seq_c 起点 P0  → P0,  P1,  P2,  P3
  ...（共 128 行）

然后 train loop 内部:
  for primitive_idx in [0, 1, 2, 3]:           ← num_primitive 这层
      取 128 个样本（这一时间步的）
      forward + backward + optimizer.step      ← 一次 gradient update
```

所以一个 outer step 实际上是 **4 次** forward+backward，每次 128 个样本 → 总共 **512 个 primitive forward**。

为什么需要 `num_primitive>1`？因为 stage 2+ 的训练用 rollout history：上一个 primitive 的预测结果会被喂回来当下一个的 history。模型必须学会"接住"自己的输出，而不仅仅是基于 GT history 生成。这是消除 train/test gap 的关键（standard exposure bias 问题）。

## 5. 把数字串起来

| 配置 | 一步处理 | 300k step 总样本 | 等效 epoch | 说明 |
|---|---|---|---|---|
| **原版 DART denoiser** `batch=1024, np=4` | 4096 | **1.23B** | **~18,479** | README 里实测命令 |
| 我们 v2/v3/v4（错） `batch=4096, np=1` | 4096 | 1.23B | ~18,479 | 总样本数跟原版相同，但缺自回归训练 |
| 04-08 误判方案 `batch=128, np=4` | 512 | 154M | ~2,316 | 训练样本量不足，会欠拟合 |
| **v5 我们用** `batch=1024, np=4`，**80k×3** | 4096 | 983M | **~14,785** | 跟原版基本对齐，省 20% 时间 |

注意一个反直觉的事实：v2/v3/v4 的 `batch=4096, np=1` 跟原版 `batch=1024, np=4` **总样本数完全相同**（都是 1.23B）。区别只在 `num_primitive`：
- `np=1`: 没自回归训练 → rollout 时误差爆炸 → 飘
- `np=4`: 有自回归训练 → rollout 稳定

所以"飘"的根本原因是 `num_primitive`，不是 `batch_size`。

## 6. 概念对照表

| 概念 | 含义 | G1-DART v5 值 |
|---|---|---|
| primitive | 1 个训练样本（10 帧动作 ≈ 0.33s） | — |
| dataset 总样本数 | 训练集里有多少 primitive | 66,496 |
| `batch_size` | 一次 gradient update 喂多少 primitive（在一个时间点） | **1024** |
| `num_primitive` | 一个 outer step 里串联多少个**连续的**时间点 | **4** |
| outer step | 一次 batch 处理（包含 num_primitive 次 forward+backward） | — |
| `total_steps` | 训练总共跑多少 outer step | **240,000** (80k × 3 stage) |
| 等效 epoch | 完整过一遍数据集的等效次数 | **≈ 14,785** |

## 7. 一个直白的类比

想象数据集是一本 **66,496 页的书**：

- **`batch_size=1024`**：每次拿一摞 1024 页出来读
- **`num_primitive=4`**：每读一摞 1024 页，要读"这一摞的下一摞"再下一摞...连续 4 摞才算 1 个 outer step（因为模型要学连贯性，不能只看孤立的页）
- **epoch**：把整本书从头到尾读完一遍
- **240k step ≈ 14785 epoch**：相当于把这本书通读了 1.4 万多遍

DART 这种 diffusion 模型需要训得很多遍，因为每个样本在不同的 noise level 上要被学很多次。1 万 epoch 听起来吓人但对 diffusion 是正常的。

## 8. DDP 注意事项（2026-04-09 加的）

[mld/train_g1_mld.py](../mld/train_g1_mld.py) 现在支持 DDP，`batch_size` 在 DDP 模式下是**全局** batch（per-rank = global / world_size）。所以：

- **单卡** `--train_args.batch_size 1024` → 一次 1024 个 sequence
- **DDP 2 卡** `--train_args.batch_size 1024` → 每张卡 512 个 sequence，**全局还是 1024**

要保持等效，DDP 不需要把 batch_size 翻倍。512/卡 比 128/卡 大很多，GPU 也能更好饱和，DDP 加速比应该比小 batch 时好。但 v5 还是推荐先单卡跑（避免 DDP 引入新变量），跑完再考虑 DDP 加速。

## 9. 相关代码位置

- [data_loaders/humanml/data/dataset_g1.py](../data_loaders/humanml/data/dataset_g1.py) `get_batch()` —— `batch_size` 和 `num_primitive` 的实际采样逻辑
- [mld/train_g1_mld.py](../mld/train_g1_mld.py) `train()` —— 外层 `while step <= total_steps` + 内层 `for primitive_idx in range(num_primitive)`
- [data_scripts/process_motion_primitive_g1.py](../data_scripts/process_motion_primitive_g1.py) —— primitive 的切片逻辑（`t += FUTURE_LENGTH`）
