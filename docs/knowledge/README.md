# Knowledge Base

项目相关知识点的结构化归档。和 `notes/` 不同：
- `notes/` = **规划性文档**（paper plan, architecture design, 任务拆解）
- `knowledge/` = **参考性知识**（数据集结构、算法原理、踩坑记录、对比表）

## 怎么用

### 最直接：grep + VS Code search

```bash
# 找带 "vad" tag 的卡片
grep -rl "tags:.*vad" knowledge/

# 搜关键词
grep -rn "quaternion" knowledge/

# VS Code: Cmd/Ctrl+Shift+F 在 knowledge/ 内搜
```

### 按主题目录浏览

```
datasets/          数据集详解 (BONES, BABEL, ...)
representations/   特征表示 (69-d, quat 约定, G1 拓扑)
methods/           方法与算法 (FM, VAD aug, kinematic regressor)
architecture/      项目架构 (9-module, 4 层, pipeline 目录)
external_tools/    外部工具 (GMR, SOMA, Kimodo, HandoverSim)
experiments/       实验记录 (v7 recipe, v12 否决)
```

### 快速索引：[INDEX.md](INDEX.md)

## 卡片格式

每个 `.md` 都有 YAML frontmatter：

```markdown
---
title: 卡片标题
tags: [tag1, tag2]            # 用于 grep 分组
related: [another_card.md]    # 交叉引用
last_updated: 2026-04-23
status: stable | draft | stale
---

# 标题

## TL;DR
(1-2 句话精髓)

## 内容...
```

## 添加新卡片

1. 选对应主题目录
2. 复制任意一个卡片当模板
3. 更新 frontmatter 的 `tags` 和 `related`
4. 如果重要，在 `INDEX.md` 里加一行

## 未来升级路线

- [ ] `search.py` — 简单 BM25 检索
- [ ] `embed.py` — sentence-transformers 语义搜索
- [ ] MCP tool 封装 → Claude agent 直接 query
