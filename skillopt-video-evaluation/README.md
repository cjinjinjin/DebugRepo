# 跑通 SkillOpt：web-video-presentation 评价体系

## 目标

以 `web-video-presentation` skill 为例，测试 SkillOpt 的完整优化流程：把 skill 文档当成可训练对象，通过生成结果、评价结构、失败反馈和验证集 gate，迭代得到更好的视频生成 skill。

核心不是只优化页面是否能跑，而是优化整个生成闭环：

- `script.md` 是否适合口播
- `outline.md` 是否控制节奏和信息密度
- 章节实现是否像视频而不是 PPT
- 音频和 step 是否健康同步
- review 是否能阻止低质量结果进入最终产物

## SkillOpt 映射

| SkillOpt 概念 | 视频生成任务中的对应物 |
|---|---|
| Skill document | `web-video-presentation` skill 文档 |
| Rollout | 用当前 skill 对一个 article 生成 `script.md`、`outline.md`、`presentation/`、音频和 review-log |
| Trajectory | 生成过程记录、产物文件、构建结果、音频时长、review-log、用户反馈 |
| Evaluator | VideoSkillOpt 评价体系 |
| Reflect | optimizer 从失败轨迹中总结问题，提出 skill edits |
| Update | 对 skill 文档做 add / delete / replace 修改 |
| Validation gate | 在 held-out 文章上比较 candidate skill 是否优于 current skill |
| Best skill | 验证集上表现最好的 `best_skill.md` |

## 推荐数据集结构

每个样本是一篇文章或口播稿，加上生成约束：

```json
{
  "id": "decoration-relevance",
  "article_path": "article.md",
  "target_language": "en",
  "target_duration_min": 7.5,
  "theme": "blueprint",
  "must_have": [
    "click-driven 16:9 web presentation",
    "one narration beat per click",
    "compile without TypeScript errors",
    "no segment over 15 seconds unless justified",
    "outline does not prescribe animation"
  ],
  "quality_preferences": [
    "less dense technical chapters",
    "more visual explanation, less table dumping",
    "keep domain numbers accurate"
  ]
}
```

建议 split：

| Split | 用途 |
|---|---|
| train | 3-5 个文章，用来发现 skill 问题 |
| val | 2-3 个文章，用来决定 edit 是否接受 |
| test | 2-3 个文章，用来最终报告泛化效果 |

当前 `decoration-relevance-video/` 可作为第一个 train trajectory。

初始 topic split 已放在 `topics/`：

| Split | 文件 | Topics |
|---|---|---|
| train | `topics/train/items.json` | `topic1`, `topic3`, `topic5`, `topic6` |
| val | `topics/val/items.json` | `topic2`, `topic4` |
| test | `topics/test/items.json` | `topic7`, `topic8` |

完整 topic registry 见 `topics/all-topics.json`。这些 topic 覆盖 public blog、SharePoint Word、Azure DevOps Wiki、GitHub repo、WeChat article 等不同来源和格式。

Rollout 产物保存规范见 `rollout-data-contract.md`。

首个外部 rollout 已按 GPT-5.3 目标模型版本落盘：

| Run | 位置 | Target rollout model | Planned evaluator |
|---|---|---|---|
| topic3 baseline | `rollouts/gpt-5.3/steps/step_0001/rollout/` | `gpt-5.3-codex` | `gpt-5.5` |

该 rollout 使用现有 `decoration-relevance-video/` 作为已生成结果，按 SkillOpt artifact contract 归档了 `results.jsonl`、`predictions/topic3/conversation.json`、`evaluator-report.json`、`scorecard.json`、`hard_gate.json`、build/typecheck log、script/outline/review/audio segments 等证据。

当前 GPT-5.3 外部 rollout 状态：

| Step | Topic | Status | hard | soft | Fail reason |
|---|---|---|---:|---:|---|
| `step_0001` | `topic3` | full generated baseline | 0 | 0.62 | `pacing_gate_failed: actual duration exceeds target by more than 20%` |
| `step_0002` | `topic1` | full presentation generated | 1 | 0.92 | — |
| `step_0003` | `topic4` | full presentation generated | 1 | 0.93 | — |
| `step_0004` | `topic5` | full presentation generated | 1 | 0.91 | — |
| `step_0005` | `topic6` | full presentation generated | 1 | 0.90 | — |
| `step_0006` | `topic7` | full presentation generated | 1 | 0.92 | — |
| `step_0007` | `topic8` | full presentation generated | 1 | 0.93 | — |
| `step_0008` | `topic2` | full presentation generated | 1 | 0.91 | — |

For `topic1`, `topic2`, `topic4`, `topic5`, `topic6`, `topic7`, and `topic8`, intermediate web-video-presentation checkpoints were auto-accepted with default/first choices. The presentation projects passed narration extraction, typecheck, and production build. Audio synthesis was attempted but blocked by local shell/tooling issues (`jq` for topic1/topic4/topic5/topic6, bash `pipefail` compatibility for topic2/topic7/topic8); this is recorded in each rollout and is not treated as a presentation-generation failure.

## 评价体系总览

生成式视频任务不适合只用一个 accuracy。推荐三层评价：

1. **Hard Gate**：硬门槛，不过直接 reject。
2. **Scorecard**：多维打分，衡量内容、节奏、视觉、工程、音频。
3. **Pairwise Preference**：同一任务下比较 current skill 和 candidate skill 的生成结果。

## A. Hard Gate

这些不是加权平均项，而是阻断项。

| Gate | 判断 |
|---|---|
| 工程可运行 | `tsc --noEmit` / build / dev server 正常 |
| step 一致性 | `narrations.ts` 数量 = 章节 step 数 = audio segments |
| 内容忠实 | 关键数字、结论、专有名词不能错 |
| 无明显 fake | 不编造数据、logo、案例、素材 |
| 视觉底线 | 每章至少 1-2 个 CSS / SVG / Canvas / JS 视觉演示 |
| 非 PPT | 不能整章纯文字、不能大量标题 + bullet |
| 音画同步 | 动画时长不应超过 narration / audio 步长 |
| 主题约束 | 颜色 / 字体家族必须走 theme token |
| 反 AI 味 | 禁紫粉渐变、emoji 当图标、圆角彩边框、同质 fade / blur |
| 目标时长约束 | 若用户明确目标时长，实际时长超出目标 20% 应 fail 或进入人工确认 |

## B. 自动 / 半自动量化指标

| 维度 | 指标 |
|---|---|
| 时长控制 | 总时长偏差、章节时长偏差、>=15s segment 数量、最长 segment |
| 节奏颗粒度 | 每 step 字数、每章 step 数、每章 narration 平均长度 |
| 信息密度 | 每屏核心元素数量，是否超过 1-3 个主信息 |
| 结构平衡 | 是否某章异常膨胀，例如单章超过 90s |
| 视觉多样性 | 每章主导动作是否重复，是否只有 fade / slide |
| 渐进揭示 | 列表是否 1 item = 1 step |
| 双源原则 | 画面是否引用 article 细节，而不是只复述 script |
| 音频健康 | 空段、过短段、过长段、TTS 失败率 |
| 工程卫生 | CSS prefix、跨章 import、hardcoded color / font、storage key |

## C. LLM-as-Judge 质量评分

每个维度建议 1-5 分，并要求 judge 给证据。

| 维度 | 评估问题 |
|---|---|
| Story Arc | 视频有没有清晰起承转合？观众能不能跟住？ |
| Audience Comprehension | 复杂概念是否被视觉化解释，而不是堆术语？ |
| Visual Semantics | 动画是否来自内容关系，而不是无脑入场？ |
| Screen Composition | 留白、字号、层级是否适合 16:9 录屏？ |
| Information Fidelity | 是否保留原文关键事实但没有硬塞？ |
| Pacing | 每个 step 是否像一个口播节拍？ |
| Rewatch Value | 是否有画面记忆点？ |
| Theme Fit | 是否符合选定 theme 的气质？ |
| Anti-AI Taste | 是否避免模板化 AI 网页味？ |
| Production Readiness | 是否能直接录屏发布？ |

## D. Human Preference

最终建议保留人工偏好判断，尤其在样本较少时：

```text
A: current_skill 生成结果
B: candidate_skill 生成结果

1. 你更愿意发布哪个？
2. 哪个更像视频而不是 PPT？
3. 哪个更容易听懂？
4. 哪个更不累？
5. 哪个更符合主题？
```

Human preference 可以作为 validation gate 的最高优先级信号。

## 评分输出格式

```json
{
  "hard_gate": "pass",
  "blocking_issues": [],
  "scores": {
    "content_fidelity": 4.5,
    "pacing": 2.5,
    "visual_semantics": 4.0,
    "non_ppt_feel": 4.0,
    "audio_sync": 3.0,
    "theme_fit": 4.2,
    "engineering": 5.0
  },
  "weighted_score": 3.88,
  "pairwise_preference": "candidate_wins",
  "judge_summary": "Candidate improves visual semantics but still overproduces narration."
}
```

## 针对当前 decoration-relevance-video 的第一条训练信号

当前结果的主要问题不是工程失败，而是评价协议过于偏“是否完成”：

- 目标约 7.5 分钟，实际音频约 11.7 分钟。
- 19 个 segment >=15s。
- `llm-pipeline` 章节约 181.6s，明显过密。
- review-log 记录了 duration issue，但没有把 pacing 当作 blocking issue。

因此第一轮 skill edit 应优先改 review / pacing 协议：

```text
Pacing Gate:
- 目标时长是约束，不是参考。
- 实际时长超过目标 20% 必须 fail 或请求用户确认。
- 单 step audio >=15s 必须拆分或说明。
- 单章超过 90s 必须拆章、压缩或重规划。
- Dense technical chapter 必须遵守 one metric family per step。
- review-log 不能只记录 duration issue 后继续 PASS。
```

## 最小跑通流程

1. 固定一版 `current_skill.md`。
2. 用 train articles 生成视频项目，保存完整 trajectory。
3. 跑 VideoSkillOpt evaluator，输出 scorecard 和失败原因。
4. optimizer 读取 trajectory + scorecard + rejected edits，提出有限 edits。
5. 应用 edits 得到 `candidate_skill.md`。
6. 用 val articles 对 current / candidate 各生成一次。
7. 通过 hard gate 后比较 scorecard + pairwise preference。
8. candidate 更好则接受，否则 reject，并把 rejected edit 作为负反馈。
9. 最终在 test articles 上报告 best skill 的泛化效果。
