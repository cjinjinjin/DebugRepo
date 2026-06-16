# Video Outline

> **主题**：`midnight-press`（默认首选：技术测评 / AI 工具深度解读）
> **总时长**：约 3 分 48 秒（口播约 910 字）
> **章节数**：5 章 / 20 步

---

## 1. coldopen-strength — GPT-Image-2 为什么是断层领先（4 steps · ~44s）

**信息池**：
- 发布节点：2026-04-21 发布 GPT-Image-2（ChatGPT 内名 Images 2.0）—— 来源 article §开场
- 排名数据：Arena.AI 1512 分，领先第二名 242 分，官方称“前所未见的悬殊优势”—— 来源 article §开场
- 三个能力支柱：文字渲染 / 指令遵循 / 编辑能力 —— 来源 article §一
- 编辑场景：保留 Logo、局部替换、风格统一、参考图延展 —— 来源 article §一

**开发计划**：
- step 1 (~11s) — 大标题 + 榜单断层数字（1512 / +242）作为开场冲击
- step 2 (~11s) — 文字渲染前后对照（乱码风险 vs 稳定中文渲染）
- step 3 (~11s) — 指令遵循“brief 面板”逐项点亮（位置/背景/版式/禁改项）
- step 4 (~11s) — 编辑能力流程图（输入图→局部替换→统一风格→成图）

口播节选：
> GPT-Image-2 不只是“会画图”，而是把可用性推前了一大截。它在 Arena.AI 的断层领先背后，核心是文字渲染、指令遵循和编辑能力三件事同时变强。

---

## 2. access-routes — 在哪用最顺手（4 steps · ~45s）

**信息池**：
- 官方入口：ChatGPT 付费订阅可直接使用 —— 来源 article §二 官方渠道
- Codex 集成：研发环境里直接生成 UI/图标/贴图 —— 来源 article §二 官方渠道
- Lovart ChatCanvas：可在协作画布串联多模型 —— 来源 article §二 三方平台
- API 生态：OpenAI API、OpenRouter、302.AI 三条接入路径 —— 来源 article §二 API 调用

**开发计划**：
- step 1 (~11s) — 渠道总览地图（官方 / 平台 / API 三列）
- step 2 (~11s) — ChatGPT 与 Codex 双入口对照卡
- step 3 (~11s) — Lovart 协作画布示意（多模型串联节点）
- step 4 (~12s) — API 路由流程（OpenAI / OpenRouter / 302.AI）

口播节选：
> ChatGPT 是最快入口，Codex 把研发和视觉放到同一链路；如果做产品集成，就走官方 API 或路由平台。

---

## 3. case-playbook — 玩法案例库怎么用（4 steps · ~46s）

**信息池**：
- 案例站定位：不是纯图库，支持 prompt / 模板 / 字段 / 复现指令查看 —— 来源 article §三 案例网站
- 典型方向：UI 样机、海报品牌、信息图、学术配图、漫画角色、技术架构、头像贴纸 —— 来源 article §三 典型案例
- 技术限制：技术图为 PNG 位图，非可编辑 SVG —— 来源 article §三 技术架构图说明
- 分类规模：案例覆盖 18 个分类、数百张示例 —— 来源 article §三 与 §五

**开发计划**：
- step 1 (~11s) — 案例站四项信息面板（prompt/模板/可改字段/复现句）
- step 2 (~11s) — “UI+海报+信息图”三类样式卡逐个高亮
- step 3 (~12s) — “学术图+漫画角色+技术图”三类能力递进展示
- step 4 (~12s) — PNG 限制提示 + 适用场景边界（文档配图/演示/PPT）

口播节选：
> 案例库的价值不在“看”，而在“复现”。你能直接拿 prompt 和模板去改；同时也要知道边界，比如技术图是位图，不是源文件。

---

## 4. skill-system — 开源 Skill 的方法论（4 steps · ~46s）

**信息池**：
- 核心判断：效果差异主要来自 prompt 工程化程度 —— 来源 article §四 开场
- 工作流：识别模式→分类→选模板→填参数→渲染 prompt→调用工具 —— 来源 article §四 Skill 流程
- 模板规模：18 大类、79 模板，模板文件内含参数表与变体 —— 来源 article §四 模板规模
- 三种模式：Mode A / B / C 对应不同宿主能力 —— 来源 article §四 三种运行模式

**开发计划**：
- step 1 (~11s) — “随手写 prompt vs 工程化流程”左右对照
- step 2 (~11s) — Skill 六步流水线动画（节点依次点亮）
- step 3 (~12s) — 18 类 / 79 模板数据看板
- step 4 (~12s) — Mode A/B/C 三卡片逐个进入并标注适用环境

口播节选：
> 同模型不同效果，本质常常是流程能力差异。Skill 把这件事固化成可复用流水线，稳定输出高质量提示词或成图。

---

## 5. rollout-next — 立刻上手的执行路径（4 steps · ~47s）

**信息池**：
- 场景一 Codex：宿主可直接生图，Skill 负责模板与 prompt —— 来源 article §四 场景一
- 场景二 Claude Code / Cursor：配 ENABLE_GARDEN_IMAGEGEN、OPENAI_BASE_URL、OPENAI_API_KEY 走 Mode A —— 来源 article §四 场景二
- 场景三 ChatGPT/Lovart：无 API 也可拿结构化 prompt 手动执行 —— 来源 article §四 场景三
- 行动建议：先翻案例站，再把 garden-skills 接入 Agent 环境 —— 来源 article §五 结语

**开发计划**：
- step 1 (~11s) — 三场景矩阵（Codex / 自配 API Agent / ChatGPT-Lovart）
- step 2 (~12s) — 环境变量终端面板逐项出现（A 模式启动条件）
- step 3 (~12s) — “一句话需求→结构化 prompt→成图”闭环示意
- step 4 (~12s) — 收束行动清单（先案例站、再仓库接入、持续迭代）

口播节选：
> 真正能复用的不是某张图，而是流程。先找到最接近的案例，再把 Skill 接进你的 Agent 工作流。

---

## 素材清单

### 1. coldopen-strength
- ✓ WeChat 导出图 1（presentation/public/assets/wechat-01.jpg）
- ✓ WeChat 导出图 2（presentation/public/assets/wechat-02.png）
- ⚠️ Arena 排行榜截图（暂无，使用数据卡片 placeholder）

### 2. access-routes
- ✓ WeChat 导出图 3（presentation/public/assets/wechat-03.webp）
- ⚠️ Codex / Lovart 实际操作截图（暂无，使用流程图 placeholder）

### 3. case-playbook
- ✓ WeChat 导出图 4（presentation/public/assets/wechat-04.webp）
- ⚠️ 各分类示例原图集（暂无，使用分类卡 placeholder）

### 4. skill-system
- ✓ 文本化流程信息（来自 article）
- ⚠️ 仓库图示素材（暂无，使用代码风可视化）

### 5. rollout-next
- ✓ 环境变量与流程信息（来自 article）
- ⚠️ 真实终端截图（暂无，使用终端样式 placeholder）
