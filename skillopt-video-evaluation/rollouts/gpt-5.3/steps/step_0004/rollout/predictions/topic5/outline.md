# Video Outline

> **主题**：`midnight-press`（自动采用首个推荐主题）—— 深色社论感，适合 AI 工具拆解与方法论讲解  
> **总时长**：约 3 分 40 秒  
> **章节数**：5 章 / 20 步

---

## 1. coldopen — Claude Design 到底改了什么（4 steps · ~44s）

**信息池**：
- 发布时间与市场反应：2026-04-17 发布，当天 Figma 股价大跌 —— 来源 article §1 / L10-L13
- 产品形态：左聊天右画布，可对话、行内评论、直接编辑、滑杆修改 —— 来源 article §1 / L16-L21
- 核心定位：更像 Claude Code，不是传统画布工具 —— 来源 article §1 / L24-L29
- 三个能力差异：可运行代码、读代码库、主动提问与自检 —— 来源 article §1 / L44-L55

**开发计划**：
- step 1 (~11s) — 标题冲击屏 + “发布日 / 股价波动”对照条
- step 2 (~10s) — 传统工具 vs Claude Design 的角色翻转双栏
- step 3 (~11s) — “可运行代码”主视觉 + link/tab/diff 三标签点亮
- step 4 (~12s) — 20+ 轮到 2 轮的效率对比条形图

口播节选：
> 它不是 AI 版 Figma，而是把 AI 放到主生成位，人退到主审阅位。

---

## 2. prompt-core — 420 行提示词的骨架（4 steps · ~43s）

**信息池**：
- 系统提示词规模：约 420 行，高信息密度 —— 来源 article §2 / L76-L77
- 角色句：AI 设计师 + 用户经理 —— 来源 article §2.1 / L80-L90
- 动态身份：animator / UX / slide designer / prototyper —— 来源 article §2.1 / L94-L97
- 方法启发：角色应动态切换，不应固定为“前端开发者” —— 来源 article §2.1 / L98-L104

**开发计划**：
- step 1 (~10s) — “420 lines”巨型数字 + 结构分层网格
- step 2 (~11s) — 角色翻转示意：Assistant→Designer，User→Manager
- step 3 (~11s) — 多身份轮盘逐个高亮四种专业角色
- step 4 (~11s) — “静态角色 vs 动态角色”对照卡

口播节选：
> 关键不是让 AI 听话，而是让 AI 在正确身份里做判断。

---

## 3. operating-rules — 真正可复用的执行规则（4 steps · ~46s）

**信息池**：
- 先问后做的条件判断与示例 —— 来源 article §2.2 / L126-L137
- 提问策略：至少 10 问，多选 + 自由输入 —— 来源 article §2.2 / L142-L143
- 极简总结原则：只写 caveat 与 next step —— 来源 article §2.2 / L146-L150
- 内容原则：One thousand no's for every yes —— 来源 article §2.5 / L220-L246

**开发计划**：
- step 1 (~11s) — “信息充足→执行 / 信息不足→提问”决策树
- step 2 (~11s) — 三个输入案例卡片逐个翻转显示 ask/no-ask
- step 3 (~12s) — 极简总结示例：长复述被划掉，只保留两行
- step 4 (~12s) — “1000 NO / 1 YES”配额盘 + 留白对比

口播节选：
> 不要把勤奋误当专业，问不问问题也应该是可计算的决策。

---

## 4. anti-ai-design — 去 AI 味与视觉系统（4 steps · ~45s）

**信息池**：
- 黑名单：渐变背景、emoji、彩色边框圆角卡片、假数据等 —— 来源 article §2.3 / L156-L179
- 字体策略：避免常见模板字体，推荐替代字体族 —— 来源 article §2.3 / L180-L183
- 配色策略：品牌色优先，不够用时 oklch 派生 —— 来源 article §2.4 / L186-L195
- oklch 优势：感知均匀，调色更稳定 —— 来源 article §2.4 / L196-L216

**开发计划**：
- step 1 (~11s) — AI 味黑名单墙逐条亮起并打叉
- step 2 (~11s) — 字体替换演示：常见字体淡出，替代字体升格
- step 3 (~11s) — 品牌色→oklch 派生流程带箭头动画
- step 4 (~12s) — HSL 与 OKLCH 的感知一致性对照柱

口播节选：
> 去 AI 味不是“反 AI”，而是拒绝默认模板对审美的劫持。

---

## 5. skill-rollout — 从产品洞察到可执行 Skill（4 steps · ~42s）

**信息池**：
- Claude Design 局限：访问门槛、无 API、封闭性 —— 来源 article §3.1 / L306-L313
- 核心判断：竞争力来自 Prompt Engineering + 模型能力 —— 来源 article §3.1 / L314-L322
- Skill 结构：角色定义、六步流程、反 AI 味、占位符哲学 —— 来源 article §3.2 / L332-L438
- 两处增强：设计系统宣告 + v0 半成品早展示 —— 来源 article §3.2 / L400-L418

**开发计划**：
- step 1 (~10s) — 封闭产品痛点三卡片并列
- step 2 (~11s) — “方法抽取”管道：Prompt → Skill → 多工具复用
- step 3 (~10s) — 增强点 A/B：先宣告系统、再给 v0
- step 4 (~11s) — 结尾宣言 + 下一步清单（录屏可直接收尾）

口播节选：
> 真正可迁移的不是某个模型，而是可复用、可验证、可协作的方法。

---

## 素材清单

### 1. coldopen
- ✓ 本地文章配图 1（presentation/public/assets/wechat-01.jpg）
- ✓ 本地文章配图 2（presentation/public/assets/wechat-02.jpg）
- ⚠️ 若需高分辨率封面图，需后补原图链接

### 2. prompt-core
- ✓ 可用文本与图形演示，无硬依赖外部素材
- ⚠️ 如需作者头像 / 原推文截图，需人工补充

### 3. operating-rules
- ✓ 可用程序化图表与流程图
- ⚠️ 如需官方文档截图，需后补版权可用版本

### 4. anti-ai-design
- ✓ 可用程序化字体与配色对比
- ⚠️ 若要加入字体 specimen 图，需补素材

### 5. skill-rollout
- ✓ 可用本地文章配图 3（presentation/public/assets/wechat-03.jpg）
- ⚠️ 如需 Skill 仓库真实界面截图，需用户提供
