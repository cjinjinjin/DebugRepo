# GPT-Image-2 完全指南！附大量玩法案例，顺便开源我的生图 Skill ～

- 作者：ConardLi
- 发布时间（推测）：2026-04-28 08:30:00
- 来源文件：C:\Users\jinjinchen\Downloads\GPT-Image-2 完全指南！附大量玩法案例，顺便开源我的生图 Skill ～.html

---

2026 年 4 月 21 日，OpenAI 发布 GPT-Image-2，在 ChatGPT 中被称为 Images 2.0。
在 Arena.AI 的 Text-to-Image 排行榜上，GPT-Image-2 以 1512 分登顶，比第二名谷歌 Nano-Banana-2 高出 242 分。Arena.AI 官方评价：从未有任何模型能以如此悬殊优势排名第一。
作者作为 Nano-Banana-2 的长期用户，给出结论：GPT-Image-2 是当前最强图像生成模型，多数场景效果碾压上一代。

## 一、GPT-Image-2 强在哪

1) 文字渲染更稳：
过去 AI 生图常见问题是图中文字乱码，尤其多语言场景更容易翻车。GPT-Image-2 把图中文字当核心能力，适合海报、封面、菜单、招牌、PPT 风格图、UI 标签和信息图。

2) 指令遵循更强：
可写非常具体的需求：主体位置、背景类型、文案排版、杂志或电商风格、不可改元素等。虽然做不到 Figma 那种像素级控制，但更接近“按 brief 出图”。

3) 图像编辑能力更实用：
支持图像输入与图像编辑，并高保真处理输入图片。适合产品换背景、局部替换、风格统一、Logo/包装保留、参考图延展。

## 二、GPT-Image-2 在哪里可用

### 官方渠道
- ChatGPT（Plus / Pro / Business 等付费订阅）：https://chatgpt.com/
- OpenAI Codex 环境已集成图像能力，可在写代码时生成 UI、游戏贴图、应用图标等资产：https://openai.com/zh-Hans-CN/codex/

### 三方平台
- Lovart（ChatCanvas 视觉协作画布，支持模型串联）：https://www.lovart.ai/zh/home

### API 调用
- OpenAI Image API（model: "gpt-image-2"，支持 images.generate / images.edit）：https://developers.openai.com/api/docs/guides/image-generation?api=image
- OpenRouter（模型路由，统一 API、自动负载、多模型切换，调用名 openai/gpt-5.4-image-2）：https://openrouter.ai/openai/gpt-5.4-image-2
- 302.AI（国内按量付费、无需订阅）：https://302.ai/product/detail/gpt-image-2

## 三、有哪些有意思的玩法

作者建立了案例网站，收录大量结构化模板与实测案例：
- 网站：https://gpt-image2.mmh1.top/

每张图可查看：完整 prompt、对应模板、可改字段、如何一句话复现。
支持瀑布流与分类浏览。

代表性方向包括：
1) UI 界面样机（直播电商、社交动态、短视频封面、聊天界面）
2) 海报与品牌视觉（主海报、Campaign KV、Web Banner、杂志封面）
3) 信息图与数据可视化（bento、教程图、KPI 面板）
4) 学术配图（pipeline figure、架构图、机理图、Graphical Abstract）
5) 漫画与角色（四格、分镜、角色设定、关系图）
6) 技术架构图（系统架构、流程图、时序图、ER、状态机、拓扑）
7) 头像与贴纸（风格头像、角色网格、3D 拟物图标、历史人物系列）

说明：技术图多为 PNG 位图，不是可编辑 SVG，更适合文档配图与快速表达，不替代 draw.io / Excalidraw。

## 四、最佳实践：作者开源的 GPT-Image-2 Skill

核心观点：效果差距主要来自 prompt 工程化程度。

Skill 的流程：
1) 判断运行模式（是否有 API Key、宿主是否具备图像工具）
2) 分析需求所属视觉类型
3) 匹配结构化模板
4) 填充模板参数
5) 渲染高质量 prompt
6) 调图像工具生成，或输出可直接粘贴的 prompt

开源仓库：
- https://github.com/ConardLi/garden-skills/
- 安装说明：https://github.com/ConardLi/garden-skills/blob/main/README.zh-CN.md

规模：覆盖 18 大类、79 个结构化模板，每个模板为 Markdown，定义 JSON/结构化自然语言模板、参数表、变体与案例。

三种运行模式：
- Mode A（Garden 本地模式）：有 API Key，完整自动化“模板→prompt→脚本→出图落盘”
- Mode B（Host-Native）：在 Codex 等宿主内，Skill 负责模板与 prompt，宿主执行生图
- Mode C（Advisor 顾问模式）：无图像工具且无 API Key 时，输出高质量 prompt 供手动使用

常见场景：
- Codex：直接走 Mode B
- Claude Code / Cursor：配置 ENABLE_GARDEN_IMAGEGEN、OPENAI_BASE_URL、OPENAI_API_KEY 后走 Mode A
- ChatGPT Web / Lovart：可把 Skill 当 prompt 工程手册，先在 Agent 中生成结构化 prompt，再粘贴使用

## 五、模板体系与结语

模板体系覆盖 18 类方向，包括学术配图、头像人设、品牌包装、图像编辑、网格拼贴、信息图、地图、人物肖像、海报活动、产品视觉、场景插画、演示文档、叙事序列、技术架构图、字体排版、UI 样机、编辑工作流等。

作者建议两件事：
1) 去案例站点直接复制 prompt 实测
2) 在 Codex / Claude Code / Cursor 配置 garden-skills，实现“说一句话就能出图”

仓库与模板会持续更新，欢迎 Star、提 Issue、参与贡献。
