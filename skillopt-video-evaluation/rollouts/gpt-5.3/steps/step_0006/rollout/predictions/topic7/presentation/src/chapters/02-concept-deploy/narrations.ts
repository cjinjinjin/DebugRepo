import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "OpenClaw 的能力链路可以拆成五层：Inference、Memory、RAG、MCP、Skills。",
  "推理服务本身无状态，所以必须用记忆层做短期与长期的上下文管理。",
  "部署时先跑通主链路：环境检查、安装初始化、基础联调，不要一开始就过度优化。",
  "参数层重点看 temperature、top_p、penalty、max_tokens，并给主模型配置 fallback。",
];
