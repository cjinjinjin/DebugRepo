import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "大模型做 SFT 或 RAG，真正难的从来不是训练按钮，而是你怎么证明它真的变好了。",
  "MMLU、ARC 这些通用榜单分再高，一到法律、医疗、设备维修这种垂域场景，参考价值都会急剧下降。",
  "所以垂域评估必须直面三个工程目标：模型选型、SFT 微调验证、RAG 检索与幻觉甩锅。",
  "没有评估闭环就没有稳定迭代，测试集、自动化评估和人工盲测必须连成一条证据链。"
];
