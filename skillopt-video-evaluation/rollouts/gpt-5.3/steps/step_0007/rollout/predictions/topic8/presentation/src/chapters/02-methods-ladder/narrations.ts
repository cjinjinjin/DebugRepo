import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "第一种方法是确定性评估，判断题、单选题、多选题可以用规则直接判分，成本最低但覆盖面有限。",
  "第二种是 BLEU、ROUGE 这类文本相似度，在 LLM 时代会误杀大量意思对但表达不同的好答案。",
  "第三种是 LLM as a Judge，用更强模型做裁判，按明确评分细则评估开放题和简答题。",
  "评估成败不在名词，而在评分锚点是否可执行、可复用、可解释。"
];
