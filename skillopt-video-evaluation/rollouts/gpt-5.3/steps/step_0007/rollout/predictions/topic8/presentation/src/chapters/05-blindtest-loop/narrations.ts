import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "当两个模型分数咬得很紧，就要进入人工盲测：隐藏模型名，只看回答质量做 Side-by-Side 选择。",
  "盲测界面通常采用左右对照与流式输出，标注者只需判断左好、右好或平局。",
  "如果业务有特殊口径，还可以自定义评估提示词，覆盖题目生成、作答与教师评分全链路。",
  "到这一步，你就拿到了一个可持续的闭环：测试集构建、自动化评估、人工验证、Prompt 回写优化。"
];
