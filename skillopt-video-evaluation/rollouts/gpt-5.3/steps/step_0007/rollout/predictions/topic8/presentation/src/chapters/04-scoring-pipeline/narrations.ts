import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "评分环节分两路：客观题直接规则比对，主观题交给教师模型按 rubric 打分并解释理由。",
  "这里的关键不是跑一次，而是固定教师模型和评分配置，保证多轮迭代可横向比较。",
  "工程上推荐先用 20 题小样本跑通，再扩到大规模任务，避免大批量低质量评估浪费成本。",
  "最终要看的是分维度 dashboard：准确率、幻觉率和主观质量一起看，才能定位真正瓶颈。"
];
