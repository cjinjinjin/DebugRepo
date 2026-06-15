import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "The labeling pipeline itself has three stages. Stage 1: understand the query — extract intent, entity, category. Stage 2: evaluate each decoration's relation to the query. Stage 3: make the final Bad or NonBad call.",
  "We went through multiple generations of this pipeline. DV3 in 2023 — 13 iterations just on the US market. The starting accuracy? 53%. Basically a coin flip. The biggest breakthrough was one prompt change. We told the model: evaluate query-decoration relevance. Not ad-query relevance. Not decoration-ad relevance. Just query to decoration. That jumped accuracy from 53% to nearly 70%.",
  "After cleaning up training data, the finalized DV3 pipeline hit 85.64% overall accuracy in the US. Multi-market results looked solid too — Germany at 81.50%, France at 84.79%. But Japanese was tough at 61.96%, and Portuguese only hit 63.27%.",
  "Then GPT-4 Turbo arrived in early 2024. We tried applying the same DV3 prompts directly. Result? 42% accuracy. Worse than random guessing.",
  "The problem was format. We had to switch to ChatML format with proper start and end tokens. Adjust max token limits — 200 for decoration, 100 for crossing. Tune frequency and presence penalties. 18 experiments later, we recovered to 86.90%.",
  "The flight verification was interesting. DV3 vs GPT-4 Turbo showed some nuance. DCO decorations got significantly better — 39% lower defect rate. But HSL regressed by 17.77%. Trade-offs everywhere, all measured with p-values under 0.05.",
  "Mid-2024, GPT-4o came in. 25 experiments across 7 markets. The key discovery: make the model explain its reasoning before it scores. That explanation before judgment approach pushed accuracy from 73% to about 84%. Short explanations worked better than long ones — experiment #6 at 84.53% beat experiment #5 at 83.30%. Concise reasoning, then score.",
  "The cost trajectory tells the whole story. DV3: $0.184 per item. GPT-4 Turbo: $0.019. GPT-4o: $0.0072. That's a 25x reduction in 18 months.",
];
