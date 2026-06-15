import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "How do we judge quality? There's a 4-level labeling scale. Excellent means the decoration matches the query directly. Good means it's relevant but not specific. Fair means it's weakly related — like a competitor product. Bad means it shouldn't be there at all.",
  "In production, we collapse it down. Excellent, Good, and Fair all count as NonBad. Only Bad gets filtered out.",
  "Now, where do these labels come from? For six years — 2017 through 2023 — we used managed human labeling. Trained judges scoring every item. It worked, but it was slow. And the cost? $0.238 per item.",
  "In late 2023, we switched to LLM labeling. We started with DV3, moved to GPT-4 Turbo, then GPT-4o.",
  "GPT-4o gets us about 83% accuracy. And the cost dropped to $0.0072 per item — that's roughly 27 times cheaper than human judges.",
  "For context, GPT-4o pricing is $2.50 per million input tokens and $7.50 per million output. Compared to DV3's $0.03 per thousand tokens, that's a massive drop.",
];
