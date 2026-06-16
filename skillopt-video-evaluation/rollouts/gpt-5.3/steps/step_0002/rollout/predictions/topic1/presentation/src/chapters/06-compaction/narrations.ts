import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "Compaction starts when the context window is near full.",
  "Naive summarization with a separate prompt and tools loses cache immediately.",
  "That divergence starts at token one and cost spikes.",
  "Cache-safe forking reuses the exact parent prefix.",
  "Reserve a compaction buffer for prompt and output tokens.",
  "Outcome: only the new tail is uncached, so cost and latency stay controlled.",
];
