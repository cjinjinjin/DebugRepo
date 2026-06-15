import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "UberMarket migration was the other big project. We unified US and INTL onto the same infrastructure. Finished UMV1 on PaidSearch in North America by July 2024.",
  "The challenge was latency. Enabling relevance models for all decorations across all traffic groups would blow up CPU usage.",
  "The fix: ConditionalCall. The Queene model only serves Bing O&O traffic groups — MarketplaceClassifications 1 and 2, RelatedToAccountId 1004.",
  "Everyone else gets a default zero vector. NA runs on Queene69, INTL on Queene13.",
  "CPU increase? Under 1%. No extra machines.",
];
