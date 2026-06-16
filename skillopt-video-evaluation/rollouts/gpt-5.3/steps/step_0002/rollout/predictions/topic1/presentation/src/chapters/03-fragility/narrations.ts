import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "The scary part is fragility: harmless-looking changes break cache.",
  "A timestamp in the static prefix can invalidate every request.",
  "Non-deterministic tool ordering also creates misses.",
  "Changing tool parameters or schemas mid-session does the same.",
  "Use guardrails: deterministic ordering, frozen static blocks, explicit change lanes.",
];
