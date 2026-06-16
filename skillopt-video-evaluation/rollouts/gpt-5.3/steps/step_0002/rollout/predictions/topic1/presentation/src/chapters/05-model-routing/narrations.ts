import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "Prompt caches are model-specific.",
  "Late Opus-to-Haiku switching can increase total cost.",
  "A subagent handoff is safer because context is compressed before routing.",
  "Rule of thumb: switch models early, or hand off with compact context.",
];
