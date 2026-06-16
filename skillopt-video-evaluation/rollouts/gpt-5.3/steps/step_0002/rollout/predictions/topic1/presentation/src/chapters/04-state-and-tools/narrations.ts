import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "When state changes, avoid rewriting the system prompt.",
  "Send updates as messages on the next turn instead.",
  "Tool-set churn mid-session is another cache killer.",
  "Plan Mode should be behavior toggles, not tool catalog swaps.",
  "For huge MCP catalogs, defer-load full schemas behind stable stubs.",
  "Result: stable prefix with dynamic capabilities.",
];
