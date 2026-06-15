import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "You ever wonder why some ads show up with a bunch of extra links and callouts, while others are just... bare?",
  "That's decoration relevance. We look at those extras — sitelinks, callouts, structured snippets, price extensions — and ask: does this actually match what you were searching for?",
  "If it doesn't match, it shouldn't be there. A bad decoration is worse than no decoration at all.",
  "We split decorations into four sub-tasks. Task 0 handles Sitelinks and SLAB. The others cover everything from SmartCategory and StructuredSnippets to Callouts and Price Extensions.",
  "And we built one unified model for all of them. Same architecture, different classification heads per task.",
];
