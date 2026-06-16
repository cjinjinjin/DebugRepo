import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "Prompt caching is prefix matching from the first token.",
  "The good ordering is static first, dynamic last.",
  "That ordering creates a large shared prefix across sessions.",
  "Bad ordering pushes volatile fields into the front and destroys reuse.",
  "Rule card: put static blocks first and keep dynamic tails at the end.",
];
