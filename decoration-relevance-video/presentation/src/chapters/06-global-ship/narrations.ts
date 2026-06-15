import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "Now the big move — going global. Before, we had a US-only CLR teacher. International markets were basically getting a model that didn't understand their language.",
  "CULR V4 fixed that. Cross-lingual, multilingual, trained on data from German, French, Chinese, Japanese, Portuguese, Danish.",
  "The best teacher config used query + web4Ads + QU1 as text_a, and adcopy + decorationText as text_b. Learning rate 5e-6, gradient accumulation steps 8, one epoch, max sequence length 256.",
  "The improvement on international markets was dramatic. INTL BSCAP on the test set went from 0.617 to 0.934. Goldset went from 0.533 to 0.757. Not incremental — a completely different league.",
  "Per-language breakdown: German at 0.88, French at 0.90, Simplified Chinese leading at 0.91. Even Japanese, which is notoriously hard for cross-lingual models, reached 0.84.",
  "We ran ablation experiments. web4Ads gave marginal gains on goldset, minimal on test. QU1 helped some splits but not others. EEM Rewrites for data augmentation gave a solid INTL boost — QBSCAPTerm goldset jumped from 0.589 to 0.626.",
];
