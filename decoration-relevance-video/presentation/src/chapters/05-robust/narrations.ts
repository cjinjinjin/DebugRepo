import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "Now, the production TriBert has some gaps. Query vectors return all zeros sometimes in livesite. Item vector coverage varies \u2014 Sitelinks hit 99%, but FourthLine is only at 80%. SmartCategory is at 90%. Ad vector coverage sits at 93%.",
  "So we do robust training. We mix in zero vectors during training \u2014 controlled amounts \u2014 so the model learns to handle missing features gracefully. Two signals track this: IsBertValid and IsWBValid.",
  "The tradeoff is intentional. You accept a small AUC drop on normal validation in exchange for much better livesite AUC when features are missing. Pick the right zero vector ratio, and both sides stay acceptable.",
  "WoodBlock deserves a note. Right now we only have a WoodBlock trained on Sitelink and SLAB data \u2014 query-decoration relevance pairs. For other decoration types, it just returns a default value of 0.",
  "Calibration matters too. Extensions and annotations share the same display position as sitelinks. So filtering thresholds need to be tuned per task, not per individual decoration type.",
];
