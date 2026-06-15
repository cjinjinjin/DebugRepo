import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "We've got a teacher-student pipeline. The teacher is CULR V4 \u2014 Cross-lingual Universal Language Representation, version 4. It generates soft labels. Then TriBert V9, the student, learns from those and runs in production.",
  "The teacher evolved over time. CLR V2 was the US-only baseline. CLR V3 improved on it. CULR V3 went cross-lingual. CULR V4 is the current best \u2014 strongest global performance across all tasks.",
  "TriBert is a multi-task model. Four tasks share the same BERT backbone, but each has its own classification head. So sitelinks, callouts, snippets \u2014 same encoder, scored differently.",
  "The feature set is rich. Metastream features give query, ad, and decoration signals. RC2 features handle contextual relevance \u2014 things like QKRelevance score and QACDefect score. WoodBlock does lightweight feature extraction. TriBert ties it all together.",
  "Query features come from QAS \u2014 health, navigational, adult, image, commerce, finance, video signals. About two dozen categories total.",
  "Training happens in two stages. Stage 1: finetune on managed human labels from years of judge data. Stage 2: finetune on LLM-generated labels. We found two-stage beats either source alone.",
  "The V9 experiments proved it clearly. The CLR V2 baseline scored 0.8527 on US test. CULR V4 with two-stage training? 0.9541. Starting from a finetuned checkpoint instead of a pretrained one \u2014 consistently better.",
];
