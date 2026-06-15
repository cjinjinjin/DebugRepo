import type { Narration } from "../../registry/types";

export const narrations: Narration[] = [
  "What runs every day? MetaStream schedules handle three things: daily new impression annotation, combining 180 days of history with extensions, and SLAB plus Image Extension streams.",
  "Document vectors and ads vectors refresh on their own schedules. Vector publishing goes xlite first, then prod — and bin file generation is limited to once daily.",
  "Check-in process for new model versions follows a specific path. You create a PR in the Rnr-Offline-ClickPrediction repo, follow the deployment OneNote steps, and use the FPS publish portal for vector publishing.",
  "The monitoring stack covers multiple layers. PowerBI dashboards for live metrics. Aether pipelines for offline monitoring. Cosmos SStream monitors for both Gene vector publish and TriBert decoration relevance vectors.",
  "That's TA Decoration Relevance. A unified model, four task types, teacher-student pipeline, LLM labeling that dropped costs 27x, robust training for livesite, and now it works globally. If you're onboarding to this system, start with the DRI handbook — it'll walk you through the operational details.",
];
