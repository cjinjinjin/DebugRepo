import type { ChapterDef } from "./types";
import HookChapter from "../chapters/01-hook/Hook";
import { narrations as hookNarrations } from "../chapters/01-hook/narrations";
import LabelScaleChapter from "../chapters/02-label-scale/LabelScale";
import { narrations as labelScaleNarrations } from "../chapters/02-label-scale/narrations";
import LlmPipelineChapter from "../chapters/03-llm-pipeline/LlmPipeline";
import { narrations as llmPipelineNarrations } from "../chapters/03-llm-pipeline/narrations";
import ModelArchChapter from "../chapters/04-model-arch/ModelArch";
import { narrations as modelArchNarrations } from "../chapters/04-model-arch/narrations";
import RobustChapter from "../chapters/05-robust/Robust";
import { narrations as robustNarrations } from "../chapters/05-robust/narrations";
import GlobalShipChapter from "../chapters/06-global-ship/GlobalShip";
import { narrations as globalShipNarrations } from "../chapters/06-global-ship/narrations";
import UberMarketChapter from "../chapters/07-ubermarket/UberMarket";
import { narrations as uberMarketNarrations } from "../chapters/07-ubermarket/narrations";
import DailyOpsChapter from "../chapters/08-daily-ops/DailyOps";
import { narrations as dailyOpsNarrations } from "../chapters/08-daily-ops/narrations";

/**
 * Order = order of presentation.
 *
 * Each chapter MUST provide a `narrations: Narration[]` array. Its length
 * is the chapter's step count — there is no `totalSteps` to maintain
 * separately. This guarantees the audio synthesis pipeline, the runtime
 * stepper, and the chapter `.tsx` switch on `step` cannot drift apart.
 *
 * Visual styling (color, fonts) comes entirely from the active theme —
 * chapters never hard-code palette / font names. See THEMES.md.
 */
export const CHAPTERS: ChapterDef[] = [
  {
    id: "hook",
    title: "What Is Decoration Relevance?",
    narrations: hookNarrations,
    Component: HookChapter,
  },
  {
    id: "label-scale",
    title: "Labeling at Scale",
    narrations: labelScaleNarrations,
    Component: LabelScaleChapter,
  },
  {
    id: "llm-pipeline",
    title: "LLM Labeling Pipeline",
    narrations: llmPipelineNarrations,
    Component: LlmPipelineChapter,
  },
  {
    id: "model-arch",
    title: "Model Architecture & Features",
    narrations: modelArchNarrations,
    Component: ModelArchChapter,
  },
  {
    id: "robust",
    title: "Robust Training & Coverage",
    narrations: robustNarrations,
    Component: RobustChapter,
  },
  {
    id: "global-ship",
    title: "Going Global",
    narrations: globalShipNarrations,
    Component: GlobalShipChapter,
  },
  {
    id: "ubermarket",
    title: "UberMarket Migration",
    narrations: uberMarketNarrations,
    Component: UberMarketChapter,
  },
  {
    id: "daily-ops",
    title: "Daily Operations",
    narrations: dailyOpsNarrations,
    Component: DailyOpsChapter,
  },
];
