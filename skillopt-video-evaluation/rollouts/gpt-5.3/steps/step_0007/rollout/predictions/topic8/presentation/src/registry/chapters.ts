import type { ChapterDef } from "./types";
import FoundationGapChapter from "../chapters/01-foundation-gap/FoundationGap";
import { narrations as foundationGapNarrations } from "../chapters/01-foundation-gap/narrations";
import MethodsLadderChapter from "../chapters/02-methods-ladder/MethodsLadder";
import { narrations as methodsLadderNarrations } from "../chapters/02-methods-ladder/narrations";
import DatasetEngineChapter from "../chapters/03-dataset-engine/DatasetEngine";
import { narrations as datasetEngineNarrations } from "../chapters/03-dataset-engine/narrations";
import ScoringPipelineChapter from "../chapters/04-scoring-pipeline/ScoringPipeline";
import { narrations as scoringPipelineNarrations } from "../chapters/04-scoring-pipeline/narrations";
import BlindtestLoopChapter from "../chapters/05-blindtest-loop/BlindtestLoop";
import { narrations as blindtestLoopNarrations } from "../chapters/05-blindtest-loop/narrations";

export const CHAPTERS: ChapterDef[] = [
  { id: "foundation-gap", title: "为什么通用榜单救不了垂域", narrations: foundationGapNarrations, Component: FoundationGapChapter },
  { id: "methods-ladder", title: "三种评估方法怎么选", narrations: methodsLadderNarrations, Component: MethodsLadderChapter },
  { id: "dataset-engine", title: "一键生成测试集", narrations: datasetEngineNarrations, Component: DatasetEngineChapter },
  { id: "scoring-pipeline", title: "自动评估如何跑稳", narrations: scoringPipelineNarrations, Component: ScoringPipelineChapter },
  { id: "blindtest-loop", title: "人工盲测与闭环", narrations: blindtestLoopNarrations, Component: BlindtestLoopChapter },
];
