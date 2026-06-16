import type { ChapterDef } from "./types";

import HookChapter from "../chapters/01-hook/Hook";
import { narrations as hookNarrations } from "../chapters/01-hook/narrations";
import PrefixLawChapter from "../chapters/02-prefix-law/PrefixLaw";
import { narrations as prefixLawNarrations } from "../chapters/02-prefix-law/narrations";
import FragilityChapter from "../chapters/03-fragility/Fragility";
import { narrations as fragilityNarrations } from "../chapters/03-fragility/narrations";
import StateToolsChapter from "../chapters/04-state-and-tools/StateTools";
import { narrations as stateToolsNarrations } from "../chapters/04-state-and-tools/narrations";
import ModelRoutingChapter from "../chapters/05-model-routing/ModelRouting";
import { narrations as modelRoutingNarrations } from "../chapters/05-model-routing/narrations";
import CompactionChapter from "../chapters/06-compaction/Compaction";
import { narrations as compactionNarrations } from "../chapters/06-compaction/narrations";
import OperatingModelChapter from "../chapters/07-operating-model/OperatingModel";
import { narrations as operatingModelNarrations } from "../chapters/07-operating-model/narrations";

export const CHAPTERS: ChapterDef[] = [
  { id: "hook", title: "Caching Is Product Infrastructure", narrations: hookNarrations, Component: HookChapter },
  { id: "prefix-law", title: "Prefix Matching Mental Model", narrations: prefixLawNarrations, Component: PrefixLawChapter },
  { id: "fragility", title: "Easy Ways Cache Breaks", narrations: fragilityNarrations, Component: FragilityChapter },
  { id: "state-and-tools", title: "Stable Prefix, Dynamic Behavior", narrations: stateToolsNarrations, Component: StateToolsChapter },
  { id: "model-routing", title: "Model Switching Cost Trap", narrations: modelRoutingNarrations, Component: ModelRoutingChapter },
  { id: "compaction", title: "Cache-Safe Forking", narrations: compactionNarrations, Component: CompactionChapter },
  { id: "operating-model", title: "Five Operating Rules", narrations: operatingModelNarrations, Component: OperatingModelChapter },
];
