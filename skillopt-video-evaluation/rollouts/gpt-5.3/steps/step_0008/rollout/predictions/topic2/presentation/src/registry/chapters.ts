import type { ChapterDef } from "./types";
import FlowOverview from "../chapters/01-flow-overview/FlowOverview";
import { narrations as flowOverviewNarrations } from "../chapters/01-flow-overview/narrations";
import LocalDataImage from "../chapters/02-local-data-image/LocalDataImage";
import { narrations as localDataImageNarrations } from "../chapters/02-local-data-image/narrations";
import PolarisProduction from "../chapters/03-polaris-production/PolarisProduction";
import { narrations as polarisProductionNarrations } from "../chapters/03-polaris-production/narrations";
import OpsHardening from "../chapters/04-ops-hardening/OpsHardening";
import { narrations as opsHardeningNarrations } from "../chapters/04-ops-hardening/narrations";

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
    id: "flow-overview",
    title: "Deployment spine and failure surface",
    narrations: flowOverviewNarrations,
    Component: FlowOverview,
  },
  {
    id: "local-data-image",
    title: "Local validation and parallel prep",
    narrations: localDataImageNarrations,
    Component: LocalDataImage,
  },
  {
    id: "polaris-production",
    title: "Polaris quality gate to production cutover",
    narrations: polarisProductionNarrations,
    Component: PolarisProduction,
  },
  {
    id: "ops-hardening",
    title: "Environment separation and observability reliability",
    narrations: opsHardeningNarrations,
    Component: OpsHardening,
  },
];
