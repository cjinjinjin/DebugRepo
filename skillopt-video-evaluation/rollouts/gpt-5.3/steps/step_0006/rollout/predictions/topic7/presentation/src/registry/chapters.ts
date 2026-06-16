import type { ChapterDef } from "./types";
import HookHeat from "../chapters/01-hook-heat/HookHeat";
import { narrations as hookHeatNarrations } from "../chapters/01-hook-heat/narrations";
import ConceptDeploy from "../chapters/02-concept-deploy/ConceptDeploy";
import { narrations as conceptDeployNarrations } from "../chapters/02-concept-deploy/narrations";
import FeishuSecurityUsecases from "../chapters/03-feishu-security-usecases/FeishuSecurityUsecases";
import { narrations as fsuNarrations } from "../chapters/03-feishu-security-usecases/narrations";

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
    id: "hook-heat",
    title: "现象级开场",
    narrations: hookHeatNarrations,
    Component: HookHeat,
  },
  {
    id: "concept-deploy",
    title: "概念与部署",
    narrations: conceptDeployNarrations,
    Component: ConceptDeploy,
  },
  {
    id: "feishu-security-usecases",
    title: "飞书、技能与安全落地",
    narrations: fsuNarrations,
    Component: FeishuSecurityUsecases,
  },
];
