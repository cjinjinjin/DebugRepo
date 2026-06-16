import type { ChapterDef } from "./types";
import ColdopenChapter from "../chapters/01-coldopen-strength/Coldopen";
import { narrations as coldopenNarrations } from "../chapters/01-coldopen-strength/narrations";
import AccessRoutesChapter from "../chapters/02-access-routes/AccessRoutes";
import { narrations as accessRoutesNarrations } from "../chapters/02-access-routes/narrations";
import CasePlaybookChapter from "../chapters/03-case-playbook/CasePlaybook";
import { narrations as casePlaybookNarrations } from "../chapters/03-case-playbook/narrations";
import SkillSystemChapter from "../chapters/04-skill-system/SkillSystem";
import { narrations as skillSystemNarrations } from "../chapters/04-skill-system/narrations";
import RolloutNextChapter from "../chapters/05-rollout-next/RolloutNext";
import { narrations as rolloutNextNarrations } from "../chapters/05-rollout-next/narrations";

export const CHAPTERS: ChapterDef[] = [
  { id: "coldopen-strength", title: "GPT-Image-2 断层领先", narrations: coldopenNarrations, Component: ColdopenChapter },
  { id: "access-routes", title: "可用渠道与接入路径", narrations: accessRoutesNarrations, Component: AccessRoutesChapter },
  { id: "case-playbook", title: "案例库与玩法边界", narrations: casePlaybookNarrations, Component: CasePlaybookChapter },
  { id: "skill-system", title: "Skill 方法论", narrations: skillSystemNarrations, Component: SkillSystemChapter },
  { id: "rollout-next", title: "立即上手路径", narrations: rolloutNextNarrations, Component: RolloutNextChapter },
];

