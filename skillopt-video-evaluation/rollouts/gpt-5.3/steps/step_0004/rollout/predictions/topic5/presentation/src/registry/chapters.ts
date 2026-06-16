import type { ChapterDef } from "./types";
import ColdopenChapter from "../chapters/01-coldopen/Coldopen";
import { narrations as coldopenNarrations } from "../chapters/01-coldopen/narrations";
import PromptCoreChapter from "../chapters/02-prompt-core/PromptCore";
import { narrations as promptCoreNarrations } from "../chapters/02-prompt-core/narrations";
import OperatingRulesChapter from "../chapters/03-operating-rules/OperatingRules";
import { narrations as operatingRulesNarrations } from "../chapters/03-operating-rules/narrations";
import AntiAiDesignChapter from "../chapters/04-anti-ai-design/AntiAiDesign";
import { narrations as antiAiDesignNarrations } from "../chapters/04-anti-ai-design/narrations";
import SkillRolloutChapter from "../chapters/05-skill-rollout/SkillRollout";
import { narrations as skillRolloutNarrations } from "../chapters/05-skill-rollout/narrations";

export const CHAPTERS: ChapterDef[] = [
  { id: "coldopen", title: "Claude Design 到底改了什么", narrations: coldopenNarrations, Component: ColdopenChapter },
  { id: "prompt-core", title: "420 行提示词骨架", narrations: promptCoreNarrations, Component: PromptCoreChapter },
  { id: "operating-rules", title: "可复用执行规则", narrations: operatingRulesNarrations, Component: OperatingRulesChapter },
  { id: "anti-ai-design", title: "去 AI 味与视觉系统", narrations: antiAiDesignNarrations, Component: AntiAiDesignChapter },
  { id: "skill-rollout", title: "方法提炼成 Skill", narrations: skillRolloutNarrations, Component: SkillRolloutChapter },
];
