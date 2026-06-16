import type { ChapterDef } from "./types";
import ShiftToSystemsChapter from "../chapters/01-shift-to-systems/ShiftToSystems";
import { narrations as shiftToSystemsNarrations } from "../chapters/01-shift-to-systems/narrations";
import WhyNowChapter from "../chapters/02-why-now/WhyNow";
import { narrations as whyNowNarrations } from "../chapters/02-why-now/narrations";
import PrimitivesMemoryChapter from "../chapters/03-primitives-and-memory/PrimitivesMemory";
import { narrations as primitivesMemoryNarrations } from "../chapters/03-primitives-and-memory/narrations";
import PatternLibraryChapter from "../chapters/04-pattern-library/PatternLibrary";
import { narrations as patternLibraryNarrations } from "../chapters/04-pattern-library/narrations";
import OperatingToolchainChapter from "../chapters/05-operating-toolchain/OperatingToolchain";
import { narrations as operatingToolchainNarrations } from "../chapters/05-operating-toolchain/narrations";
import SafetyGovernanceChapter from "../chapters/06-safety-and-governance/SafetyGovernance";
import { narrations as safetyGovernanceNarrations } from "../chapters/06-safety-and-governance/narrations";
import AdoptionPlaybookChapter from "../chapters/07-adoption-playbook/AdoptionPlaybook";
import { narrations as adoptionPlaybookNarrations } from "../chapters/07-adoption-playbook/narrations";

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
    id: "shift-to-systems",
    title: "The Shift: Prompting → System Design",
    narrations: shiftToSystemsNarrations,
    Component: ShiftToSystemsChapter,
  },
  {
    id: "why-now",
    title: "Why Teams Care Now",
    narrations: whyNowNarrations,
    Component: WhyNowChapter,
  },
  {
    id: "primitives-and-memory",
    title: "The Core Primitives",
    narrations: primitivesMemoryNarrations,
    Component: PrimitivesMemoryChapter,
  },
  {
    id: "pattern-library",
    title: "Pattern Library as Execution Recipes",
    narrations: patternLibraryNarrations,
    Component: PatternLibraryChapter,
  },
  {
    id: "operating-toolchain",
    title: "Tooling and Operational Readiness",
    narrations: operatingToolchainNarrations,
    Component: OperatingToolchainChapter,
  },
  {
    id: "safety-and-governance",
    title: "Safety, Failure Modes, and Human Gates",
    narrations: safetyGovernanceNarrations,
    Component: SafetyGovernanceChapter,
  },
  {
    id: "adoption-playbook",
    title: "Practical Adoption Path",
    narrations: adoptionPlaybookNarrations,
    Component: AdoptionPlaybookChapter,
  },
];
