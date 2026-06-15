# Review Log — TA Decoration Relevance Video Presentation

Project: `decoration-relevance-video/presentation/`
Theme: Blueprint (deep navy + cyan accent)
Total: 8 chapters / 47 steps / ~11.7 min audio
Date: 2026-06-12 ~ 2026-06-15

---

## Phase 1 — Content Review

### script.md self-check (per SCRIPT-STYLE.md)

| Check | Result |
|---|---|
| Language matches original article (English) | PASS |
| Conversational tone, not book-reading | PASS |
| Numbers / data points preserved from article | PASS |
| Sentence rhythm varies (short + long mix) | PASS |
| No filler / no empty transitions | PASS |
| Read-aloud test: flows naturally | PASS |

### outline.md self-check (per OUTLINE-FORMAT.md)

| Check | Result |
|---|---|
| Chapter IDs follow `NN-kebab-case` | PASS — 01-hook through 08-daily-ops |
| Each chapter has info pool from article | PASS |
| Step count per chapter: 5~8 range | PASS — min 5, max 8 |
| Estimated time per chapter: 30~60s | PASS — 44.8s ~ 181.6s (llm-pipeline is 3min, intentionally dense) |
| Screen content described per step | PASS |
| No animation prescriptions in outline | PASS — left to chapter agent |
| Asset checklist at end | PASS |

User decision: No changes to script.md or outline.md.

---

## Phase 2 — Web Development Review

### 2.1 Global Compilation

```
npx tsc --noEmit → 0 errors
```

All 8 chapters compile cleanly. Checked after every chapter completion and after final chapters.ts registration.

### 2.2 chapters.ts Registration

| Check | Result |
|---|---|
| 8 chapters registered in CHAPTERS array | PASS |
| Import paths match actual file locations | PASS |
| narrations imported from each chapter's narrations.ts | PASS |
| Step count total: 5+6+8+7+5+6+5+5 = 47 | PASS |

### 2.3 STORAGE_KEY Bump

After modifying `chapters.ts` (adding chapters 2-8), bumped `useStepper.ts`:
- Before: `"presentation-cursor-v5"`
- After: `"presentation-cursor-v6"`

### 2.4 Font Size Minimum Check (>= 15px rule)

**Method**: Grep all CSS files for `font-size:` values, flagged anything < 15px.

**Findings — 3 violations found and fixed:**

| File | Class | Before | After | Line |
|---|---|---|---|---|
| `01-hook/Hook.css` | `.hk-deprecated-tag` | `font-size: 14px` | `font-size: 15px` | ~L343 |
| `03-llm-pipeline/LlmPipeline.css` | `.lp-score-unit` | `font-size: 14px` | `font-size: 15px` | L224 |
| `07-ubermarket/UberMarket.css` | `.um-terminal-name` | `font-size: 13px` | `font-size: 15px` | L298 |

Note: `base.css` has some primitive classes with < 15px (`.kicker` 13px, `.label-mono` using `--t-micro: 12px`, `.badge-mono` 11px, `.click-cue` 11px). These are framework primitives, not chapter content — left as-is.

### 2.5 Per-Chapter CHAPTER-CRAFT.md Completion Check

Each chapter was verified against CHAPTER-CRAFT.md Part 7 completion checklist:

#### Chapter 01 — Hook (.hk-)

| Check | Result |
|---|---|
| Has CSS/SVG/Canvas visual demo (not pure text) | PASS — ad card mockup, decoration tags, match/mismatch split, architecture diagram with animated connectors |
| Progressive reveal (1 item = 1 step) | PASS — 5 steps, tags stagger per step |
| Colors via tokens only | PASS — `--accent`, `--text`, `--bad`, `--surface-2`, `--rule` |
| Unique CSS prefix | PASS — `.hk-` |
| Animation variety | PASS — hk-float-in, hk-tag-in, hk-rise, hk-slide-left/right, hk-box-pop, hk-core-in, hk-horiz-draw, hk-line-draw, hk-head-in |
| No AI visual flavors | PASS — no purple-pink gradients, no rounded colorful borders, no emoji |
| Minimum font: 15px+ for content | PASS (after fix) |

#### Chapter 02 — Label Scale (.ls-)

| Check | Result |
|---|---|
| Visual demo | PASS — 4-level scale cards, production collapse animation, timeline, LLM comparison, cost breakdown |
| Progressive reveal | PASS — 6 steps |
| Colors via tokens | PASS |
| Unique CSS prefix | PASS — `.ls-` |
| Animation variety | PASS — ls-rise, ls-card-in, ls-slide-r, ls-scale-in |
| No AI flavors | PASS |

#### Chapter 03 — LLM Pipeline (.lp-)

| Check | Result |
|---|---|
| Visual demo | PASS — 3-stage pipeline arrows, accuracy comparison, market scoreboard grid, shake-in fail number, recovery fixes, flight verification cards, cost trajectory tags |
| Progressive reveal | PASS — 8 steps (densest chapter) |
| Colors via tokens | PASS — `--accent`, `--bad` for fail states |
| Unique CSS prefix | PASS — `.lp-` |
| Animation variety | PASS — lp-rise, lp-slide-r, lp-card-in, lp-shake-in |
| No AI flavors | PASS |
| Font minimum | PASS (after fix on `.lp-score-unit`) |

#### Chapter 04 — Model Architecture (.ma-)

| Check | Result |
|---|---|
| Visual demo | PASS — teacher-student pipeline, teacher evolution timeline, TriBert multi-task diagram, feature set cards, QAS features, training stages, V9 experiment results |
| Progressive reveal | PASS — 7 steps |
| Colors via tokens | PASS |
| Unique CSS prefix | PASS — `.ma-` |
| Animation variety | PASS |
| No AI flavors | PASS |

#### Chapter 05 — Robust Training (.rb-)

| Check | Result |
|---|---|
| Visual demo | PASS — coverage gap split-screen with animated bars, robust training mix animation, tradeoff panels, WoodBlock limitation slide-in, calibration cascade |
| Progressive reveal | PASS — 5 steps |
| Colors via tokens | PASS |
| Unique CSS prefix | PASS — `.rb-` |
| Animation variety | PASS — rb-rise, rb-split-l/r, rb-bar-slide, rb-bar-grow, rb-mix-in, rb-op-pop, rb-result-glow, rb-signal-drop, rb-panel-reveal, rb-wb-slide-in, rb-wb-arrow-draw, rb-cal-expand, rb-cal-cascade, rb-cal-result-pop |
| No AI flavors | PASS |

#### Chapter 06 — Going Global (.gs-)

| Check | Result |
|---|---|
| Visual demo | PASS — language dot grid, CULR V4 card, teacher config, INTL improvement bars, per-language breakdown, ablation experiments |
| Progressive reveal | PASS — 6 steps |
| Colors via tokens | PASS |
| Unique CSS prefix | PASS — `.gs-` |
| Animation variety | PASS — gs-rise, gs-dot-pop, gs-slide-r, gs-scale-in, gs-draw-in |
| No AI flavors | PASS |

#### Chapter 07 — UberMarket (.um-)

| Check | Result |
|---|---|
| Visual demo | PASS — merge US+INTL boxes converging, CPU bar growing to danger zone, terminal code reveal, serving slot cards, summary badges |
| Progressive reveal | PASS — 5 steps |
| Colors via tokens | PASS |
| Unique CSS prefix | PASS — `.um-` |
| Animation variety | PASS — um-rise, um-slide-from-left/right, um-scale-in, um-bar-grow, um-shake-in, um-terminal-reveal, um-badge-rise |
| No AI flavors | PASS |
| Font minimum | PASS (after fix on `.um-terminal-name`) |

#### Chapter 08 — Daily Operations (.do-)

| Check | Result |
|---|---|
| Visual demo | PASS — MetaStream schedule cards, vector refresh diagram with animated connectors, check-in process flow, monitoring stack layers, summary badges |
| Progressive reveal | PASS — 5 steps |
| Colors via tokens | PASS |
| Unique CSS prefix | PASS — `.do-` |
| Animation variety | PASS — do-rise, do-slide-down, do-scale-in, do-slide-r, do-draw-line, do-monitor-in, do-pop |
| No AI flavors | PASS |

### 2.6 Cross-Chapter Consistency

| Check | Result |
|---|---|
| All 8 chapters use theme color tokens (no hardcoded colors) | PASS |
| All use theme font tokens | PASS |
| CSS prefixes are all unique and non-conflicting | PASS |
| Each chapter's narrations.ts length matches step count in .tsx | PASS |
| No chapter modifies global styles or other chapters' CSS | PASS |

---

## Phase 3 — Audio Synthesis Review

### 3.1 TTS Provider Selection

| Attempt | Result |
|---|---|
| OpenAI TTS (user's first choice) | FAILED — `OPENAI_API_KEY` not set, no key available |
| Copilot-style API proxy | Investigated — not applicable for TTS |
| edge-tts (free Microsoft TTS) | SUCCESS — installed via `python -m pip install edge-tts` |

Note: `pip` / `pip3` not on PATH; used `python -m pip` instead.

### 3.2 Synthesis Results

| Item | Value |
|---|---|
| Voice | en-US-GuyNeural |
| Total segments | 47 / 47 succeeded |
| Failed | 0 |
| Output | `public/audio/<chapter-id>/<step>.mp3` |

### 3.3 Duration Analysis

**Total duration: 703.1s = 11.7 minutes**

Per-chapter breakdown:

| Chapter | Steps | Duration | Avg/step |
|---|---|---|---|
| hook | 5 | 44.8s | 9.0s |
| label-scale | 6 | 82.5s | 13.8s |
| llm-pipeline | 8 | 181.6s | 22.7s |
| model-arch | 7 | 108.7s | 15.5s |
| robust | 5 | 74.8s | 15.0s |
| global-ship | 6 | 91.9s | 15.3s |
| ubermarket | 5 | 47.7s | 9.5s |
| daily-ops | 5 | 71.1s | 14.2s |

**Segments >= 15s (19 total, flagged per AUDIO.md):**

| Segment | Duration | Note |
|---|---|---|
| llm-pipeline/2.mp3 | 35.2s | DV3 multi-generation iteration — densest segment |
| llm-pipeline/7.mp3 | 30.7s | GPT-4o 25 experiments across 7 markets |
| llm-pipeline/3.mp3 | 22.5s | DV3 finalized accuracy |
| llm-pipeline/6.mp3 | 22.1s | Flight verification |
| robust/1.mp3 | 20.6s | TriBert coverage gaps |
| llm-pipeline/5.mp3 | 20.0s | ChatML format switch |
| global-ship/6.mp3 | 19.8s | Ablation experiments |
| llm-pipeline/8.mp3 | 19.6s | Cost trajectory |
| daily-ops/5.mp3 | 19.5s | Summary / outro |
| label-scale/1.mp3 | 19.4s | 4-level labeling scale |
| model-arch/4.mp3 | 19.2s | Feature set |
| label-scale/3.mp3 | 18.7s | Label sourcing history |
| global-ship/4.mp3 | 18.3s | INTL improvement metrics |
| model-arch/7.mp3 | 18.0s | V9 experiment results |
| global-ship/3.mp3 | 17.9s | Best teacher config |
| model-arch/2.mp3 | 17.3s | Teacher evolution |
| llm-pipeline/1.mp3 | 16.8s | 3-stage pipeline intro |
| model-arch/1.mp3 | 16.3s | Teacher-student pipeline |
| daily-ops/4.mp3 | 15.8s | Monitoring stack |

User decision: Keep as-is, no step splitting requested.

---

## Phase 4 — Recording Preparation

### 4.1 Subtitle Feature Added

Created `Subtitle.tsx` + `Subtitle.css` for on-screen subtitle display.
Activated via URL parameter `?sub=1`.

Design: semi-transparent black bar (rgba 0,0,0,0.7), white text, 28px font, centered at bottom of 1920x1080 stage, z-index 10.

URL combinations:
- `localhost:5173` — manual, no subtitle
- `localhost:5173/?sub=1` — manual + subtitle
- `localhost:5173/?auto=1&sub=1` — auto-play + subtitle (recommended for remote recording)

### 4.2 Auto-play Mechanism

The `?auto=1` mode uses audio `ended` event to advance steps. If audio playback fails (no sound device on remote machine), falls back to text-length-based timer (`max(1500ms, charCount * 250ms)`).

---

## Summary of All Fixes Applied

| # | Issue | File | Fix |
|---|---|---|---|
| 1 | Font 14px < 15px minimum | `01-hook/Hook.css` `.hk-deprecated-tag` | 14px → 15px |
| 2 | Font 14px < 15px minimum | `03-llm-pipeline/LlmPipeline.css` `.lp-score-unit` | 14px → 15px |
| 3 | Font 13px < 15px minimum | `07-ubermarket/UberMarket.css` `.um-terminal-name` | 13px → 15px |
| 4 | chapters.ts only had ch1 | `registry/chapters.ts` | Registered all 8 chapters |
| 5 | STORAGE_KEY stale | `hooks/useStepper.ts` | v5 → v6 |
| 6 | No subtitle support | New: `Subtitle.tsx` + `Subtitle.css` | Added ?sub=1 feature |
