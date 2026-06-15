# Video Outline

> **Theme**: `blueprint` — clean engineering blueprint style, technical diagrams, monospaced data
> **Total Duration**: ~7 min 30 sec (~1800 words / 4 words per sec)
> **Chapters**: 8 chapters / 56 steps

---

## 1. hook — What Is Decoration Relevance? (5 steps · ~30s)

**Info Pool** (from article):
- Decoration types: SiteLink, SLAB, LongDescription, DynamicDescription, SmartCategory, StructuredSnippetExtension, CalloutExtension, PriceExtension — article §Background table
- 4-level scale: Excellent / Good / Fair / Bad → binary NonBad / Bad — article §Labeling Scale
- Sub-task split: TaskId 0/1/2/3 by decoration content and display position — article §Background table
- Feature sets vary per task: Metastream, RC2, WoodBlock, TriBert — article §Background table

**Dev Plan**:

- step 1 (~5s) — Mock search result ad with bare text only; hook question "why do some ads have extra links?"
- step 2 (~5s) — Same ad now with decoration elements visible: sitelinks, callouts, snippets labeled
- step 3 (~7s) — Hero text: "Does this decoration match the query?" with split: matching vs non-matching example
- step 4 (~7s) — 4 task boxes visible: Task 0 (Sitelinks), Task 2 (Snippets), Task 3 (Callouts); Task 1 dimmed as deprecated
- step 5 (~6s) — Unified model diagram: one architecture box with 4 classification head branches

Script excerpt:
> You ever wonder why some ads show up with a bunch of extra links... That's decoration relevance. Does this actually match what you were searching for?

---

## 2. label-scale — Labeling Scale & Data Sources (6 steps · ~40s)

**Info Pool** (from article):
- 4-level scale definitions: E = matches query, G = relevant not specific, F = weakly relevant, B = off-intent — article §Labeling Scale
- Binary collapse: E/G/F → NonBad, B → Bad — article §Labeling Scale
- Managed labeling period: 2017.9 – 2023.12 — article §Label Data
- Cost per item: managed $0.238, DV3 $0.184, GPT4-Turbo $0.019, GPT-4o $0.0072 — article §LLM labeling cost table
- GPT-4o accuracy: ~83% across tasks — article §LLM labeling accuracy
- Token pricing: GPT-4o $2.5/1M input, $7.5/1M output — article §LLM labeling cost footnote
- 27x cost reduction from managed to GPT-4o — derived from $0.238 / $0.0072

**Dev Plan**:

- step 1 (~7s) — Labeling scale: 4 tiers — Excellent, Good, Fair, Bad — each with one-line definition
- step 2 (~6s) — Binary collapse diagram: E/G/F grouped as "NonBad", B isolated as "Bad → filter out"
- step 3 (~7s) — Timeline bar: 2017 → 2023.12 labeled "Managed Labeling Era". Cost badge: $0.238/item
- step 4 (~6s) — Timeline extends: 2023.12 → now labeled "LLM Labeling Era". Three model icons: DV3, GPT-4T, GPT-4o
- step 5 (~7s) — Accuracy gauge showing 83%. GPT-4o badge highlighted
- step 6 (~7s) — Cost waterfall chart: $0.238 → $0.184 → $0.019 → $0.0072. Hero number: "27x cheaper"

Script excerpt:
> Excellent means the decoration matches the query directly. Bad means it shouldn't be there at all. In production, we collapse it — E/G/F all count as NonBad.

---

## 3. llm-pipeline — LLM Labeling Pipeline Evolution (8 steps · ~60s)

**Info Pool** (from article):
- 3-stage architecture: QU → Decoration Intent → Crossing — article §LLM Pipeline Architecture
- DV3 iteration: baseline 53.48% → Prompt 5 at 69.76% — key change "evaluate query-decoration, not ad-query" — article §5.1
- DV3 finalized: US 85.64%, DE 81.50%, FR 84.79%, JA 61.96%, PT 63.27% — article §5.1 multi-market
- GPT-4T raw accuracy: 41.96% (directly applying DV3 prompts) — article §5.2
- GPT-4T required: ChatML format, stop tokens, token limit 200/100, frequency/presence penalty — article §5.2
- GPT-4T flight: DCO -39.27% DR diff, HSL +17.77%, VSL -11.17%, p<0.05 — article §5.2 flight verification
- GPT-4o #6 "short explanations" 84.53%, #22 balanced 74.31% — article §5.3
- GPT-4o cost: $0.0072/item, 25x reduction in 18 months — article §5.3
- GPT-4o-1120: 10-item evaluation criteria — article §5.4
- Cost trajectory: $0.184 → $0.019 → $0.0072 — derived

**Dev Plan**:

- step 1 (~7s) — 3-stage pipeline: Query Understanding → Decoration Intent → Crossing, with connecting arrows
- step 2 (~7s) — DV3 era (2023): 13 iterations badge. Key result: accuracy 53% → 70%. Quote callout: "evaluate query-decoration relevance"
- step 3 (~8s) — DV3 finalized multi-market results: US 85.64%, DE 81.50%, FR 84.79%, JA 61.96%, PT 63.27% in a scoreboard
- step 4 (~7s) — GPT-4 Turbo (2024): "42%" large error display. Required changes listed: ChatML format, token limits, penalty tuning
- step 5 (~7s) — GPT-4T recovery: 18 experiments counter, accuracy recovered to 86.90%. Flight result: DCO -39% defect, HSL +17.77%
- step 6 (~8s) — GPT-4o: two paths side by side — "judge first" (73%) vs "explain then judge" (84%). Short > long explanations
- step 7 (~8s) — Cost comparison: three price tags — $0.184, $0.019, $0.0072. "25x reduction in 18 months"
- step 8 (~8s) — GPT-4o-1120 migration: 10 evaluation criteria in a compact list (ground truth, parallel runs, agreement rate, tokens, DSAT, sensitive segments, SLA, auditing, prompt review, case study)

Script excerpt:
> The biggest breakthrough was one prompt change — evaluate query-decoration relevance. That jumped accuracy from 53% to nearly 70%.

---

## 4. model-arch — Model Architecture & Features (7 steps · ~50s)

**Info Pool** (from article):
- Teacher evolution: CLR V2 → CLR V3 → CULR V3 → CULR V4 (current) — article §Teacher Student Pipeline
- Student: TriBert V9, multi-task with shared BERT backbone + per-task classification heads — article §Tribert
- Features: Metastream (query/ad/decoration signals), RC2 (QKRelevance PLF_31, QACDefect PLF_33, QLPDefect PLF_34), WoodBlock, TriBert — article §Model Structure + §Detail Features
- Query features from QAS: ~24 categories (health, navigational, adult, commerce, finance, etc.) — article §Query features
- Two-stage training: Stage 1 managed labels → Stage 2 LLM labels — article §Two-Stage Training
- Key finding: CULR > CLR, Two-stage > one-stage, Finetune CKPT > pretrain CKPT — article §V9 experiments
- V9 experiment results: CLR V2 baseline 0.8527, CULR V4 two-stage 0.9541 — article §Student Model Experiments

**Dev Plan**:

- step 1 (~7s) — Architecture overview: flow diagram — Metastream, RC2, WoodBlock, TriBert boxes feeding into unified FR ranker
- step 2 (~7s) — Teacher-Student pipeline: CULR V4 box → "soft labels" → TriBert V9 box → "production" output
- step 3 (~7s) — Teacher evolution timeline: CLR V2 → V3 → CULR V3 → CULR V4, each with checkmark. CULR V4 highlighted as current
- step 4 (~7s) — TriBert detail: shared BERT backbone center, 4 classification heads branching out (Task 0/1/2/3)
- step 5 (~8s) — Feature breakdown: Metastream (query/ad/decoration), RC2 (PLF_31/33/34 scores), QAS (~24 categories), WoodBlock
- step 6 (~7s) — Two-stage training: Stage 1 (managed data, human icon) → checkpoint → Stage 2 (LLM data, robot icon)
- step 7 (~7s) — Three key findings listed: "CULR > CLR" / "Two-stage > One-stage" / "Finetune CKPT > Pretrain CKPT". V9 best: 0.9541

Script excerpt:
> The teacher is CULR V4 — Cross-lingual Universal Language Representation. TriBert V9 learns from those labels and runs in production.

---

## 5. robust — Robust Training & Coverage (6 steps · ~45s)

**Info Pool** (from article):
- Zero vector mixing strategy for livesite failures — article §Robust Training
- IsBertValid / IsWBValid signals — article §Robust Training
- Coverage: SL/SLAB 99%, FL 80%, SSExtension 99%, SmartCategory 90%, CO 99%, DCO 90%, Ad vector 93% — article §Tribert coverage
- Query vector returns all zeros in livesite — article §Tribert
- Tradeoff: normal AUC drop vs livesite AUC/AvgRelScore drop — article §Robust Training evaluation
- WoodBlock only trained on SL/SLAB data, default 0 for others — article §Woodblock
- Calibration: per-task threshold tuning, not per-decoration — article §Calibration

**Dev Plan**:

- step 1 (~7s) — Split screen: "Normal" with full feature vectors vs "Livesite" with zeroed-out vectors, broken connections shown
- step 2 (~8s) — Robust training concept: real vectors + zero vectors mixed into training pipeline. IsBertValid / IsWBValid signal labels
- step 3 (~8s) — Coverage dashboard: horizontal bars per decoration type — SL 99%, FL 80%, SSExt 99%, SmartCat 90%, CO 99%, DCO 90%, Ad 93%
- step 4 (~7s) — Tradeoff panel: two AUC meters side by side — "Normal validation" vs "Livesite validation", showing the tradeoff
- step 5 (~8s) — WoodBlock note: only trained on SL/SLAB query-decoration pairs, other decorations get default 0
- step 6 (~7s) — Calibration: same display position → per-task threshold tuning, not per-decoration

Script excerpt:
> We mix in zero vectors during training so the model handles missing features. Sitelink vectors hit 99%. But it's not 100%, so robustness matters.

---

## 6. global-ship — Going Global with CULR V4 (7 steps · ~55s)

**Info Pool** (from article):
- US baseline INTL BSCAP test: 0.617 → CULR V4: 0.934 — article §Global Model Ship prod baseline
- US baseline INTL goldset BSCAP: 0.533 → 0.757 — article §Global Model Ship prod baseline
- Per-language AUC: de 0.88, fr 0.90, zh-Hans 0.91, zh-Hant 0.86, ja 0.84, pt-BR 0.89, da 0.89 — article §Per-Language Results
- Teacher config: text_a = query + web4Ads + QU1, text_b = adcopy + decorationText, lr 5e-6, grad_accum 8, epoch 1, max_seq 256 — article §Best Teacher Configuration
- EEM Rewrites: INTL QBSCAPTerm goldset 0.589→0.626 — article §Key Experiment Findings
- Ablation: web4Ads marginal, QU1 mixed, EEM solid — article §Key Experiment Findings
- Student V9 experiments: 5 configs, best #6.1 two-stage at 0.9541 — article §Student Model Experiments

**Dev Plan**:

- step 1 (~7s) — World map dimmed, US highlighted. Caption: "US-only CLR teacher"
- step 2 (~7s) — Map with multiple countries lit: DE, FR, CN, JP, PT-BR, DA. "CULR V4: cross-lingual, multilingual"
- step 3 (~8s) — Teacher config detail: text_a / text_b inputs, hyperparameters (lr 5e-6, grad_accum 8, epoch 1, max_seq 256)
- step 4 (~9s) — Before/After panel: INTL BSCAP 0.617 → 0.934, goldset 0.533 → 0.757. Hero: "Not incremental — a different league"
- step 5 (~8s) — Per-language scoreboard: flag + language + AUC (de 0.88, fr 0.90, zh-Hans 0.91 highlighted, ja 0.84, pt-BR 0.89, da 0.89)
- step 6 (~8s) — Ablation findings: web4Ads (marginal), QU1 (mixed), EEM Rewrites (0.589→0.626 solid boost)
- step 7 (~8s) — Student V9 results table, best config #6.1 (two-stage, 0.9541) highlighted. Pretrain CKPT worse than finetune CKPT

Script excerpt:
> INTL BSCAP test set went from 0.617 to 0.934. Not incremental — a completely different league.

---

## 7. ubermarket — UberMarket & ConditionalCall (5 steps · ~35s)

**Info Pool** (from article):
- UberMarket migration completed 2024/07/31 on PaidSearch in North America — article §UberMarket Migration
- ConditionalCall: Queene model only serves Bing O&O (MarketplaceClassifications 1/2, RelatedToAccountId 1004) — article §ConditionalCall Logic
- NA slot: Queene69, INTL slot: Queene13 — article §ConditionalCall Logic
- Default output: DefaultValueVector, size 64, value 0 — article §ConditionalCall code
- CPU increase < 1%, no extra machines — article §UberMarket Migration
- DisableGene setting maintained for TGs that disabled Gene — article §UberMarket Migration

**Dev Plan**:

- step 1 (~7s) — UberMarket migration: US and INTL boxes merging into single unified infrastructure. Date: 2024/07/31
- step 2 (~7s) — Latency challenge: CPU usage bar high, then ConditionalCall solution brings it under 1%
- step 3 (~7s) — ConditionalCall logic code block: MarketplaceClassifications 1/2, Bing O&O highlighted, others → default zero vector
- step 4 (~7s) — Two serving slots: NA = Queene69, INTL = Queene13. Default vector: size 64, all zeros
- step 5 (~7s) — Summary badge: "Unified infrastructure. <1% CPU. No extra machines."

Script excerpt:
> ConditionalCall — Queene model only serves Bing O&O. Everyone else gets a default zero vector. CPU increase? Under 1%.

---

## 8. daily-ops — Daily Operations & Monitoring (7 steps · ~50s)

**Info Pool** (from article):
- MetaStream schedules: daily impression annotation (61e77843), 180-day combine (cbbd73f3), SLAB/ImageExt (b0ad06fc) — article §Daily running pipeline
- Doc vector refresh: 1c395c0f, Ads vector refresh: c8f1bd40 — article §Daily running pipeline
- Vector publish: xlite (08fc2378) before prod (b0c2e142), bin file once/day — article §Daily running pipeline
- FPS publish portal: fpsmanager.azurewebsites.net for xlite and prod — article §MS/MW check in
- Model check-in: PR in Rnr-Offline-ClickPrediction repo (PR 5230244 example) — article §Model check in
- Monitor: PowerBI dashboard, Aether pipeline (af8103bf), Cosmos SStream monitors — article §Appendix
- Data paths: managed labels, LLM labels, GoldSet — all on Cosmos shares — article §Data Paths
- DRI handbook reference — article §DRI handbook

**Dev Plan**:

- step 1 (~7s) — Daily pipeline flow: MetaStream → impression annotation → 180-day combine → SLAB/ImageExt streams
- step 2 (~7s) — Vector refresh schedules: doc vectors and ads vectors on separate cycles
- step 3 (~7s) — Publish sequence: xlite → gate → prod. "Bin files: once a day, xlite before prod"
- step 4 (~7s) — Model check-in process: create PR in Rnr-Offline-ClickPrediction, follow deployment steps, FPS portal publish
- step 5 (~7s) — Data paths panel: managed labels path, LLM labels path, GoldSet path — all Cosmos share locations
- step 6 (~8s) — Monitoring stack: PowerBI dashboard + Aether pipeline + Cosmos SStream monitors (Gene vector + TriBert decoration)
- step 7 (~7s) — Closing: unified model + 4 tasks + global map + "27x cheaper" + daily pipeline. "Start with the DRI handbook"

Script excerpt:
> MetaStream schedules handle daily impression annotation, history combining, and SLAB streams. Xlite publishes before prod. Bin files once a day.

---

## Asset Checklist

### 1. hook
- ⚠️ Mock search result ad wireframe (CSS/SVG drawn)
- ⚠️ Decoration type labels (sitelinks, callouts, snippets — CSS text)

### 2. label-scale
- ⚠️ 4-level labeling scale visualization (CSS-drawn)
- ⚠️ Cost waterfall chart data (from article, hardcoded)

### 3. llm-pipeline
- ⚠️ 3-stage pipeline diagram (CSS/SVG)
- ⚠️ Multi-market results data (from article, hardcoded)
- ⚠️ Cost trajectory data (from article, hardcoded)
- ⚠️ GPT-4o-1120 criteria list (from article)

### 4. model-arch
- ⚠️ Architecture flow diagram (CSS/SVG recreation of wiki's RankerOverview)
- ⚠️ TriBert multi-task diagram (CSS/SVG)
- ⚠️ Feature breakdown table (from article)

### 5. robust
- ⚠️ Coverage percentage data (from article, hardcoded)
- ⚠️ Zero vector mixing visualization (CSS/JS)
- ⚠️ WoodBlock coverage note (CSS text)

### 6. global-ship
- ⚠️ World map SVG with country highlights (simplified SVG)
- ⚠️ Per-language results data (from article, hardcoded)
- ⚠️ Teacher config detail (from article)

### 7. ubermarket
- ⚠️ ConditionalCall code block (from article, styled as terminal)
- ⚠️ CPU usage visualization (CSS bar)

### 8. daily-ops
- ⚠️ Daily pipeline flow diagram (CSS/SVG)
- ⚠️ Data paths panel (from article)
- ⚠️ Monitoring stack icons (CSS-drawn)
