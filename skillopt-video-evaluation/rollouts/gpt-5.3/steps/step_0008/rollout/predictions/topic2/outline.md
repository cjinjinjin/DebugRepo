# Video Outline

> **Theme**: `blueprint`（default chosen）— engineering blueprint visual language suited for architecture and deployment workflows.
> **Total Duration**: ~5 min 10 sec
> **Chapters**: 4 chapters / 24 steps

---

## 1. flow-overview — Deployment spine and failure surface（6 steps · ~72s）

**Information Pool**:
- v5.5 merges Gemma4, ZImage, ChangXu v2, Hao docs, and team insights — source article title / L1-L4.
- Deployment main flow: local test → upload/migrate/build parallel → Polaris → DLIS deploy → verification — source article §2 / L52-L63.
- Key principle: integrate Kusto logging during local phase — source article §2 / L58-L63.

**Development Plan**:
- step 1 (~12s) — Cold-open risk statement with one high-impact hero panel.
- step 2 (~12s) — Full deployment spine map with five gates.
- step 3 (~12s) — Highlight parallel lane split for data and image.
- step 4 (~12s) — Show source-provenance chips for v5.5 consolidation.
- step 5 (~12s) — Emphasize local-first rule in large warning card.
- step 6 (~12s) — Chapter close: readiness threshold before pipeline advance.

Voiceover excerpt:
> Keep the five-gate structure in your head. Local validation is the cost-control point.

---

## 2. local-data-image — Local validation and parallel prep（6 steps · ~78s）

**Information Pool**:
- Core file map: model.py, dlis_inter.py, http_server.py, requirements-vllm.txt — source article §3.2 / L77-L84.
- Docker options: Dockerfile_vllm_fast (<1s) vs full pinned stack (tens of minutes) — source article §3.4 / L99-L104.
- Local pitfalls: GPU selection, port mapping 8888, stale cache, Kusto verification — source article §3.5 / L114-L133.
- Gen1 flat layout and root-file constraints for dlis_inter.py and certs — source article §4.2 / L154-L162.
- Gen2 is serving source of truth; migration required — source article §5 / L168-L199.
- CI auto-build timing and tag format — source article §6.2 / L214-L220.

**Development Plan**:
- step 1 (~13s) — Personal-branch setup and scope framing.
- step 2 (~13s) — Core file map visual.
- step 3 (~13s) — Docker strategy comparison board.
- step 4 (~13s) — Local run loop sequence (build/run/request/logs).
- step 5 (~13s) — Data lane: Gen1 layout plus Gen1→Gen2 migration callout.
- step 6 (~13s) — Image lane: CI trigger, tag output, merge condition with data lane.

Voiceover excerpt:
> Do local correctness and observability first, then run data and image lanes in parallel.

---

## 3. polaris-production — Polaris quality gate to production cutover（6 steps · ~78s）

**Information Pool**:
- Polaris config fields: ModelPath, ModelDataPath, env vars, WaitingModelReadyInMin — source article §7.1 / L239-L245.
- Success indicator: Instance loading 100% + success — source article §7.2 / L254-L256.
- Verification dimensions: output, latency, resource usage, stability — source article §7.3 / L258-L264.
- Hardware guidance by workload profile — source article §8.1 / L280-L287.
- Portal sequence and page-specific config scope — source article §8.2 / L304-L314.
- ACL misconfiguration leads to 403 — source article §8.3 / L316-L321.
- Endpoint and route formatting pitfalls — source article §9.2 / L335-L347.

**Development Plan**:
- step 1 (~13s) — Polaris gate intro and pass/fail framing.
- step 2 (~13s) — High-risk configuration matrix.
- step 3 (~13s) — 100% loading pass visual and test-dimension board.
- step 4 (~13s) — Loop-back path: fix → rebuild → retest.
- step 5 (~13s) — Production cutover pages and hardware decision panel.
- step 6 (~13s) — ACL and endpoint verification checklist before go-live.

Voiceover excerpt:
> Polaris is a quality gate, not a ritual. Production failures are usually configuration failures.

---

## 4. ops-hardening — Environment separation and observability reliability（6 steps · ~72s）

**Information Pool**:
- SI/Prod endpoint separation is mandatory; shared endpoint is blocker — source article §11.1 / L373-L375.
- Cert types and expiry discipline, including .pfx updates — source article §11.2-11.3 / L378-L393.
- Cert context determines Kusto environment visibility — source article §12.1 / L397-L403.
- Common logging failures: swallowed EventHub exceptions, record.msg misuse, crash flush gaps — source article §12.5 / L441-L463.
- Additional operational issue matrix in common issues section — source article §13 / L507-L568.

**Development Plan**:
- step 1 (~12s) — SI vs Prod separation board.
- step 2 (~12s) — Certificate lifecycle timeline with expiry alert.
- step 3 (~12s) — Kusto routing diagram by cert and namespace.
- step 4 (~12s) — Logging anti-pattern and corrected pattern board.
- step 5 (~12s) — Common issue heatmap from deployment incidents.
- step 6 (~12s) — Final deployment readiness checklist and sign-off frame.

Voiceover excerpt:
> If logging and environment boundaries are weak, your launch is blind under pressure.

---

## Asset Checklist

### 1. flow-overview
- ✓ Source facts from `article.md`
- ⚠️ Real internal screenshots not embedded; placeholders used

### 2. local-data-image
- ✓ Commands, paths, and timing facts from `article.md`
- ⚠️ Optional ADO pipeline screenshots can be added later

### 3. polaris-production
- ✓ Polaris field names and verification dimensions from `article.md`
- ⚠️ Optional portal screenshots can be added later

### 4. ops-hardening
- ✓ Cert/Kusto issue patterns from `article.md`
- ⚠️ Optional Kusto dashboard screenshot can be added later
