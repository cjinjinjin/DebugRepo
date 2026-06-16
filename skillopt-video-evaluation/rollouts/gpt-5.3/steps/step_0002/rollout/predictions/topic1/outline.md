# Video Outline

> **Theme**: `blueprint` — technical system diagrams and process clarity  
> **Total Duration**: ~6 min (~34 steps)  
> **Chapters**: 7 chapters / 34 steps

---

## 1. hook — Caching Is Product Infrastructure (4 steps · ~35s)

**Info Pool**:
- "cache rules everything" framing
- Claude Code treats cache hit drops as incident-level events
- Goal: lower latency + lower cost for long sessions

**Dev Plan**:
- step 1 — Cold open: "Prompt caching is everything"
- step 2 — Contrast panel: usable agent product vs expensive demo
- step 3 — Ops dashboard visual: cache hit rate alerting
- step 4 — Bridge line: this is an architecture constraint

---

## 2. prefix-law — Prefix Matching Mental Model (5 steps · ~50s)

**Info Pool**:
- Caching is prefix matching from request start to cache breakpoints
- Stable-first, dynamic-last ordering is mandatory
- Claude Code ordering: system+tools, project context, session context, conversation

**Dev Plan**:
- step 1 — Explain "prefix match" with horizontal token strip
- step 2 — Good ordering timeline
- step 3 — Shared-prefix across sessions visual
- step 4 — Bad ordering visual that invalidates cache
- step 5 — Rule card: static first, dynamic last

---

## 3. fragility — Easy Ways Teams Break Cache (5 steps · ~50s)

**Info Pool**:
- In-depth timestamp in static prompt can break cache
- Non-deterministic tool order can break cache
- Tool parameter drift can break cache

**Dev Plan**:
- step 1 — "Looks harmless" headline
- step 2 — Timestamp injected into static prefix
- step 3 — Tool list reorder diff view
- step 4 — Tool schema mutation example
- step 5 — Guardrail checklist

---

## 4. state-and-tools — Keep Prefix Stable While Behavior Changes (6 steps · ~65s)

**Info Pool**:
- Prefer message updates over editing system prompt
- Do not add/remove tools mid-session
- Plan Mode via EnterPlanMode/ExitPlanMode tools
- defer_loading stubs + tool search pattern

**Dev Plan**:
- step 1 — Message update pattern diagram
- step 2 — System prompt rewrite = miss warning
- step 3 — Tool set churn anti-pattern
- step 4 — Plan mode modeled as tool-mediated state
- step 5 — Deferred tool stubs flow
- step 6 — Result: stable prefix, dynamic capability

---

## 5. model-routing — Why Mid-Session Model Switch Can Cost More (4 steps · ~45s)

**Info Pool**:
- Prompt cache is model-specific
- Switching Opus→Haiku deep in session may rebuild huge cache
- Subagent handoff is safer

**Dev Plan**:
- step 1 — Model-specific cache boundary visual
- step 2 — Cost paradox chart for late switch
- step 3 — Subagent handoff pattern
- step 4 — Routing rule of thumb card

---

## 6. compaction — Cache-Safe Forking for Summaries (6 steps · ~70s)

**Info Pool**:
- Compaction needs full conversation context
- Separate summarizer prompt+tools loses cache and spikes cost
- Reuse exact parent prefix; append compaction instruction
- Reserve compaction buffer for prompt + output tokens

**Dev Plan**:
- step 1 — Context window full event
- step 2 — Naive compaction uncached-cost trap
- step 3 — Prefix divergence at first token
- step 4 — Cache-safe fork architecture
- step 5 — Buffer budgeting strip
- step 6 — Cost outcome comparison

---

## 7. operating-model — Five Rules to Run By (4 steps · ~45s)

**Info Pool**:
1) Prefix match is law  
2) Use messages for updates  
3) Do not change tools/models mid-conversation  
4) Monitor hit rate like uptime  
5) Fork ops with shared prefix

**Dev Plan**:
- step 1 — Five-rule board
- step 2 — Cost/latency impact summary
- step 3 — Implementation starter checklist
- step 4 — Closing line: design around prefix stability from day one

---

## Assets Needed
- Simple system diagrams (prefix strip, cache boundary, fork tree)
- Lightweight dashboard mock for cache hit SLO
- Icon set for tools/models/messages
- Placeholder charts for cost and latency comparisons
