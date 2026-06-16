# Lessons from Building Claude Code: Prompt Caching Is Everything

Source: https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything

## Core thesis
Prompt caching is the cost-and-latency foundation for long-running agent sessions. At Claude Code scale, cache hit rate is treated like an SRE metric with alerts and incidents.

## Key lessons from the article

### 1) Prompt layout determines cache hit rate
Prompt caching is prefix-based. Put stable content first and volatile content last.

Recommended order:
1. Static system prompt + tools
2. Project-level context (e.g., CLAUDE.md)
3. Session context
4. Conversation messages

### 2) Prefer message-level updates over rewriting system prompt
If time, file state, or mode changes, inject updates in new messages (e.g., system reminder tags) instead of editing cached prefix.

### 3) Avoid model switches mid-session
Caches are model-specific. Switching from a large model to a smaller model midstream can cost more if it forces rebuilding a huge prefix. Use subagents + handoff when switching is necessary.

### 4) Never add/remove tools in the middle of a session
Tool schema changes break cache prefixes. Represent mode changes as tool calls (e.g., EnterPlanMode / ExitPlanMode) while keeping tool definitions stable.

### 5) Defer tool loading instead of pruning tools
Use lightweight deferred tool stubs (`defer_loading`) and resolve full schemas only when selected via tool search.

### 6) Compaction must be cache-safe
Compaction/summarization calls should reuse the exact parent prefix (same system prompt, tool set, context) and append compaction instruction as a new message. Otherwise compaction can become uncached and expensive.

### 7) Operational mindset
Monitor cache hit rate like uptime. Small cache miss changes can have outsized impact on cost and latency.

## Closing takeaway
Design agent architecture around prefix stability from day one. Prompt structure, tool strategy, model routing, and compaction design all need to preserve cache reuse.
