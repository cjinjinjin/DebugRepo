Prompt caching sounds like infrastructure trivia. It isn't.

---

For long-running agents, it's the difference between a usable product and an expensive demo.

---

At Claude Code, cache hit rate is monitored like uptime. If cache efficiency drops, that's incident-level serious.

---

Why? Because prompt caching is prefix matching.

---

The model reuses computation only when the beginning of the request stays byte-for-byte stable.

---

So prompt order becomes architecture.

---

Put static first. Put dynamic last.

---

A practical layout looks like this: static system prompt and tools first, then project context, then session context, and only then conversation messages.

---

That way, many sessions share the same expensive prefix.

---

But this is fragile.

---

A timestamp in the wrong place can break cache.

---

Non-deterministic tool ordering can break cache.

---

Changing tool parameters mid-session can break cache.

---

When state changes, resist rewriting the system prompt.

---

Instead, send updates as new messages in the next turn.

---

In Claude Code terms, that can be a system-reminder style message attached to the next user or tool result.

---

Model switching has another trap.

---

If you are deep into a long Opus conversation, jumping to Haiku for one "easy" question may cost more, because Haiku has no cache for that giant prefix.

---

If you must switch, use a subagent handoff: let one model summarize and pass compact context to another.

---

Tools follow the same rule.

---

Adding or removing tools mid-session invalidates cached prefixes.

---

So Plan Mode is implemented as tools and instructions, not by swapping tool sets.

---

The tool catalog stays stable, while behavior changes through EnterPlanMode and ExitPlanMode.

---

For large MCP catalogs, defer loading instead of deleting tools.

---

Ship lightweight stubs in a fixed order, then resolve full schemas only when selected.

---

Compaction is where teams often lose money.

---

If summarization runs with a different system prompt and no tools, cache is lost from token one.

---

Claude Code avoids this by cache-safe forking.

---

Compaction calls reuse the parent prefix exactly, then append a compaction request at the end.

---

Only the new tail is uncached.

---

So the bigger lesson is simple.

---

Prompt caching is not a tuning trick. It is a product design constraint.

---

Design prompts, tools, model routing, and compaction around prefix stability from day one.

---

Do that, and you buy lower latency, lower cost, and much more generous agent experiences.
