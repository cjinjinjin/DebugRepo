# Script — Loop Engineering (topic4)

## Chapter 1 — The Shift: Prompting → System Design
Loop engineering means you stop being the person who writes one-off prompts.
You design a repeatable system that prompts agents for you.
A loop has a goal, a cadence, verification, and memory.
The key shift: your leverage comes from control design, not prompt phrasing.

## Chapter 2 — Why Teams Care Now
Leaders from agent ecosystems describe the same pattern: loops beat ad-hoc prompting.
The repository frames this as a practical operating model for real software teams.
It targets tools like Grok, Claude Code, Codex, and Cursor.
This is not theory; it is about shipping work repeatedly with guardrails.

## Chapter 3 — The Core Primitives
A production loop combines scheduling, worktrees, skills, connectors, and sub-agents.
Scheduling drives cadence.
Worktrees isolate concurrent changes safely.
Skills preserve project context.
Connectors reach real systems.
Sub-agents split maker and checker responsibilities.
Memory keeps state durable across runs.

## Chapter 4 — Pattern Library as Execution Recipes
The repo publishes concrete patterns such as daily triage, PR babysitter, and CI sweeper.
Each pattern maps cadence, risk, and cost.
You start from low-risk report-only behavior, then increase autonomy.
Pattern choice is a governance decision, not just an engineering choice.

## Chapter 5 — Tooling and Operational Readiness
Three utilities form an operating loop: init, cost estimation, and audit.
Init scaffolds loop structure.
Cost estimates token exposure before rollout.
Audit scores readiness and identifies missing controls.
This converts AI workflow design into measurable engineering practice.

## Chapter 6 — Safety, Failure Modes, and Human Gates
The repo emphasizes safety docs, anti-patterns, and explicit escalation points.
Unattended loops can amplify mistakes, so verification remains mandatory.
Human gates are required for risky or ambiguous actions.
The message is clear: automate aggressively, but keep accountability human.

## Chapter 7 — Practical Adoption Path
Start with one narrow loop.
Measure outcomes and incidents.
Move from L1 report-only to L2 assisted fixes, then L3 unattended only when justified.
Loop engineering is iterative systems design, not instant autonomy.

## Closing
Loop engineering is an engineering discipline for agent orchestration.
The value is durability: repeatable execution, safer scale, and better decision loops.
