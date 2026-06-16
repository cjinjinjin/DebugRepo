# Rollout Data Contract for VideoSkillOpt

## What SkillOpt expects from a rollout

In SkillOpt, a benchmark connects through `EnvAdapter.rollout(env_manager, skill_content, out_dir, **kwargs)`.

The rollout must do two things:

1. Run the target agent with the current `skill_content`.
2. Persist enough evidence under `out_dir` so reflection and later debugging can understand what happened.

The return value is a list of result rows. Each row must include:

```json
{
  "id": "topic1",
  "hard": 0,
  "soft": 0.72
}
```

Recommended common fields:

```json
{
  "id": "topic1",
  "hard": 0,
  "soft": 0.72,
  "n_turns": 12,
  "fail_reason": "pacing_gate_failed: total_duration_exceeds_target_by_43_percent",
  "task_type": "web_video_presentation",
  "task_description": "Generate a web video presentation from a public technical blog",
  "predicted_answer": "",
  "question": "",
  "reference_text": "",
  "target_system_prompt": "",
  "target_user_prompt": "",
  "extras": {
    "source_type": "public_blog",
    "language": "en",
    "target_duration_min": 6
  }
}
```

For VideoSkillOpt, `hard` should represent hard-gate pass/fail. `soft` should represent the normalized weighted scorecard.

## What SkillOpt saves around each step

The main trainer persists:

```text
outputs/<run_name>/
├── config.json
├── history.json
├── runtime_state.json
├── best_skill.md
├── skills/
│   └── skill_v0001.md
├── steps/
│   └── step_0001/
│       ├── rollout/
│       ├── patches/
│       ├── merged_patch.json
│       ├── ranked_edits.json
│       ├── candidate_skill.md
│       ├── edit_apply_report.json
│       ├── selection_eval/
│       ├── trajectory_digest.json
│       └── step_record.json
├── slow_update/
└── meta_skill/
```

If `accumulation > 1`, rollout and patches are nested under `steps/step_XXXX/batch_N/`.

The built-in rollout convention is:

```text
rollout/
├── results.jsonl
└── predictions/
    └── <task-id>/
        ├── conversation.json
        ├── target_system_prompt.txt
        └── target_user_prompt.txt
```

The default reflect stage reads `predictions/<task-id>/conversation.json`, plus result-row fields such as `fail_reason`, `task_description`, `task_type`, and optional hidden reference text.

## Video rollout artifacts to save

For `web-video-presentation`, each task rollout should save both the normal SkillOpt files and video-specific artifacts.

```text
rollout/
├── results.jsonl
├── scorecard.json
├── hard_gate.json
├── pairwise_preference.json             # only when comparing two skills
├── predictions/
│   └── <topic-id>/
│       ├── conversation.json
│       ├── target_system_prompt.txt
│       ├── target_user_prompt.txt
│       ├── source.json
│       ├── article.md
│       ├── current_skill.md
│       ├── script.md
│       ├── outline.md
│       ├── review-log.md
│       ├── audio-segments.json
│       ├── build-log.txt
│       ├── typecheck-log.txt
│       ├── evaluator-report.json
│       ├── generated-file-manifest.json
│       └── presentation/                # optional full copy or path reference
```

Recommended `conversation.json` shape:

```json
[
  {
    "role": "system",
    "content": "Skill content and rollout instructions"
  },
  {
    "role": "user",
    "content": "Topic metadata, source material, target duration, theme, quality constraints"
  },
  {
    "type": "tool_call",
    "cmd": "materialize source to article.md",
    "obs": "success, article has 3120 words"
  },
  {
    "type": "tool_call",
    "cmd": "generate script.md and outline.md",
    "obs": "8 chapters, 47 narration steps"
  },
  {
    "type": "tool_call",
    "cmd": "build presentation and run checks",
    "obs": "typecheck passed; audio total 703.1s; 19 segments >=15s"
  },
  {
    "role": "system",
    "content": "[EVALUATION RESULT]\nhard=0\nsoft=0.62\nfail_reason=pacing_gate_failed..."
  }
]
```

The conversation should be compact but evidence-rich. It does not need every file's full content if those files are saved separately and referenced in the observation.

## Minimum evaluator output

`evaluator-report.json` should include:

```json
{
  "hard_gate": "fail",
  "blocking_issues": [
    {
      "gate": "target_duration",
      "severity": "blocking",
      "evidence": "target=450s actual=703.1s deviation=56.2%"
    }
  ],
  "scores": {
    "content_fidelity": 4.5,
    "pacing": 2.0,
    "visual_semantics": 4.0,
    "non_ppt_feel": 4.0,
    "audio_sync": 3.0,
    "theme_fit": 4.2,
    "engineering": 5.0
  },
  "weighted_score": 0.62,
  "fail_reason": "pacing_gate_failed: actual duration exceeds target by more than 20%"
}
```

Mapping to SkillOpt result row:

```json
{
  "id": "topic3",
  "hard": 0,
  "soft": 0.62,
  "fail_reason": "pacing_gate_failed: actual duration exceeds target by more than 20%",
  "task_type": "azure_devops_wiki",
  "task_description": "TA Decoration Relevance video generation"
}
```

## Can Copilot's GPT model be used for rollout collection?

Short answer: useful for collection, but not directly plug-and-play as a SkillOpt `model` backend.

SkillOpt's public model router supports:

- `openai_chat` / Azure OpenAI-compatible chat
- `claude_chat`
- `qwen_chat`
- `minimax_chat`
- `codex_exec`
- `claude_code_exec`

There is no built-in `copilot_chat` target backend in the main SkillOpt training loop. The `plugins/copilot` area belongs to SkillOpt-Sleep style integrations, not the core benchmark trainer interface.

Practical options:

1. **Use Copilot CLI as an external rollout harness.** Let Copilot generate the video project, then write the expected rollout files (`results.jsonl`, `predictions/<id>/conversation.json`, evaluator reports). This is the fastest way to collect trajectories.
2. **Use SkillOpt's trainer only after trajectories are structured.** Implement a `web_video_presentation` adapter whose rollout calls a local script or previously materialized Copilot run outputs.
3. **Add a custom SkillOpt backend only if needed.** A real `copilot_chat` or `copilot_exec` backend would require a stable non-interactive API/CLI contract that can return model messages and tool traces. If that contract is unavailable, an external harness is safer.

Recommended first implementation: use Copilot to produce rollouts and a local evaluator to normalize them into the SkillOpt artifact layout. Then use SkillOpt-style reflection over the saved trajectories.
