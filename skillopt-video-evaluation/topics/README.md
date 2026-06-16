# Topic Split for VideoSkillOpt

This folder defines the initial topic set for testing a SkillOpt-style loop on the `web-video-presentation` skill.

## Split policy

The first split is intentionally small and source-diverse:

| Split | Topics | Purpose |
|---|---:|---|
| `train` | 4 | Discover recurring skill failures and generate candidate skill edits |
| `val` | 2 | Decide whether candidate skill edits should be accepted |
| `test` | 2 | Final held-out check after a best skill is selected |

## Source diversity

| Source type | Topics |
|---|---|
| Public technical blog | `topic1` |
| SharePoint / private Word doc | `topic2` |
| Azure DevOps Wiki / internal wiki | `topic3` |
| GitHub repository | `topic4` |
| WeChat article | `topic5`, `topic6`, `topic7`, `topic8` |

## Files

| File | Purpose |
|---|---|
| `all-topics.json` | Full topic registry with URLs, source type, access notes, and expected format challenges |
| `train/items.json` | Training split |
| `val/items.json` | Validation split |
| `test/items.json` | Test split |

## Notes

- These files only register topics and split assignments. They do not cache article content.
- Private sources may require authenticated browser/session access before rollout.
- WeChat pages may require manual capture if automated fetch is blocked.
- Each rollout should materialize the source into an `article.md` before invoking `web-video-presentation`.
