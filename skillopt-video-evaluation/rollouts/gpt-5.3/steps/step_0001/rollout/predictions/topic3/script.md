You ever wonder why some ads show up with a bunch of extra links and callouts, while others are just... bare?

---

That's decoration relevance. We look at those extras — sitelinks, callouts, structured snippets, price extensions — and ask: does this actually match what you were searching for?

---

If it doesn't match, it shouldn't be there. A bad decoration is worse than no decoration at all.

---

We split decorations into four sub-tasks. Task 0 handles Sitelinks and SLAB. The others cover everything from SmartCategory and StructuredSnippets to Callouts and Price Extensions.

---

And we built one unified model for all of them. Same architecture, different classification heads per task.

---

How do we judge quality? There's a 4-level labeling scale. Excellent means the decoration matches the query directly. Good means it's relevant but not specific. Fair means it's weakly related — like a competitor product. Bad means it shouldn't be there at all.

---

In production, we collapse it down. Excellent, Good, and Fair all count as NonBad. Only Bad gets filtered out.

---

Now, where do these labels come from? For six years — 2017 through 2023 — we used managed human labeling. Trained judges scoring every item.

---

It worked, but it was slow. And the cost? $0.238 per item.

---

In late 2023, we switched to LLM labeling. We started with DV3, moved to GPT-4 Turbo, then GPT-4o.

---

GPT-4o gets us about 83% accuracy. And the cost dropped to $0.0072 per item — that's roughly 27 times cheaper than human judges.

---

For context, GPT-4o pricing is $2.50 per million input tokens and $7.50 per million output. Compared to DV3's $0.03 per thousand tokens, that's a massive drop.

---

The labeling pipeline itself has three stages. Stage 1: understand the query — extract intent, entity, category. Stage 2: evaluate each decoration's relation to the query. Stage 3: make the final Bad or NonBad call.

---

We went through multiple generations of this pipeline. DV3 in 2023 — 13 iterations just on the US market. The starting accuracy? 53%. Basically a coin flip.

---

The biggest breakthrough was one prompt change. We told the model: evaluate query-decoration relevance. Not ad-query relevance. Not decoration-ad relevance. Just query to decoration. That jumped accuracy from 53% to nearly 70%.

---

After cleaning up training data, the finalized DV3 pipeline hit 85.64% overall accuracy in the US. Multi-market results looked solid too — Germany at 81.50%, France at 84.79%. But Japanese was tough at 61.96%, and Portuguese only hit 63.27%.

---

Then GPT-4 Turbo arrived in early 2024. We tried applying the same DV3 prompts directly. Result? 42% accuracy. Worse than random guessing.

---

The problem was format. We had to switch to ChatML format with proper start and end tokens. Adjust max token limits — 200 for decoration, 100 for crossing. Tune frequency and presence penalties. 18 experiments later, we recovered to 86.90%.

---

The flight verification was interesting. DV3 vs GPT-4 Turbo showed some nuance. DCO decorations got significantly better — 39% lower defect rate. But HSL regressed by 17.77%. Trade-offs everywhere, all measured with p-values under 0.05.

---

Mid-2024, GPT-4o came in. 25 experiments across 7 markets. The key discovery: make the model explain its reasoning before it scores. That "explanation before judgment" approach pushed accuracy from 73% to about 84%.

---

Short explanations worked better than long ones — experiment #6 at 84.53% beat experiment #5 at 83.30%. Concise reasoning, then score.

---

Adding query understanding with location and language info gave us the best balanced accuracy — 74.31% in experiment #22. But the overall accuracy difference between #22 and #25 was small: 83.38% vs 83.03%.

---

The cost trajectory tells the whole story. DV3: $0.184 per item. GPT-4 Turbo: $0.019. GPT-4o: $0.0072. That's a 25x reduction in 18 months.

---

We're now evaluating GPT-4o-1120. Ten criteria for migration: ground truth data, parallel runs, agreement rates across multiple runs, token usage optimization, DSAT capture, sensitive segment analysis, end-to-end SLA, six months of auditing, native speaker sign-off on prompts, and case studies on failure patterns.

---

OK. How does the actual model work?

---

We've got a teacher-student pipeline. The teacher is CULR V4 — Cross-lingual Universal Language Representation, version 4. It generates soft labels. Then TriBert V9, the student, learns from those and runs in production.

---

The teacher evolved over time. CLR V2 was the US-only baseline. CLR V3 improved on it. CULR V3 went cross-lingual. CULR V4 is the current best — strongest global performance across all tasks.

---

TriBert is a multi-task model. Four tasks share the same BERT backbone, but each has its own classification head. So sitelinks, callouts, snippets — same encoder, scored differently.

---

The feature set is rich. Metastream features give query, ad, and decoration signals. RC2 features handle contextual relevance — things like QKRelevance score and QACDefect score. WoodBlock does lightweight feature extraction. TriBert ties it all together.

---

Query features come from QAS — health, navigational, adult, image, commerce, finance, video signals. About two dozen categories total.

---

Training happens in two stages. Stage 1: finetune on managed human labels from years of judge data. Stage 2: finetune on LLM-generated labels. We found two-stage beats either source alone.

---

The V9 experiments proved it clearly. The CLR V2 baseline scored 0.8527 on US test. CULR V4 with two-stage training? 0.9541. Starting from a finetuned checkpoint instead of a pretrained one — consistently better.

---

Now, the production TriBert has some gaps. Query vectors return all zeros sometimes in livesite. Item vector coverage varies — Sitelinks hit 99%, but FourthLine is only at 80%. SmartCategory is at 90%. Ad vector coverage sits at 93%.

---

So we do robust training. We mix in zero vectors during training — controlled amounts — so the model learns to handle missing features gracefully. Two signals track this: IsBertValid and IsWBValid.

---

The tradeoff is intentional. You accept a small AUC drop on normal validation in exchange for much better livesite AUC when features are missing. Pick the right zero vector ratio, and both sides stay acceptable.

---

WoodBlock deserves a note. Right now we only have a WoodBlock trained on Sitelink and SLAB data — query-decoration relevance pairs. For other decoration types, it just returns a default value of 0.

---

Calibration matters too. Extensions and annotations share the same display position as sitelinks. So filtering thresholds need to be tuned per task, not per individual decoration type.

---

Now the big move — going global.

---

Before, we had a US-only CLR teacher. International markets were basically getting a model that didn't understand their language.

---

CULR V4 fixed that. Cross-lingual, multilingual, trained on data from German, French, Chinese, Japanese, Portuguese, Danish.

---

The best teacher config used query + web4Ads + QU1 as text_a, and adcopy + decorationText as text_b. Learning rate 5e-6, gradient accumulation steps 8, one epoch, max sequence length 256.

---

The improvement on international markets was dramatic. INTL BSCAP on the test set went from 0.617 to 0.934. Goldset went from 0.533 to 0.757. Not incremental — a completely different league.

---

Per-language breakdown: German at 0.88, French at 0.90, Simplified Chinese leading at 0.91. Even Japanese, which is notoriously hard for cross-lingual models, reached 0.84.

---

We ran ablation experiments. web4Ads gave marginal gains on goldset, minimal on test. QU1 helped some splits but not others. EEM Rewrites for data augmentation gave a solid INTL boost — QBSCAPTerm goldset jumped from 0.589 to 0.626.

---

UberMarket migration was the other big project. We unified US and INTL onto the same infrastructure. Finished UMV1 on PaidSearch in North America by July 2024.

---

The challenge was latency. Enabling relevance models for all decorations across all traffic groups would blow up CPU usage.

---

The fix: ConditionalCall. The Queene model only serves Bing O&O traffic groups — MarketplaceClassifications 1 and 2, RelatedToAccountId 1004. Everyone else gets a default zero vector. NA runs on Queene69, INTL on Queene13. CPU increase? Under 1%. No extra machines.

---

What runs every day? MetaStream schedules handle three things: daily new impression annotation, combining 180 days of history with extensions, and SLAB plus Image Extension streams.

---

Document vectors and ads vectors refresh on their own schedules. Vector publishing goes xlite first, then prod — and bin file generation is limited to once daily.

---

Check-in process for new model versions follows a specific path. You create a PR in the Rnr-Offline-ClickPrediction repo, follow the deployment OneNote steps, and use the FPS publish portal for vector publishing.

---

The monitoring stack covers multiple layers. PowerBI dashboards for live metrics. Aether pipelines for offline monitoring. Cosmos SStream monitors for both Gene vector publish and TriBert decoration relevance vectors.

---

That's TA Decoration Relevance. A unified model, four task types, teacher-student pipeline, LLM labeling that dropped costs 27x, robust training for livesite, and now it works globally. If you're onboarding to this system, start with the DRI handbook — it'll walk you through the operational details.
