[[_TOC_]]

# Background
The Decoration Relevance Task aims to evaluate the quality of the decorations in terms of their relevance to user intent with the context of the base Ad. To achieve this goal, we have split the task into four sub-tasks based on the decoration content and display position. We have built a unified model for all decorations to filter out low-quality decorations. 

| Task| SL(Sitelink)| FL(FourthLine)-Deprecated| BSCAP(Below Snippet Caption)|BSCAPTerm (Below Snippet Caption Term)|
| --- | --- | --- | --- | --- |
| TaskId | 0 | 1 | 2 | 3 |
|Decorations| SiteLink, SLAB| LongDescription, DynamicDescription, CoCampaignDesc, SmartLandingPageTitle| SmartCategory, StructuredSnippetExtension, SmartLandingPageList, FilterLink, AutomatedFilterLink | CalloutExtension,DynamicCalloutAnnotation, PriceExtension, DynamicProductExtensio|
|Model structure|FR|FR|FR|FR|
|Feature set| Metastream features, RC2 feature, WoodBloack, TriBert|Basic features, Metastream features, WoodBloack,TriBert|Metastream features, RC2 feature, WoodBloack,TriBert|Metastream features, RC2 feature, WoodBloack,TriBert|

### Labeling Scale (4-level → binary)

*   **Excellent (E)**: Decoration matches the query or is a more specific instance
*   **Good (G)**: Relevant but not specific enough; user may still have interest
*   **Fair (F)**: Weakly relevant (competitive/alternative/complementary products)
*   **Bad (B)**: Should not be displayed — off-intent or meaningless (e.g., non-clickable "See More")
In production: E/G/F → **NonBad**, B → **Bad**

We have finalized a new unified global model for North American and other international markets. Will start the flight soon!

# Label Data:
## Managed labeling: 2017.9 – 2023.12
- Labeling guideline: [Ads Decorations_Labeling Guideline v6.9(DPE).docx](/.attachments/Ads%20Decorations_Labeling%20Guideline%20v6.9(DPE)-1e3149f2-1f19-44f6-b05a-3b12416d4662.docx)

## LLM labeling: 2023.12 - now 
[Decoration Relevance GPT-4O Labeling Migration.pptx](/.attachments/Decoration%20Relevance%20GPT-4O%20Labeling%20Migration-5ca7733e-93b4-4a37-baf1-435b3c09b61c.pptx)
- We have a shared LLM Labeling prompt for US and other international markets.
- Prompt(DV3,GPT4-Turbo,GPT-4o): https://msasg.visualstudio.com/Bing_and_IPG/_git/AdPrompt?path=/offline/relevance/Decoration
* GPT-4o Accuracy: 
 ![GPT-4o accuracy resize.png](/.attachments/GPT-4o%20accuracy%20resize-4b7a219c-17ce-4957-bea7-a5ef91cc84d5.png)
* GPT-4o Cost:

| Labeling | Cost |
|--|--|
| EN-US Managed Labeling| $0.238 |
| DV3*  | $0.184 |
|GPT4-Turbo** | $0.019|
|GPT-4O***| $0.0072|

 *Base on public price from OpenAI is $0.03 for input and $0.06 for output per 1k tokens
 **Base on public price from OpenAI is $0.01 for input and $0.03 for output per 1k tokens
 ***Base on public price from OpenAI is $2.5 for input and $7.5 for output per 1M tokens

- Pipeline(POC: AlexLei): aether://experiments/b2106ac8-c15c-4a5f-9b5d-7a55a4c57f72


# Model Structure
In this chapter, we will introduce the production model, which is a unified decoration model but not a global model yet.First, we will provide a overview of the model and the features we have used.
![RankerOverviewResize.png](/.attachments/RankerOverviewResize-a127bd2c-1b00-4d41-8ec8-8c83a27605ff.png)

*   **Metastream features**: query/ad/decoration signals
*   **RC2 features**: contextual relevance features
*   **WoodBlock**: lightweight feature extraction
*   **TriBert**: Multi-task student model (shared BERT backbone + per-task classification heads)

### Teacher → Student Pipeline

    Teacher (CULR V4) → generates soft labels → Student (TriBert V9) → serves in production
    

**Teacher model evolution:**
*   CLR V2 (US-only, baseline)
*   CLR V3 (US-only, improved)
*   CULR V3 (Cross-lingual Universal Language Representation, multilingual)
*   **CULR V4** (current, best global performance)

### Two-Stage Training Strategy

    Stage 1: Finetune on managed (human) labeling data
        ↓ checkpoint
    Stage 2: Finetune on LLM labeling data
    

Key findings from V9 experiments:
*   **CULR > CLR**: CULR models consistently outperform CLR across all tasks
*   **Two-stage > one-stage**: Stage1 (managed) → Stage2 (LLM) better than LLM-only
*   **Finetune CKPT > pretrain CKPT**: Starting from finetuned checkpoint is better

### Robust Training

Mixes zero vectors to handle livesite feature failures:
*   `IsBertValid` / `IsWBValid` signals indicate whether BERT/WoodBlock features are available
*   Model trained to be robust when these features are missing at serving time

### ConditionalCall Logic

Queene model has ConditionalCall logic to limit latency — only serves for Bing O&O traffic groups:
*   NA slot: Queene69
*   INTL slot: Queene13

# Detail Features
## RC2 Metastream features
[meta stream features.docx](/.attachments/meta%20stream%20features-fa4318c6-8e01-41e8-a260-6c533c99d997.docx)
NumberOfOccurrences, NumberOfOccurrences2,Triple, QueryWordDistanceMatch, DomainMatch, WordsFound, Document(Frequency|Counts), Spans, StreamLength, MsnRankOfOccurrences, MsnRankOfOccurrences2, Tuples, WordCandidateStreamPresence, PhrasalMatch, OriginalQuery, QueryPath, BodyBlock, BM25F, TAU, SpuriousAnchor, DoublePhrase, TriplePhrase,  TriplePhrasePartialDouble, PhrasePerWord, TermWeight, AdsMatchType, IRFeature, StructuredDistance

## Query features
https://microsoft.sharepoint.com/teams/IPGSPIN/Intent/QAS%20Wiki/Home.aspx
- QAS feature: health, name, navigational, noncelebname, adult, url, recipes, image, imagev2, movietitle, BiteRestaurant, financestock, autos, book, celebrities, commercev2, EnglishQueryV2, finance, flight, mmqcssportsv2, Retail, travelguide, video

## RC2 Features
- PLF_31: QKRelevance score
- PLF_33: QACDefect score
- PLF_34: QLPDefect score

## TaskId
- Task 0: Sitelink, SLAB
- Task 1: LongDescription, DynamicDescription, CoCampaignDesc, SmartLandingPageTitle	
- Task 2: SmartCategory, StructuredSnippetExtension, SmartLandingPageList, FilterLink, AutomatedFilterLink	
- Task 3: CalloutExtension,DynamicCalloutAnnotation, PriceExtension, DynamicProductExtensio

## Tribert
[Gene Unified Decoration Relevance Model.pptx](/.attachments/Gene%20Unified%20Decoration%20Relevance%20Model-7b4d56fe-9af1-42dd-b348-d57f12d27453.pptx)
Tribert is a multi-task student model. 4 tasks share parameters of Bert part and have different classification blocks.
The production tribert model at 2024/10/21 is a English-only student model based on CLR teacher model serving at North American.
![Tribert2.png](/.attachments/Tribert2-ba527f7e-2921-49bd-b5a6-7ede78174772.png)

The Tribert has 2 parts: online serving for Query and offline serving for Bert item vector(Decoration) and Ads vector. Query vector returns all zeros when livesite and Item vector/WB, Ads vector coverage is not 100%. 
- Query vector returns all zeros when livesite 
- Item vector/WB coverage: 
SL/SLAB: 99%
FL: 80%
SS: SSExtension 99%, SmartCategory 90%
BS: CO 99%, DCO 90%
- Ad vector coverage: 93%
![robustTrainingResize.png](/.attachments/robustTrainingResize-a758bd63-bd4e-45d4-852a-f587f5d9a360.png)

So we proposed a robust ranker: adding 2 signals: IsBertValid, IsWBValid and mixing some zero value vectors in training stage to handle all situations in one ranker. We have two model validation: normal validation is to compare baseline ranker with robust ranker on test set with normal vectors. Livesite validation is to compare Robust rankers on test set with normal vectors vs all zero Queene vectors. The key point of robust training is to choose a zero vector ratio to make tradeoff between normal AUC drop and livesite AUC/AvgRelScore drop comparing ranker without robust training.
![EvaluationResize.png](/.attachments/EvaluationResize-79d7ca76-3b25-4fb4-856a-815b7523e8fb.png)

## Woodblock
Currently we only have a woodblock for Sitelink and SLAB trained by <Query, Decoration> relevance data. To support unified ranker for all decorations, we will give a default value 0 for other decorations.

## Calibration
The extensions and annotations are displayed in the same position like Sitelink and SLAB, which makes it necessary to filter them equally. Therefore, we aim to perform calibration tuning for each task, rather than for each individual decoration.

# Global Model Ship 
---------------------------

### Background

The global model migration unified US and INTL decoration relevance into a single multilingual model using CULR V4 as the teacher, replacing the US-only CLR teacher.

### Best Teacher Configuration

| Component | Configuration |
| --- | --- |
| text_a | query + web4Ads + QU1 |
| text_b | adcopy + decorationText |
| Base | CULR V4, finetune from stage1 checkpoint |
| Training | lr: 5e-6, grad_accum_steps: 8, epoch: 1, max_seq_len: 256 |

### Teacher Model Results (Best Config)

| Split | QSL | QBSCAP | QBSCAPTerm |
| --- | --- | --- | --- |
| US-test | 0.9568 | 0.9463 | 0.9515 |
| US-URA | 0.9113 | 0.9114 | 0.9208 |
| US-Goldset | 0.9019 | 0.8687 | 0.8543 |
| INTL-test | 0.9438 | 0.9353 | 0.9167 |
| INTL-Goldset | 0.7876 | 0.7570 | 0.7192 |

**Prod baseline comparison (INTL improvement):**
| Split | Baseline | CULR V4 |
| --- | --- | --- |
| INTL-test QBSCAP | 0.617 | 0.934 |
| INTL-Goldset QBSCAP | 0.533 | 0.757 |

### Per-Language Results (teacher at 250 steps)

| Language | max_roc |
| --- | --- |
| de | 0.8826 |
| fr | 0.8985 |
| zh-Hans | 0.9076 |
| zh-Hant | 0.8617 |
| ja | 0.8373 |
| pt-BR | 0.8887 |
| da | 0.8898 |

### Key Experiment Findings

1.  **web4Ads**: Marginal gains on goldset, minimal on test set
2.  **QU1**: Gives gains for some (split, task) combinations but not others
3.  **EEM Rewrites for data augmentation**: Improves INTL significantly (0.589→0.626 QBSCAPTerm goldset)
4.  **Stage1 CKPT**: Little gain over released finetune CKPT
5.  **Exchanging QU and web4Ads order**: Not helpful

### Student Model (TriBert V9) Experiments

Best student model config from V9 experiments:
| ID | Description | US Test QSL | US GoldSet QSL | Experiment |
| --- | --- | --- | --- | --- |
| #0 | CLR V2 baseline | 0.8527 | 0.9384 | d190389a |
| #2 | CULR V4 Stage1 | 0.8585 | 0.9478 | 8ac3989c |
| #5.1 | CULR V4 Stage2 (LLM) | 0.9524 | 0.9015 | e6f0573c |
| #6.1 | Two-stage (Stage1→Stage2) | 0.9541 | 0.9011 | a9f4af45 |
| #7 | Pretrain CKPT + Stage2 | 0.9492 | 0.8966 | 2b559445 |

### Training Pipeline References

*   Teacher training: `aether://experiments/d15d48ab-144e-4cb6-8431-c82089fee9ef`
*   Student training: `aether://experiments/7b3a890c` (see V9 page for full list)

### Data Paths

*   Managed labeling (before 20220416): `/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/AdsPlus/DecorationRelevance/DecorationRelevanceV9/ManagedLabel/Before20220416_withTaskId.ss`
*   LLM labeling: `/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/AdsPlus/DecorationRelevance/DecorationRelevanceV9/US_Data/`
*   GoldSet: `/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/AdsPlus/DecorationRelevance/GoldenSet/En-US_GoldSet2_1.8k/GoldSet1_2_ItemLevelData.ss`

### MS/MW check in 
  
Check in steps: [Steps to ingest MS/MW [2020/03/25]](onenote:https://microsoft.sharepoint.com/teams/RichAds/Shared%20Documents/Scrum%20Meeting/STCA%20Rich%20Ads%20OneNote/Decoration%20Quality.one#Steps%20to%20ingest%20MS/MW%20%5b2020/03/25%5d&section-id={8F89C6B1-A815-462A-84C1-FB420B6CEDEA}&page-id={C6951E77-537F-47A1-B857-344CA02839EA}&end)

1.  Slot status: [Slot status](onenote:https://microsoft.sharepoint.com/teams/RichAds/Shared%20Documents/Scrum%20Meeting/STCA%20Rich%20Ads%20OneNote/Decoration%20Quality.one#Vector%20status&section-id={8F89C6B1-A815-462A-84C1-FB420B6CEDEA}&page-id={816AEAC5-46F3-48CB-92D3-32F040FADE97}&end) tracking list
New vector FPS publish portal: [FPS Portal (fpsmanager.azurewebsites.net)](https://fpsmanager.azurewebsites.net/#/publish/detail/richadsgenecdssm_multivector_xlite)      [Dataset Detail - FPS Portal (fpsmanager.azurewebsites.net)](https://fpsmanager.azurewebsites.net/#/publish/detail/richadsgenecdssm_multivector_prod)

2.  Online publish(POS IGS team): Conifg path:[TAGENE_AMER.json - Repos](https://msasg.visualstudio.com/Defaultcollection/Bing_Ads/_git/Pipeline?path=/private/ADF/AzureFunctions/EnrichmentPipelineFunc/EnrichmentConfigFile/TAGENE/TAGENE_AMER.json)

### Model check in 
create PR like: [Pull request 5230244: check in Gene Relevance Global Model for US - Repos](https://msasg.visualstudio.com/Bing_and_IPG/_git/Rnr-Offline-ClickPrediction/pullrequest/5230244?_a=files)
More steps:    [Deployment](onenote:https://microsoft.sharepoint.com/teams/RichAds/Shared%20Documents/Scrum%20Meeting/STCA%20Rich%20Ads%20OneNote/Jinjin.one#Deployment&section-id={236633E3-450B-4A19-921C-04E3D6D19043}&page-id={B5C9F02F-2F0F-4982-8C29-A4B6E268A519}&end)  ([Web view](https://microsoft.sharepoint.com/teams/RichAds/_layouts/Doc.aspx?sourcedoc=%7bEE2932AB-C11E-4741-ADFB-154F9CFB10E3%7d&wd=target%28Jinjin.one%7C236633E3-450B-4A19-921C-04E3D6D19043%2FDeployment%7CB5C9F02F-2F0F-4982-8C29-A4B6E268A519%2F%29&wdpartid=%7b55714530-EC0B-097F-3597-DFF3B47E9D7E%7d%7b1%7d&wdsectionfileid=%7bA101C5F7-DD2C-4D35-BE04-38F9E29E36FE%7d&end))
# UberMarket Migration
[Decoration Relevance model UMV2 migration.pptx](/.attachments/Decoration%20Relevance%20model%20UMV2%20migration-44e6f0be-5736-4ef7-86a4-4fb9787b69ff.pptx)
We have finished the Ubermarket migration to UMV1 on PaidSearch in North American at 2024/07/31. 
- By migrating decoration-related parameters to the UberMarket level, we can ensure that all regions and markets are using the same infrastructure and algorithms, which will help to improve consistency and standardization across the system.

- Removing old version models will allow us to focus on higher quality candidates, which will help to improve the overall accuracy and reliability of the system. 

- Aligning model updates with decoration data source changes will help to ensure that the system is always using the most up-to-date and accurate data. This will help to improve the overall performance of the system.

## What we do?
The challenge we faced during the migration is the latency issue. Enabling relevance models for all decorations and using the same thresholds in all TGs under PaidSearch will bring up significant latency increase. 

To address the challenge. The first action is to continue to maintain the DisableGene setting for TGs that disabled Gene, and only update models for TGs that enabled Gene. The second action is to add ConditionalCall feature for Queene model to only serve Bing O&O TGs, and other TGs will use dummy Queene vector. Then the overall normalized CPU increase is less than 1%, which means that no additional machines are needed. 
```// ConditionalCall Logic:
// If (PublisherId IN (List) OR AdUnitId IN (List)) OR (NOT(MarketplaceClassification IN (List) AND RelatedToAccountId IN (List))) OR (NOT(DistributionChannelId IN (LIST)))
// Then send DefaultOutputType as response

[RequestSettings:ConditionalCall]
MarketPlaceClassifications=1;2  //O&O Core and Rest
RelatedToAccountIds=1004        //Bing
DefaultOutputType=DefaultValueVector
ExpectedOutputSize=64
DefaultOutputValue=0
```
For TGs in gray and orange the solution will update and add relevance model for all decorations and align thresholds with TG2’s. 
![UMV2Migrationresize.png](/.attachments/UMV2Migrationresize-fb15421e-7c9d-4d59-8d52-84e54a3483ea.png)

# LLM Labeling Pipeline
---------------------

### Architecture (3-Stage Pipeline)

    Stage 1: Query Understanding (QU) prompt
        → Extract query intent, entity, category
    Stage 2: Decoration Intent (DU) prompt
        → Evaluate each decoration's relation to query
    Stage 3: Crossing prompt
        → Final Bad/NonBad decision per decoration
    

### Pipeline Evolution

The LLM labeling pipeline generates training labels for the student model. It went through multiple prompt optimization iterations and model migrations:

* * *

### 5.1 ChatGPT / DV3 (Initial Version)

**Development period**: 2023
**Prompt iteration summary** (13+ iterations on US market):

| Iteration | US Accuracy | Key Change |
| --- | --- | --- |
| Baseline (human labels) | 53.48% | - |
| Iteration 2 (Prompt 3) | 53.48% | Initial structured prompt |
| Iteration 3 (Prompt 5) | 69.76% | Added: "evaluate query-decoration relationship, NOT ad-query or decoration-ad" |
| Iteration 4 (Prompt 6) | 62.97% | Added site trustworthiness definition (hurt accuracy) |
| Iteration 5 (Prompt 7) | 69.76% | Added subcategory/sibling products with price |
| Iteration 6 (Prompt 8) | 69.76% | Added examples and annotation |

**Key insight**: The most impactful prompt change was instructing GPT to evaluate only query-decoration relevance, not ad-query or decoration-ad relevance.
**3-stage pipeline development** (after cleaning up data):

| Config | Overall Acc | Good Acc | Bad Acc | Balanced Acc |
| --- | --- | --- | --- | --- |
| Crowd labeling | 86.78% | 96.85% | 23.04% | 59.94% |
| QLP QU + best Decor Intent | 83.01% | 85.65% | 66.33% | 75.99% |
| PA QU + best Decor Intent | 81.91% | 83.97% | 68.90% | 76.44% |
| **Finalized (without QU)** | **85.64%** | **91.86%** | **46.31%** | **69.08%** |
**Multi-market finalized results & experiment IDs:**

| Market | Overall Acc | Good Acc | Bad Acc | Balanced Acc | Experiment ID |
| --- | --- | --- | --- | --- | --- |
| US | 85.64% | 91.86% | 46.31% | 69.08% | 55dd0a4b |
| DE | 81.50% | 92.45% | 46.90% | 69.68% | 86a8ac41 |
| FR | 84.79% | 90.63% | 57.37% | 74.00% | 6e4368dd |
| ES | 77.82% | 83.75% | 60.52% | 72.13% | 666c1d80 |
| SV | 79.64% | 84.85% | 63.81% | 74.33% | f026234f |
| JA | 61.96% | 81.46% | 49.60% | 65.54% | fe494088 |
| PT | 63.27% | 81.74% | 27.11% | 54.42% | 489a094d |

**Cost (DV3)**: ~$0.194 per item ($0.03/1K input, $0.06/1K output)

* * *

### 5.2 GPT-4 Turbo Migration

**Date**: Early 2024
**Challenge**: Directly applying DV3 prompts to GPT4-Turbo gave only 41.96% accuracy (vs 85.64% on DV3).
**Required prompt changes:**

1.  ChatML format (`<|im_start|>`, `<|im_end|>`)
2.  Stop tokens: `<|im_end|>`
3.  Token limit adjustments: max_tokens 200/100 (decoration/crossing)
4.  Frequency/presence penalty tuning

**18 experiments to optimize prompts:**

| Exp# | Key Change | Overall Acc | Good Acc | Bad Acc |
| --- | --- | --- | --- | --- |
| #0 (DV3 baseline) | - | 85.64% | 91.86% | 46.31% |
| GPT4-T raw | No changes | 41.96% | 36.75% | 74.94% |
| #3 | 200/100 tokens + special tokens | 86.90% | 98.41% | 13.88% |
| #4 | Full sentence output + 100/50 tokens | 87.03% | 98.38% | 15.25% |
| #10 | ChatML format + examples, max_sql:300 | 86.35% | 93.02% | 38.90% |
| #14 | Add stop_token for crossing | 83.58% | 88.89% | 49.89% |
**Final multi-market results (GPT4-Turbo):**

| Market | Overall Acc | DR |
| --- | --- | --- |
| US | 83.32% | 16.37% |
| DE | 81.97% | 19.64% |
| FR | 83.66% | 22.89% |
| ES | 74.75% | 25.72% |
| SV | 78.16% | 29.85% |
| JA | 65.90% | 43.10% |
| PT | 61.34% | 22.98% |

**Flight verification (Tribert V8):**
DV3 vs GPT4-Turbo flight showed:
*   Overall DR diff: +4.03% (DV3 control), +1.38% (GPT4-Turbo control) — both with p<0.05
*   **DCO** decorations: significant improvement (-39.27% DR diff, p=0.00227) under threshold tuning
*   **HSL**: regression (+17.77% DR diff, p=0.02052 for GPT4-Turbo)
*   **VSL**: improvement (-11.17% DR diff, p=0.01947 for GPT4-Turbo threshold tuning)

* * *

### 5.3 GPT-4o Migration

**Date**: Mid 2024
**25+ experiments across 7 markets.** Key experiments:

| Exp# | Key Change | US Overall | US Balanced |
| --- | --- | --- | --- |
| #1 | Same prompt as GPT4-T | 73.41% | - |
| #5 | Explanation before judgment | 83.30% | - |
| #6 | Short explanations | **84.53%** | - |
| #7 | Concise crossing + moved instructions | 83.90% | - |
| #22 | QU + Location&Language | 83.38% | **74.31%** |
| #25 | #22 - Location&Language | 83.03% | - |

**Best multi-market results (#22 and #25):**

| Market | #22 Overall | #25 Overall |
| --- | --- | --- |
| US | 83.38% | 83.03% |
| DE | 82.47% | 82.75% |
| FR | 82.28% | 81.93% |
| ES | 72.58% | 72.73% |
| SV | 76.94% | 77.35% |
| JA | 67.12% | 66.96% |
| PT | 58.58% | 59.25% |

**Cost (GPT-4o #25)**: $0.0072 per item ($2.50/1M input, $7.50/1M output) — **~27x cheaper than DV3**

* * *

### 5.4 GPT-4o-1120
**Experiment**: `0af41f51-c120-4d65-9f78-57bcf06af080`
**Migration evaluation criteria** (10 items):
1.  Ground truth data — FBS Imps, Deeplogs
2.  Parallel run: FBS Imps, Deeplog DR, comScore DR
3.  Agreement rate (label flip consistency across multiple runs)
4.  Token size — optimize token usage
5.  DSAT data — can the new prompt capture more DSATs
6.  Sensitive segments — DR change + ground truth stats
7.  E2E running time — match current FBS SLA
8.  Auditing data set — run audit datasets for 6+ months
9.  Prompt review process — require native speaker sign-off (Howard as reviewer)
10.  Case study — segments/patterns where new version is less accurate
### 5.5 Pipelines
- LLM Labeling Prompt update Repo: [Decoration - Repos](https://msasg.visualstudio.com/Bing_and_IPG/_git/AdPrompt?path=%2Foffline%2Frelevance%2FDecoration&version=GBmain&_a=contents)
- Aether pipeline: 
Cook data for labeling: [aether://experiments/b1c840cb-125e-4884-a7f5-68befe4ce651](aether://experiments/b1c840cb-125e-4884-a7f5-68befe4ce651)
GPT5 Labeling pipeline: [aether://experiments/732eaafb-26ec-410a-b786-51c0bebe350a](aether://experiments/732eaafb-26ec-410a-b786-51c0bebe350a)
Save labeling result: [aether://experiments/66917ee0-1e95-48dd-96b4-fdd453cb623e](aether://experiments/66917ee0-1e95-48dd-96b4-fdd453cb623e)

- More update details:    [Global Decoration Relevance Model chatGPT labelling](onenote:https://microsoft.sharepoint.com/teams/RichAds/Shared%20Documents/Scrum%20Meeting/STCA%20Rich%20Ads%20OneNote/Decoration%20Quality.one#Global%20Decoration%20Relevance%20Model%20chatGPT%20labelling&section-id={8F89C6B1-A815-462A-84C1-FB420B6CEDEA}&page-id={FCFC139A-D88A-488B-B563-2F0E356DAA7F}&end)  ([Web 视图](https://microsoft.sharepoint.com/teams/RichAds/_layouts/Doc.aspx?sourcedoc=%7bEE2932AB-C11E-4741-ADFB-154F9CFB10E3%7d&wd=target%28Decoration%20Quality.one%7C8F89C6B1-A815-462A-84C1-FB420B6CEDEA%2FGlobal%20Decoration%20Relevance%20Model%20chatGPT%20labelling%7CFCFC139A-D88A-488B-B563-2F0E356DAA7F%2F%29&wdpartid=%7bBEB340CF-86E2-0DAE-280C-286E6B32CB8D%7d%7b1%7d&wdsectionfileid=%7b1B870B57-AA2E-4541-8720-CB0D9F792994%7d&end))

# DRI handbook
DRI handbook: [Rich-Ads-DRI 2024.pptx](https://microsoft.sharepoint.com/:p:/t/adsplus/cQqfMbXWpH8PRI5ph_Q1GMC1EgUC1Eqz2ULrpTstGZtvaPeBoQ)

# Daily running pipeline
- MetaStream schedule: 
Daily new impression annotation: [aether://experiments/61e77843-0e30-43f4-a5e5-012bef2218bd](aether://experiments/61e77843-0e30-43f4-a5e5-012bef2218bd)
Combine history 180 days impression annotation, and extension: [aether://experiments/cbbd73f3-ae3a-477e-b0dd-601fb41549b3](aether://experiments/cbbd73f3-ae3a-477e-b0dd-601fb41549b3)
SLAB metastream and Image Extension metastream: b0ad06fc-a351-4ec0-bdd1-fcf8b01ead10
- Doc vector refresh schedule: 1c395c0f-b57a-4b7b-8fac-7207c390fda0 
- Ads vector refresh schedule: c8f1bd40-b173-4310-b699-19ae55968dc9
- Vector publish refresh schedule: 
Xlite: 08fc2378-ee8a-4941-8e9c-da816f69fa06
Prod: b0c2e142-c88a-4c61-b9e7-b8617bb6a557
Bin file generation is limited to once a day. The xlite environment output should be generated before the prod environment.

# Appendix

- Extract RC2 features monthly: aether://experiments/33c16750-487e-440f-95ad-4875949922a0
- Save URA RGUID monthly:aether://experiments/fb3f0a8e-8946-430b-b7ed-ddb0a97a1d5f
-  Raw FBS data path:
 2017-09-07 - 2018-07-15: /shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/AdsPlus/DecorationQuality/FBS/MergedData/%Y/%m/SsLabelData.%Y-%m-%d.ss
 After 2018-07-15: /shares/BingAds.Algo.Relevance/Team/Relevance/FBSDecoration/FlightsMWMW/166/DBLabelData/%Y/%m/DbLabelData-Done-%Y-%m-%d.tsv
- Feature Extraction: 
aether://experiments/fa3d372c-be5b-4bf0-a105-947252b2397d
- Training pipeline: 
Teacher: aether://experiments/d15d48ab-144e-4cb6-8431-c82089fee9ef
Student: aether://experiments/7b3a890c-93b3-4ea7-9bee-48dfad89216e
Ranker: aether://experiments/79da2d4e-80df-41b6-9da3-7dd0bf15b08e
- TriBert quantization scripts: [TriBert_downsize_To_ONNX_for_v9-1017-ExportMask.ipynb](https://onedrive.cloud.microsoft/:u:/a@p2qa9t5j/S/IQD76aPqbj-pTo2GnI70wewHATPLLCbmCQOwwFkYbHpMIeo?e=NhFuo3)
- Flight Analysis:
DR: [DecorationRelevanceV9DR.xlsx](https://microsoftapc-my.sharepoint.com/:x:/g/personal/jinjinchen_microsoft_com/EfVFgE8Xp-pGktnDrEsx1TcBe-3FFWBkw4wLThPfwwASDg?e=88gXac)
Cross Validation: [aether://experiments/89e58da8-2f60-4095-ba5f-49ce95e0173a](aether://experiments/89e58da8-2f60-4095-ba5f-49ce95e0173a)
Feature Parity Check & delay analysis: [aether://experiments/e46e93e6-f388-43d6-9e97-c7b8c4680f42](aether://experiments/e46e93e6-f388-43d6-9e97-c7b8c4680f42)
- Monitor dashboard: https://msit.powerbi.com/groups/a19af3d7-1457-4294-938f-aa11ab11b257/reports/b23b9b6e-a008-4b54-b26b-a3521991e335/ReportSectionf4af2bca705b2b2bd1ab?ctid=72f988bf-86f1-41af-91ab-2d7cd011db47&openReportSource=EmailSubscription&experience=power-bi&bookmarkGuid=5f3dcecf-9fb0-458c-8f0e-d486a700cba7
- Monitor pipeline: aether://experiments/af8103bf-b465-4d6e-b1ef-4003afe1b960
- Cosmos SStream monitor setting:
 [GeneVectorPublishMonitoring](https://cpwebsrv.binginternal.com/OfflineMonitoringNew/StreamInfoMonitor/ShowStreams?teamName=CPOffline&missionName=GeneMultiVectorPublish#)
[DecorationRelVectorMonitoring](https://cpwebsrv.binginternal.com/OfflineMonitoringNew/StreamInfoMonitor/ShowStreams?teamName=CPOffline&missionName=DecorationRelTriBertMonitor#)














































































