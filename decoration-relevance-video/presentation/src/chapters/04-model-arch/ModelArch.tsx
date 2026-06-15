import type { ChapterStepProps } from "../../registry/types";
import "./ModelArch.css";

/**
 * Chapter 04 — Model Architecture & Features
 *
 * 7 steps:
 *   0  Teacher-student pipeline: CULR V4 → soft labels → TriBert V9
 *   1  Teacher evolution timeline: CLR V2 → V3 → CULR V3 → CULR V4
 *   2  TriBert multi-task: shared BERT + 4 classification heads
 *   3  Feature breakdown: Metastream / RC2 / QAS / WoodBlock
 *   4  QAS query categories (~24 tags)
 *   5  Two-stage training: managed → LLM
 *   6  V9 experiment results: 0.8527 → 0.9541
 */
export default function ModelArch({ step }: ChapterStepProps) {
  /* step 0 */
  if (step === 0) {
    return (
      <div className="ma-wrap">
        <div className="ma-pipeline-layout">
          <div className="ma-pipeline-title">Teacher–Student Pipeline</div>
          <div className="ma-pipeline-flow">
            <div className="ma-pipeline-box card ma-pipeline-teacher">
              <div className="ma-pipeline-box-label">Teacher</div>
              <div className="ma-pipeline-box-name hero-num">CULR V4</div>
              <div className="ma-pipeline-box-desc">Cross-lingual Universal Language Representation</div>
            </div>
            <div className="ma-pipeline-connector">
              <div className="ma-pipeline-line" />
              <div className="ma-pipeline-connector-label">soft labels</div>
            </div>
            <div className="ma-pipeline-box card ma-pipeline-student">
              <div className="ma-pipeline-box-label">Student</div>
              <div className="ma-pipeline-box-name hero-num">TriBert V9</div>
              <div className="ma-pipeline-box-desc">Multi-task production model</div>
            </div>
            <div className="ma-pipeline-connector">
              <div className="ma-pipeline-line" />
              <div className="ma-pipeline-connector-label">production</div>
            </div>
            <div className="ma-pipeline-endpoint card">
              <div className="ma-pipeline-box-name">FR Ranker</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 1 */
  if (step === 1) {
    const models = [
      { name: "CLR V2", desc: "US-only baseline", active: false },
      { name: "CLR V3", desc: "Improved US", active: false },
      { name: "CULR V3", desc: "Cross-lingual", active: false },
      { name: "CULR V4", desc: "Current best", active: true },
    ];
    return (
      <div className="ma-wrap">
        <div className="ma-evo-layout">
          <div className="ma-evo-title">Teacher Evolution</div>
          <div className="ma-evo-timeline">
            <div className="ma-evo-line" />
            {models.map((m, i) => (
              <div key={m.name} className={`ma-evo-node${m.active ? " ma-evo-node-active" : ""}`} style={{ "--i": i } as React.CSSProperties}>
                <div className="ma-evo-dot" />
                <div className="ma-evo-name">{m.name}</div>
                <div className="ma-evo-desc">{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 2 */
  if (step === 2) {
    const heads = [
      { id: "T0", name: "Sitelinks" },
      { id: "T1", name: "Callouts" },
      { id: "T2", name: "Snippets" },
      { id: "T3", name: "Price Ext." },
    ];
    return (
      <div className="ma-wrap">
        <div className="ma-tribert-layout">
          <div className="ma-tribert-title">TriBert — Multi-Task</div>
          <div className="ma-tribert-diagram">
            <div className="ma-tribert-core card">
              <div className="ma-tribert-core-label">Shared BERT Backbone</div>
            </div>
            <div className="ma-tribert-branches">
              {heads.map((h, i) => (
                <div key={h.id} className="ma-tribert-head" style={{ "--i": i } as React.CSSProperties}>
                  <div className="ma-tribert-stem" />
                  <div className="ma-tribert-head-box card">
                    <div className="ma-tribert-head-id hero-num">{h.id}</div>
                    <div className="ma-tribert-head-name">{h.name}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="ma-tribert-caption">Same encoder, scored differently</div>
        </div>
      </div>
    );
  }

  /* step 3 */
  if (step === 3) {
    const features = [
      { name: "Metastream", desc: "Query, ad, and decoration signals", tag: "Input Signals" },
      { name: "RC2", desc: "QKRelevance · QACDefect · QLPDefect", tag: "Contextual" },
      { name: "WoodBlock", desc: "Lightweight feature extraction", tag: "Extraction" },
      { name: "TriBert", desc: "Deep multi-task scoring", tag: "Core Model" },
    ];
    return (
      <div className="ma-wrap">
        <div className="ma-feat-layout">
          <div className="ma-feat-title">Feature Breakdown</div>
          <div className="ma-feat-grid">
            {features.map((f, i) => (
              <div key={f.name} className="ma-feat-card card" style={{ "--i": i } as React.CSSProperties}>
                <div className="ma-feat-tag">{f.tag}</div>
                <div className="ma-feat-name">{f.name}</div>
                <div className="ma-feat-desc">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 4 */
  if (step === 4) {
    const tags = [
      "Health", "Navigational", "Adult", "Image", "Commerce",
      "Finance", "Video", "Local", "News", "Weather",
      "Sports", "Entertainment", "Education", "Travel", "Food",
      "Tech", "Gaming", "Music", "Shopping", "Automotive",
      "Real Estate", "Jobs", "Legal", "Insurance",
    ];
    return (
      <div className="ma-wrap">
        <div className="ma-qas-layout">
          <div className="ma-qas-title">QAS Query Categories</div>
          <div className="ma-qas-sub">~24 signal categories from Query Understanding</div>
          <div className="ma-qas-tags">
            {tags.map((t, i) => (
              <div key={t} className="ma-qas-tag card" style={{ "--i": i } as React.CSSProperties}>{t}</div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 5 */
  if (step === 5) {
    return (
      <div className="ma-wrap">
        <div className="ma-train-layout">
          <div className="ma-train-title">Two-Stage Training</div>
          <div className="ma-train-flow">
            <div className="ma-train-stage card">
              <div className="ma-train-stage-num hero-num">1</div>
              <div className="ma-train-stage-name">Managed Labels</div>
              <div className="ma-train-stage-desc">Years of human judge data</div>
            </div>
            <div className="ma-train-arrow">&rarr;</div>
            <div className="ma-train-ckpt card">
              <div className="ma-train-ckpt-label">Checkpoint</div>
            </div>
            <div className="ma-train-arrow">&rarr;</div>
            <div className="ma-train-stage card ma-train-stage-active">
              <div className="ma-train-stage-num hero-num">2</div>
              <div className="ma-train-stage-name">LLM Labels</div>
              <div className="ma-train-stage-desc">GPT-4o generated labels</div>
            </div>
          </div>
          <div className="ma-train-insight">Two-stage beats either source alone</div>
        </div>
      </div>
    );
  }

  /* step 6 */
  if (step === 6) {
    const findings = [
      { label: "CULR > CLR", good: true },
      { label: "Two-stage > One-stage", good: true },
      { label: "Finetune CKPT > Pretrain CKPT", good: true },
    ];
    return (
      <div className="ma-wrap">
        <div className="ma-v9-layout">
          <div className="ma-v9-title">V9 Experiment Results</div>
          <div className="ma-v9-compare">
            <div className="ma-v9-baseline card">
              <div className="ma-v9-label">CLR V2 Baseline</div>
              <div className="ma-v9-value hero-num">0.8527</div>
              <div className="ma-v9-unit">US Test AUC</div>
            </div>
            <div className="ma-v9-arrow">&rarr;</div>
            <div className="ma-v9-best card">
              <div className="ma-v9-label">CULR V4 + Two-Stage</div>
              <div className="ma-v9-value ma-v9-value-hi hero-num">0.9541</div>
              <div className="ma-v9-unit">US Test AUC</div>
            </div>
          </div>
          <div className="ma-v9-findings">
            {findings.map((f, i) => (
              <div key={f.label} className="ma-v9-finding" style={{ "--i": i } as React.CSSProperties}>
                <span className="ma-v9-check">&check;</span>
                <span>{f.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return null;
}
