import type { ChapterStepProps } from "../../registry/types";
import "./LlmPipeline.css";

/**
 * Chapter 03 — LLM Labeling Pipeline Evolution
 *
 * 8 steps:
 *   0  3-stage pipeline (QU → DI → Crossing)
 *   1  DV3: 53% → 70%, key prompt insight
 *   2  DV3 multi-market scoreboard
 *   3  GPT-4T: 42% fail
 *   4  GPT-4T recovery: ChatML + 18 experiments → 86.90%
 *   5  Flight verification: DCO -39%, HSL +17.77%
 *   6  GPT-4o: explain-then-judge 73% → 84%
 *   7  Cost trajectory: $0.184 → $0.019 → $0.0072
 */
export default function LlmPipeline({ step }: ChapterStepProps) {
  /* step 0 */
  if (step === 0) {
    return (
      <div className="lp-wrap">
        <div className="lp-pipeline-layout">
          <div className="lp-pipeline-title">3-Stage Pipeline</div>
          <div className="lp-stages">
            <div className="lp-stage-box card">
              <div className="lp-stage-num hero-num">1</div>
              <div className="lp-stage-name">Query Understanding</div>
              <div className="lp-stage-desc">Extract intent, entity, category</div>
            </div>
            <div className="lp-arrow">&rarr;</div>
            <div className="lp-stage-box card">
              <div className="lp-stage-num hero-num">2</div>
              <div className="lp-stage-name">Decoration Intent</div>
              <div className="lp-stage-desc">Evaluate decoration vs query</div>
            </div>
            <div className="lp-arrow">&rarr;</div>
            <div className="lp-stage-box card">
              <div className="lp-stage-num hero-num">3</div>
              <div className="lp-stage-name">Crossing</div>
              <div className="lp-stage-desc">Final Bad / NonBad decision</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 1 */
  if (step === 1) {
    return (
      <div className="lp-wrap">
        <div className="lp-dv3-layout">
          <div className="lp-dv3-title">DV3 — 2023</div>
          <div className="lp-dv3-badge">13 Prompt Iterations &middot; US Market</div>
          <div className="lp-dv3-acc">
            <div className="lp-dv3-from hero-num">53%</div>
            <div className="lp-dv3-arrow">&rarr;</div>
            <div className="lp-dv3-to hero-num">70%</div>
          </div>
          <div className="lp-dv3-quote">
            &ldquo;Evaluate query-decoration relevance. Not ad-query. Not decoration-ad. Just query to decoration.&rdquo;
          </div>
        </div>
      </div>
    );
  }

  /* step 2 */
  if (step === 2) {
    const markets = [
      { code: "US", val: "85.64", hi: true },
      { code: "DE", val: "81.50", hi: false },
      { code: "FR", val: "84.79", hi: false },
      { code: "ES", val: "77.82", hi: false },
      { code: "SV", val: "79.64", hi: false },
      { code: "JA", val: "61.96", lo: true },
      { code: "PT", val: "63.27", lo: true },
    ];
    return (
      <div className="lp-wrap">
        <div className="lp-scores-layout">
          <div className="lp-scores-title">DV3 Finalized — Multi-Market</div>
          <div className="lp-scores-grid">
            {markets.map((m) => (
              <div key={m.code} className="lp-score-card card">
                <div className="lp-score-market">{m.code}</div>
                <div className={`lp-score-val hero-num${m.hi ? " lp-score-val-hi" : ""}${(m as any).lo ? " lp-score-val-lo" : ""}`}>
                  {m.val}%
                </div>
                <div className="lp-score-unit">overall accuracy</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 3 */
  if (step === 3) {
    return (
      <div className="lp-wrap">
        <div className="lp-fail-layout">
          <div className="lp-fail-title">GPT-4 Turbo — Early 2024</div>
          <div className="lp-fail-year">Same DV3 Prompts Applied Directly</div>
          <div className="lp-fail-big hero-num">42%</div>
          <div className="lp-fail-label">Worse Than Random</div>
        </div>
      </div>
    );
  }

  /* step 4 */
  if (step === 4) {
    const fixes = [
      { icon: "01", text: "ChatML format with <|im_start|> / <|im_end|>" },
      { icon: "02", text: "Stop tokens: <|im_end|>" },
      { icon: "03", text: "Token limits: 200 decoration / 100 crossing" },
      { icon: "04", text: "Frequency & presence penalty tuning" },
    ];
    return (
      <div className="lp-wrap">
        <div className="lp-recover-layout">
          <div className="lp-recover-title">Recovery: Format Matters</div>
          <div className="lp-recover-fixes">
            {fixes.map((f) => (
              <div key={f.icon} className="lp-recover-fix card">
                <div className="lp-recover-icon">{f.icon}</div>
                <div className="lp-recover-text">{f.text}</div>
              </div>
            ))}
          </div>
          <div className="lp-recover-result">
            <div className="lp-recover-exp">18 experiments</div>
            <div className="lp-recover-arrow">&rarr;</div>
            <div className="lp-recover-val hero-num">86.90%</div>
          </div>
        </div>
      </div>
    );
  }

  /* step 5 */
  if (step === 5) {
    return (
      <div className="lp-wrap">
        <div className="lp-flight-layout">
          <div className="lp-flight-title">Flight: DV3 vs GPT-4 Turbo</div>
          <div className="lp-flight-grid">
            <div className="lp-flight-card card">
              <div className="lp-flight-metric">DCO Defect</div>
              <div className="lp-flight-value lp-flight-good hero-num">&minus;39%</div>
              <div className="lp-flight-pval">p = 0.0023</div>
            </div>
            <div className="lp-flight-card card">
              <div className="lp-flight-metric">HSL Regression</div>
              <div className="lp-flight-value lp-flight-bad hero-num">+17.8%</div>
              <div className="lp-flight-pval">p = 0.021</div>
            </div>
            <div className="lp-flight-card card">
              <div className="lp-flight-metric">VSL Improvement</div>
              <div className="lp-flight-value lp-flight-good hero-num">&minus;11%</div>
              <div className="lp-flight-pval">p = 0.019</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 6 */
  if (step === 6) {
    return (
      <div className="lp-wrap">
        <div className="lp-gpt4o-layout">
          <div className="lp-gpt4o-title">GPT-4o — Explain Then Judge</div>
          <div className="lp-gpt4o-compare">
            <div className="lp-gpt4o-path card">
              <div className="lp-gpt4o-path-label">Judge First</div>
              <div className="lp-gpt4o-path-value lp-gpt4o-dim hero-num">73%</div>
              <div className="lp-gpt4o-path-desc">Score immediately, no reasoning</div>
            </div>
            <div className="lp-gpt4o-path card lp-gpt4o-path-active">
              <div className="lp-gpt4o-path-label">Explain Then Judge</div>
              <div className="lp-gpt4o-path-value lp-gpt4o-bright hero-num">84%</div>
              <div className="lp-gpt4o-path-desc">Short explanation first, then score</div>
            </div>
          </div>
          <div className="lp-gpt4o-insight">Short &gt; Long &middot; Exp #6: 84.53% &middot; Exp #5: 83.30%</div>
        </div>
      </div>
    );
  }

  /* step 7 */
  if (step === 7) {
    return (
      <div className="lp-wrap">
        <div className="lp-cost-layout">
          <div className="lp-cost-title">Cost per Item</div>
          <div className="lp-cost-tags">
            <div className="lp-cost-tag card">
              <div className="lp-cost-tag-name">DV3</div>
              <div className="lp-cost-tag-val hero-num">$0.184</div>
            </div>
            <div className="lp-cost-tag card">
              <div className="lp-cost-tag-name">GPT-4 Turbo</div>
              <div className="lp-cost-tag-val hero-num">$0.019</div>
            </div>
            <div className="lp-cost-tag card lp-cost-tag-active">
              <div className="lp-cost-tag-name">GPT-4o</div>
              <div className="lp-cost-tag-val hero-num">$0.0072</div>
            </div>
          </div>
          <div className="lp-cost-hero hero-num">25&times; in 18 months</div>
          <div className="lp-cost-sub">Cost reduction trajectory</div>
        </div>
      </div>
    );
  }

  return null;
}
