import type { ChapterStepProps } from "../../registry/types";
import "./LabelScale.css";

/**
 * Chapter 02 — Label Scale & Data Sources
 *
 * 6 steps:
 *   0  4-level labeling scale (E/G/F/B)
 *   1  Binary collapse: NonBad vs Bad
 *   2  Managed labeling era timeline + cost
 *   3  LLM era: DV3 → GPT-4T → GPT-4o
 *   4  GPT-4o accuracy gauge 83%
 *   5  Cost waterfall $0.238 → $0.0072, hero "27×"
 */
export default function LabelScale({ step }: ChapterStepProps) {
  /* step 0 */
  if (step === 0) {
    const tiers = [
      { letter: "E", name: "Excellent", desc: "Matches the query directly or is a more specific instance" },
      { letter: "G", name: "Good", desc: "Relevant but not specific enough; user may still have interest" },
      { letter: "F", name: "Fair", desc: "Weakly relevant — competitor, alternative, or complementary" },
      { letter: "B", name: "Bad", desc: "Off-intent or meaningless — should not be displayed", bad: true },
    ];
    return (
      <div className="ls-wrap">
        <div className="ls-scale-layout">
          <div className="ls-scale-title">Labeling Scale</div>
          <div className="ls-tiers">
            {tiers.map((t) => (
              <div key={t.letter} className={`ls-tier card${t.bad ? " ls-tier-bad" : ""}`}>
                <div className="ls-tier-letter hero-num">{t.letter}</div>
                <div className="ls-tier-name">{t.name}</div>
                <div className="ls-tier-desc">{t.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 1 */
  if (step === 1) {
    return (
      <div className="ls-wrap">
        <div className="ls-collapse-layout">
          <div className="ls-collapse-title">Binary in Production</div>
          <div className="ls-collapse-row">
            <div className="ls-group ls-group-nonbad card">
              <div className="ls-group-label ls-group-label-good">NonBad</div>
              <div className="ls-group-letters">
                <span className="ls-group-letter hero-num">E</span>
                <span className="ls-group-letter hero-num">G</span>
                <span className="ls-group-letter hero-num">F</span>
              </div>
              <div className="ls-group-action">Keep &rarr; show decoration</div>
            </div>
            <div className="ls-collapse-arrow">&rarr;</div>
            <div className="ls-group ls-group-bad card">
              <div className="ls-group-label ls-group-label-bad">Bad</div>
              <div className="ls-group-letters">
                <span className="ls-group-letter hero-num">B</span>
              </div>
              <div className="ls-group-action">Filter out &rarr; hide</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 2 */
  if (step === 2) {
    return (
      <div className="ls-wrap">
        <div className="ls-timeline-layout">
          <div className="ls-timeline-title">Managed Labeling Era</div>
          <div className="ls-timeline-bar-wrap">
            <div className="ls-timeline-era">6 Years of Human Judges</div>
            <div className="ls-timeline-bar ls-timeline-managed" />
            <div className="ls-timeline-dot ls-timeline-dot-start" />
            <div className="ls-timeline-dot ls-timeline-dot-end" />
            <div className="ls-timeline-year ls-timeline-year-start">2017.9</div>
            <div className="ls-timeline-year ls-timeline-year-end">2023.12</div>
          </div>
          <div className="ls-cost-badge card">
            <div className="ls-cost-label">Cost Per Item</div>
            <div className="ls-cost-value hero-num">$0.238</div>
            <div className="ls-cost-unit">managed labeling</div>
          </div>
        </div>
      </div>
    );
  }

  /* step 3 */
  if (step === 3) {
    return (
      <div className="ls-wrap">
        <div className="ls-llm-layout">
          <div className="ls-llm-title">LLM Labeling Era</div>
          <div className="ls-llm-timeline-wrap">
            <div className="ls-llm-bar-old" />
            <div className="ls-llm-bar-new" />
            <div className="ls-llm-dot ls-llm-dot-start" />
            <div className="ls-llm-dot ls-llm-dot-mid" />
            <div className="ls-llm-dot ls-llm-dot-end" />
            <div className="ls-llm-year ls-llm-year-start">2017</div>
            <div className="ls-llm-year ls-llm-year-mid">2023.12</div>
            <div className="ls-llm-year ls-llm-year-end">Now</div>
          </div>
          <div className="ls-llm-models">
            <div className="ls-llm-model card">
              <div className="ls-llm-model-name">DV3</div>
              <div className="ls-llm-model-year">2023</div>
            </div>
            <div className="ls-llm-model card">
              <div className="ls-llm-model-name">GPT-4 Turbo</div>
              <div className="ls-llm-model-year">Early 2024</div>
            </div>
            <div className="ls-llm-model card">
              <div className="ls-llm-model-name">GPT-4o</div>
              <div className="ls-llm-model-year">Mid 2024</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 4 */
  if (step === 4) {
    return (
      <div className="ls-wrap">
        <div className="ls-acc-layout">
          <div className="ls-acc-title">GPT-4o Accuracy</div>
          <div className="ls-gauge-wrap">
            <div className="ls-gauge-track" />
            <div className="ls-gauge-fill" />
            <div className="ls-gauge-value hero-num">83%</div>
          </div>
          <div className="ls-acc-model-badge">GPT-4o &middot; Across All Tasks</div>
          <div className="ls-acc-cost-line">
            Cost per item: <span>$0.0072</span>
          </div>
        </div>
      </div>
    );
  }

  /* step 5 */
  if (step === 5) {
    const bars = [
      { name: "Managed", price: "$0.238", height: 340 },
      { name: "DV3", price: "$0.184", height: 264 },
      { name: "GPT-4T", price: "$0.019", height: 32 },
      { name: "GPT-4o", price: "$0.0072", height: 12, active: true },
    ];
    return (
      <div className="ls-wrap">
        <div className="ls-waterfall-layout">
          <div className="ls-waterfall-title">Cost Trajectory</div>
          <div className="ls-waterfall-chart">
            {bars.map((b) => (
              <div key={b.name} className="ls-waterfall-col">
                <div className={`ls-waterfall-price${b.active ? " ls-waterfall-price-active" : ""} hero-num`}>
                  {b.price}
                </div>
                <div
                  className={`ls-waterfall-bar${b.active ? " ls-waterfall-bar-active" : ""}`}
                  style={{ height: b.height }}
                />
                <div className="ls-waterfall-name">{b.name}</div>
              </div>
            ))}
          </div>
          <div className="ls-waterfall-hero hero-num">27&times; cheaper</div>
        </div>
      </div>
    );
  }

  return null;
}
