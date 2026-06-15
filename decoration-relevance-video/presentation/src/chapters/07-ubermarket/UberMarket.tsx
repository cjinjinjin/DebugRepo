import type { ChapterStepProps } from "../../registry/types";
import "./UberMarket.css";

/**
 * Chapter 07 — UberMarket Migration
 *
 * 5 steps:
 *   0  US + INTL unification — merge visual, date badge
 *   1  Latency challenge — CPU bar climbing red
 *   2  ConditionalCall — terminal code block
 *   3  Two slots — NA Queene69 / INTL Queene13
 *   4  Summary badges — <1% CPU, no extra machines
 */
export default function UberMarket({ step }: ChapterStepProps) {
  /* step 0 — merge US + INTL */
  if (step === 0) {
    return (
      <div className="um-wrap">
        <div className="um-merge-layout">
          <div className="um-merge-title">UberMarket Migration</div>
          <div className="um-merge-visual">
            <div className="um-merge-box card">
              <div className="um-merge-label">US</div>
              <div className="um-merge-desc">PaidSearch NA</div>
            </div>
            <div className="um-merge-plus">+</div>
            <div className="um-merge-box card">
              <div className="um-merge-label">INTL</div>
              <div className="um-merge-desc">Global Markets</div>
            </div>
            <div className="um-merge-arrow">&rarr;</div>
            <div className="um-merge-box um-merge-unified card">
              <div className="um-merge-label">Unified</div>
              <div className="um-merge-desc">Same Infrastructure</div>
            </div>
          </div>
          <div className="um-merge-date card">
            <span className="um-merge-date-label">UMV1 Complete</span>
            <span className="um-merge-date-val hero-num">2024 / 07 / 31</span>
          </div>
        </div>
      </div>
    );
  }

  /* step 1 — latency challenge */
  if (step === 1) {
    return (
      <div className="um-wrap">
        <div className="um-latency-layout">
          <div className="um-latency-title">The Challenge: CPU Explosion</div>
          <div className="um-latency-visual">
            <div className="um-cpu-bar-track">
              <div className="um-cpu-bar-fill" />
              <div className="um-cpu-bar-danger" />
            </div>
            <div className="um-cpu-labels">
              <span className="um-cpu-label-low">0%</span>
              <span className="um-cpu-label-mid">50%</span>
              <span className="um-cpu-label-hi">100%</span>
            </div>
          </div>
          <div className="um-latency-warn">
            <span className="um-warn-icon">!</span>
            All decorations &times; all traffic groups = unsustainable
          </div>
        </div>
      </div>
    );
  }

  /* step 2 — ConditionalCall code */
  if (step === 2) {
    return (
      <div className="um-wrap">
        <div className="um-code-layout">
          <div className="um-code-title">The Fix: ConditionalCall</div>
          <div className="um-terminal card">
            <div className="um-terminal-bar">
              <span className="um-terminal-dot" />
              <span className="um-terminal-dot" />
              <span className="um-terminal-dot" />
              <span className="um-terminal-name">conditional_call.py</span>
            </div>
            <pre className="um-terminal-code">
{`if traffic_group in [
    MarketplaceClassifications.BingOO_1,
    MarketplaceClassifications.BingOO_2,
]:
    # Bing O&O only
    score = queene_model.predict(features)
elif related_to_account_id == 1004:
    score = queene_model.predict(features)
else:
    # Default: zero vector (skip model)
    score = default_zero_vector`}
            </pre>
          </div>
        </div>
      </div>
    );
  }

  /* step 3 — two serving slots */
  if (step === 3) {
    return (
      <div className="um-wrap">
        <div className="um-slots-layout">
          <div className="um-slots-title">Serving Topology</div>
          <div className="um-slots-row">
            <div className="um-slot card">
              <div className="um-slot-region">North America</div>
              <div className="um-slot-name hero-num">Queene69</div>
              <div className="um-slot-scope">Bing O&amp;O traffic</div>
            </div>
            <div className="um-slot card">
              <div className="um-slot-region">International</div>
              <div className="um-slot-name hero-num">Queene13</div>
              <div className="um-slot-scope">Bing O&amp;O traffic</div>
            </div>
          </div>
          <div className="um-slot-default card">
            <div className="um-slot-default-label">Everyone Else</div>
            <div className="um-slot-default-val">Default Zero Vector &mdash; model not called</div>
          </div>
        </div>
      </div>
    );
  }

  /* step 4 — summary badges */
  if (step === 4) {
    const stats = [
      { label: "CPU Increase", value: "<1%", accent: true },
      { label: "Extra Machines", value: "0", accent: true },
      { label: "Architecture", value: "Unified", accent: false },
    ];
    return (
      <div className="um-wrap">
        <div className="um-summary-layout">
          <div className="um-summary-title">Mission Accomplished</div>
          <div className="um-summary-badges">
            {stats.map((s) => (
              <div key={s.label} className="um-summary-badge card">
                <div className="um-summary-badge-val hero-num">{s.value}</div>
                <div className="um-summary-badge-label">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return null;
}
