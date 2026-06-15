import type { ChapterStepProps } from "../../registry/types";
import "./GlobalShip.css";

/**
 * Chapter 06 — Going Global
 *
 * 6 steps:
 *   0  US-only CLR teacher — world map with US lit
 *   1  CULR V4: multilingual — six countries lit
 *   2  Best teacher config — hyperparameter table
 *   3  INTL BSCAP dramatic jump — hero numbers
 *   4  Per-language scoreboard — 7 languages
 *   5  Ablation experiments — 3 findings
 */
export default function GlobalShip({ step }: ChapterStepProps) {
  /* step 0 — world map, US only */
  if (step === 0) {
    const dots = [
      { code: "US", x: 22, y: 42, lit: true },
      { code: "DE", x: 49, y: 32, lit: false },
      { code: "FR", x: 47, y: 36, lit: false },
      { code: "CN", x: 76, y: 42, lit: false },
      { code: "JP", x: 82, y: 38, lit: false },
      { code: "BR", x: 32, y: 68, lit: false },
      { code: "DK", x: 50, y: 28, lit: false },
    ];
    return (
      <div className="gs-wrap">
        <div className="gs-map-layout">
          <div className="gs-map-title">US-Only CLR Teacher</div>
          <div className="gs-map-canvas">
            {dots.map((d) => (
              <div
                key={d.code}
                className={`gs-dot${d.lit ? " gs-dot-lit" : " gs-dot-dim"}`}
                style={{ left: `${d.x}%`, top: `${d.y}%` }}
              >
                <div className="gs-dot-circle" />
                <div className="gs-dot-code">{d.code}</div>
              </div>
            ))}
          </div>
          <div className="gs-map-sub">International markets: no language understanding</div>
        </div>
      </div>
    );
  }

  /* step 1 — CULR V4 multilingual */
  if (step === 1) {
    const langs = [
      { code: "DE", name: "German", x: 49, y: 32 },
      { code: "FR", name: "French", x: 47, y: 36 },
      { code: "ZH", name: "Chinese", x: 76, y: 42 },
      { code: "JA", name: "Japanese", x: 82, y: 38 },
      { code: "PT", name: "Portuguese", x: 32, y: 68 },
      { code: "DA", name: "Danish", x: 50, y: 28 },
      { code: "US", name: "English", x: 22, y: 42 },
    ];
    return (
      <div className="gs-wrap">
        <div className="gs-map-layout">
          <div className="gs-culr-title">CULR V4 — Multilingual</div>
          <div className="gs-map-canvas">
            {langs.map((l, i) => (
              <div
                key={l.code}
                className="gs-dot gs-dot-lit gs-dot-pulse"
                style={{
                  left: `${l.x}%`,
                  top: `${l.y}%`,
                  animationDelay: `${i * 120}ms`,
                }}
              >
                <div className="gs-dot-circle" />
                <div className="gs-dot-code">{l.code}</div>
              </div>
            ))}
          </div>
          <div className="gs-culr-badge">Cross-lingual &middot; 6 Languages</div>
        </div>
      </div>
    );
  }

  /* step 2 — teacher config table */
  if (step === 2) {
    const rows = [
      { label: "text_a", value: "query + web4Ads + QU1" },
      { label: "text_b", value: "adcopy + decorationText" },
      { label: "Learning rate", value: "5e-6" },
      { label: "Gradient accum.", value: "8 steps" },
      { label: "Epochs", value: "1" },
      { label: "Max seq length", value: "256" },
    ];
    return (
      <div className="gs-wrap">
        <div className="gs-config-layout">
          <div className="gs-config-title">Best Teacher Config</div>
          <div className="gs-config-table card">
            {rows.map((r, i) => (
              <div key={r.label} className="gs-config-row" style={{ animationDelay: `${200 + i * 100}ms` }}>
                <div className="gs-config-label">{r.label}</div>
                <div className="gs-config-value">{r.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 3 — INTL BSCAP dramatic improvement */
  if (step === 3) {
    return (
      <div className="gs-wrap">
        <div className="gs-jump-layout">
          <div className="gs-jump-title">International Breakthrough</div>
          <div className="gs-jump-row">
            <div className="gs-jump-pair card">
              <div className="gs-jump-metric">INTL BSCAP &middot; Test Set</div>
              <div className="gs-jump-nums">
                <span className="gs-jump-from hero-num">0.617</span>
                <span className="gs-jump-arrow">&rarr;</span>
                <span className="gs-jump-to hero-num">0.934</span>
              </div>
            </div>
            <div className="gs-jump-pair card">
              <div className="gs-jump-metric">INTL BSCAP &middot; Goldset</div>
              <div className="gs-jump-nums">
                <span className="gs-jump-from hero-num">0.533</span>
                <span className="gs-jump-arrow">&rarr;</span>
                <span className="gs-jump-to hero-num">0.757</span>
              </div>
            </div>
          </div>
          <div className="gs-jump-verdict">Not incremental &mdash; a completely different league</div>
        </div>
      </div>
    );
  }

  /* step 4 — per-language scoreboard */
  if (step === 4) {
    const langs = [
      { code: "DE", name: "German", auc: "0.88" },
      { code: "FR", name: "French", auc: "0.90" },
      { code: "ZH-S", name: "Simplified Chinese", auc: "0.91", best: true },
      { code: "ZH-T", name: "Traditional Chinese", auc: "0.87" },
      { code: "JA", name: "Japanese", auc: "0.84" },
      { code: "PT", name: "Portuguese", auc: "0.86" },
      { code: "DA", name: "Danish", auc: "0.85" },
    ];
    return (
      <div className="gs-wrap">
        <div className="gs-lang-layout">
          <div className="gs-lang-title">Per-Language AUC</div>
          <div className="gs-lang-grid">
            {langs.map((l) => (
              <div key={l.code} className={`gs-lang-card card${l.best ? " gs-lang-card-best" : ""}`}>
                <div className="gs-lang-code">{l.code}</div>
                <div className={`gs-lang-auc hero-num${l.best ? " gs-lang-auc-best" : ""}`}>{l.auc}</div>
                <div className="gs-lang-name">{l.name}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 5 — ablation experiments */
  if (step === 5) {
    const findings = [
      {
        label: "web4Ads",
        verdict: "Marginal",
        desc: "Small gains on goldset, minimal on test set",
        tone: "neutral" as const,
      },
      {
        label: "QU1",
        verdict: "Mixed",
        desc: "Helped some splits but not others",
        tone: "neutral" as const,
      },
      {
        label: "EEM Rewrites",
        verdict: "Solid",
        desc: "QBSCAPTerm goldset 0.589 \u2192 0.626",
        tone: "good" as const,
      },
    ];
    return (
      <div className="gs-wrap">
        <div className="gs-ablation-layout">
          <div className="gs-ablation-title">Ablation Experiments</div>
          <div className="gs-ablation-cards">
            {findings.map((f) => (
              <div key={f.label} className="gs-ablation-card card">
                <div className="gs-ablation-label">{f.label}</div>
                <div className={`gs-ablation-verdict gs-ablation-${f.tone}`}>{f.verdict}</div>
                <div className="gs-ablation-desc">{f.desc}</div>
              </div>
            ))}
          </div>
          <div className="gs-ablation-hero">
            <span className="gs-ablation-from hero-num">0.589</span>
            <span className="gs-ablation-arrow">&rarr;</span>
            <span className="gs-ablation-to hero-num">0.626</span>
            <span className="gs-ablation-tag">EEM data augmentation</span>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
