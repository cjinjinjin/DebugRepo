import type { ChapterStepProps } from "../../registry/types";
import "./Robust.css";

/**
 * Chapter 05 — Robust Training & Coverage
 *
 * 5 steps:
 *   0  Split screen: normal vectors vs livesite zeros + coverage bars
 *   1  Robust training: mixing zero vectors + IsBertValid / IsWBValid
 *   2  Tradeoff: normal AUC vs livesite AUC
 *   3  WoodBlock limitation: only SL/SLAB trained
 *   4  Calibration: per-task threshold tuning
 */
export default function Robust({ step }: ChapterStepProps) {
  /* step 0 */
  if (step === 0) {
    const coverage = [
      { name: "Sitelinks", pct: 99 },
      { name: "FourthLine", pct: 80, low: true },
      { name: "SSExtension", pct: 99 },
      { name: "SmartCategory", pct: 90 },
      { name: "Callout", pct: 99 },
      { name: "DCO", pct: 90 },
      { name: "Ad Vector", pct: 93 },
    ];
    return (
      <div className="rb-wrap">
        <div className="rb-coverage-layout">
          <div className="rb-coverage-title">Feature Coverage Gaps</div>
          <div className="rb-coverage-split">
            <div className="rb-coverage-left">
              <div className="rb-coverage-label">Normal</div>
              <div className="rb-coverage-desc">Full feature vectors</div>
              <div className="rb-vector-display rb-vector-full">
                <div className="rb-vector-row">
                  {[1,2,3,4,5,6].map(i => <div key={i} className="rb-vector-cell rb-vector-ok" />)}
                </div>
              </div>
            </div>
            <div className="rb-coverage-vs">vs</div>
            <div className="rb-coverage-right">
              <div className="rb-coverage-label rb-coverage-label-bad">Livesite</div>
              <div className="rb-coverage-desc">Vectors return zeros</div>
              <div className="rb-vector-display rb-vector-broken">
                <div className="rb-vector-row">
                  {[1,2,3,4,5,6].map(i => <div key={i} className={`rb-vector-cell${i <= 2 ? " rb-vector-zero" : " rb-vector-ok"}`} />)}
                </div>
              </div>
            </div>
          </div>
          <div className="rb-bars">
            {coverage.map((c, i) => (
              <div key={c.name} className="rb-bar-row" style={{ "--i": i } as React.CSSProperties}>
                <div className="rb-bar-name">{c.name}</div>
                <div className="rb-bar-track">
                  <div className={`rb-bar-fill${c.low ? " rb-bar-fill-low" : ""}`} style={{ width: `${c.pct}%` }} />
                </div>
                <div className={`rb-bar-pct hero-num${c.low ? " rb-bar-pct-low" : ""}`}>{c.pct}%</div>
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
      <div className="rb-wrap">
        <div className="rb-robust-layout">
          <div className="rb-robust-title">Robust Training</div>
          <div className="rb-robust-concept">
            <div className="rb-robust-mix">
              <div className="rb-robust-source card">
                <div className="rb-robust-source-label">Real Vectors</div>
                <div className="rb-robust-source-visual rb-robust-real">
                  {[1,2,3,4].map(i => <div key={i} className="rb-vector-cell rb-vector-ok" />)}
                </div>
              </div>
              <div className="rb-robust-plus">+</div>
              <div className="rb-robust-source card">
                <div className="rb-robust-source-label">Zero Vectors</div>
                <div className="rb-robust-source-visual rb-robust-zero">
                  {[1,2,3,4].map(i => <div key={i} className="rb-vector-cell rb-vector-zero" />)}
                </div>
              </div>
              <div className="rb-robust-equals">=</div>
              <div className="rb-robust-result card">
                <div className="rb-robust-source-label">Mixed Training</div>
              </div>
            </div>
          </div>
          <div className="rb-robust-signals">
            <div className="rb-robust-signal card">
              <div className="rb-robust-signal-name hero-num">IsBertValid</div>
              <div className="rb-robust-signal-desc">Tracks BERT feature availability</div>
            </div>
            <div className="rb-robust-signal card">
              <div className="rb-robust-signal-name hero-num">IsWBValid</div>
              <div className="rb-robust-signal-desc">Tracks WoodBlock feature availability</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 2 */
  if (step === 2) {
    return (
      <div className="rb-wrap">
        <div className="rb-tradeoff-layout">
          <div className="rb-tradeoff-title">The Tradeoff</div>
          <div className="rb-tradeoff-panels">
            <div className="rb-tradeoff-panel card">
              <div className="rb-tradeoff-panel-label">Normal Validation</div>
              <div className="rb-tradeoff-panel-delta rb-tradeoff-down hero-num">&darr; Small</div>
              <div className="rb-tradeoff-panel-desc">Slight AUC drop with zero mixing</div>
            </div>
            <div className="rb-tradeoff-panel card rb-tradeoff-panel-good">
              <div className="rb-tradeoff-panel-label">Livesite Validation</div>
              <div className="rb-tradeoff-panel-delta rb-tradeoff-up hero-num">&uarr; Large</div>
              <div className="rb-tradeoff-panel-desc">Much better AUC when features missing</div>
            </div>
          </div>
          <div className="rb-tradeoff-insight">Pick the right zero vector ratio &mdash; both sides stay acceptable</div>
        </div>
      </div>
    );
  }

  /* step 3 */
  if (step === 3) {
    return (
      <div className="rb-wrap">
        <div className="rb-wb-layout">
          <div className="rb-wb-title">WoodBlock Limitation</div>
          <div className="rb-wb-content">
            <div className="rb-wb-trained card">
              <div className="rb-wb-card-label">Trained On</div>
              <div className="rb-wb-items">
                <div className="rb-wb-item rb-wb-item-ok">Sitelink</div>
                <div className="rb-wb-item rb-wb-item-ok">SLAB</div>
              </div>
              <div className="rb-wb-card-desc">Query-decoration relevance pairs</div>
            </div>
            <div className="rb-wb-arrow">&rarr;</div>
            <div className="rb-wb-default card">
              <div className="rb-wb-card-label">Other Decoration Types</div>
              <div className="rb-wb-default-val hero-num">0</div>
              <div className="rb-wb-card-desc">Returns default value</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* step 4 */
  if (step === 4) {
    const tasks = [
      { name: "Sitelinks", shared: true },
      { name: "Extensions", shared: true },
      { name: "Annotations", shared: true },
      { name: "Callouts", shared: false },
    ];
    return (
      <div className="rb-wrap">
        <div className="rb-cal-layout">
          <div className="rb-cal-title">Calibration</div>
          <div className="rb-cal-concept">
            <div className="rb-cal-position card">
              <div className="rb-cal-position-label">Same Display Position</div>
              <div className="rb-cal-tasks">
                {tasks.map((t, i) => (
                  <div key={t.name} className={`rb-cal-task${t.shared ? " rb-cal-task-shared" : ""}`} style={{ "--i": i } as React.CSSProperties}>
                    {t.name}
                  </div>
                ))}
              </div>
            </div>
            <div className="rb-cal-arrow">&darr;</div>
            <div className="rb-cal-result card">
              <div className="rb-cal-result-text">Thresholds tuned per <strong>task</strong>, not per decoration type</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
