import type { ChapterStepProps } from "../../registry/types";
import "./DailyOps.css";

/**
 * Chapter 08 — Daily Operations
 *
 * 5 steps:
 *   0  MetaStream pipeline flow (3 streams)
 *   1  Vector refresh + publish sequence (xlite → prod)
 *   2  Model check-in process (PR → deploy → FPS)
 *   3  Monitoring stack (PowerBI / Aether / Cosmos)
 *   4  Closing summary — unified system recap
 */
export default function DailyOps({ step }: ChapterStepProps) {
  /* step 0 — daily pipeline flow */
  if (step === 0) {
    const streams = [
      { num: "1", name: "New Impression Annotation", desc: "Daily fresh labels" },
      { num: "2", name: "180-Day History Combine", desc: "With extensions" },
      { num: "3", name: "SLAB + Image Extension", desc: "Specialized streams" },
    ];
    return (
      <div className="do-wrap">
        <div className="do-pipeline-layout">
          <div className="do-pipeline-title">Daily Pipeline</div>
          <div className="do-pipeline-badge">MetaStream Schedules</div>
          <div className="do-pipeline-flow">
            {streams.map((s) => (
              <div key={s.num} className="do-stream card">
                <div className="do-stream-num hero-num">{s.num}</div>
                <div className="do-stream-info">
                  <div className="do-stream-name">{s.name}</div>
                  <div className="do-stream-desc">{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 1 — vector refresh + publish */
  if (step === 1) {
    const phases = [
      { tag: "xlite", desc: "Canary validation" },
      { tag: "gate", desc: "Quality check" },
      { tag: "prod", desc: "Full rollout" },
    ];
    return (
      <div className="do-wrap">
        <div className="do-vector-layout">
          <div className="do-vector-title">Vector Refresh &amp; Publish</div>
          <div className="do-vector-cycles">
            <div className="do-cycle card">
              <div className="do-cycle-icon">D</div>
              <div className="do-cycle-name">Document Vectors</div>
            </div>
            <div className="do-cycle card">
              <div className="do-cycle-icon">A</div>
              <div className="do-cycle-name">Ads Vectors</div>
            </div>
          </div>
          <div className="do-publish-seq">
            {phases.map((p, i) => (
              <div key={p.tag} className="do-publish-step" style={{ animationDelay: `${400 + i * 200}ms` }}>
                {i > 0 && <div className="do-publish-arrow">&rarr;</div>}
                <div className="do-publish-tag card">
                  <div className="do-publish-tag-name">{p.tag}</div>
                  <div className="do-publish-tag-desc">{p.desc}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="do-publish-note">Bin file generation: once daily</div>
        </div>
      </div>
    );
  }

  /* step 2 — model check-in process */
  if (step === 2) {
    const steps = [
      { num: "1", name: "Create PR", detail: "Rnr-Offline-ClickPrediction repo" },
      { num: "2", name: "Deployment Steps", detail: "Follow OneNote guide" },
      { num: "3", name: "FPS Portal", detail: "Vector publishing" },
    ];
    return (
      <div className="do-wrap">
        <div className="do-checkin-layout">
          <div className="do-checkin-title">Model Check-In Process</div>
          <div className="do-checkin-flow">
            {steps.map((s, i) => (
              <div key={s.num} className="do-checkin-step" style={{ animationDelay: `${200 + i * 250}ms` }}>
                <div className="do-checkin-node card">
                  <div className="do-checkin-num hero-num">{s.num}</div>
                  <div className="do-checkin-name">{s.name}</div>
                  <div className="do-checkin-detail">{s.detail}</div>
                </div>
                {i < steps.length - 1 && <div className="do-checkin-connector" />}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 3 — monitoring stack */
  if (step === 3) {
    const layers = [
      { name: "PowerBI", desc: "Live metrics dashboards", icon: "PBI" },
      { name: "Aether", desc: "Offline monitoring pipelines", icon: "AET" },
      { name: "Cosmos SStream", desc: "Gene vector + TriBert DR vectors", icon: "CSS" },
    ];
    return (
      <div className="do-wrap">
        <div className="do-monitor-layout">
          <div className="do-monitor-title">Monitoring Stack</div>
          <div className="do-monitor-cards">
            {layers.map((l) => (
              <div key={l.name} className="do-monitor-card card">
                <div className="do-monitor-icon">{l.icon}</div>
                <div className="do-monitor-name">{l.name}</div>
                <div className="do-monitor-desc">{l.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* step 4 — closing summary */
  if (step === 4) {
    const points = [
      { val: "1", label: "Unified Model" },
      { val: "4", label: "Task Types" },
      { val: "27\u00d7", label: "Cost Reduction" },
      { val: "\u2713", label: "Global" },
    ];
    return (
      <div className="do-wrap">
        <div className="do-close-layout">
          <div className="do-close-title">TA Decoration Relevance</div>
          <div className="do-close-grid">
            {points.map((p) => (
              <div key={p.label} className="do-close-card card">
                <div className="do-close-val hero-num">{p.val}</div>
                <div className="do-close-label">{p.label}</div>
              </div>
            ))}
          </div>
          <div className="do-close-cta">
            Start here &rarr; <span className="do-close-handbook">DRI Handbook</span>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
