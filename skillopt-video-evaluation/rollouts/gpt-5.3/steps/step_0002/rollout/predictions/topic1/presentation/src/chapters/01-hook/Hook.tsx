import type { ChapterStepProps } from "../../registry/types";
import "./Hook.css";

const metrics = [
  { label: "latency", value: "-42%" },
  { label: "cost", value: "-38%" },
  { label: "session length", value: "+3.1x" },
];

export default function HookChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="hk-scene scene-pad">
        <div className="masthead"><span className="brand">Prompt Cache Ops</span><span className="issue">chapter 1 · hook</span></div>
        <hr className="rule hk-rule" />
        <div className="hk-hero-wrap">
          <div className="hero-num hk-num">01</div>
          <h1 className="hk-title">Prompt Caching Is Product Infrastructure</h1>
          <p className="hk-sub">Not a backend footnote. A top-level design decision.</p>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="hk-scene scene-pad">
        <div className="hk-contrast">
          <div className="card hk-panel"><div className="hk-panel-title">Usable product</div><div className="hk-panel-v">Stable prefix reuse</div></div>
          <div className="hk-vs">VS</div>
          <div className="card hk-panel hk-danger"><div className="hk-panel-title">Expensive demo</div><div className="hk-panel-v">Cache miss every turn</div></div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="hk-scene scene-pad">
        <h2 className="hk-h2">Incident-style cache monitoring</h2>
        <div className="hk-metrics">
          {metrics.map((m) => (
            <div className="card hk-metric" key={m.label}><div className="label-mono">{m.label}</div><div className="hk-mv">{m.value}</div></div>
          ))}
        </div>
        <div className="hk-alert">ALERT: hit-rate anomaly &lt; SLO</div>
      </div>
    );
  }

  return (
    <div className="hk-scene scene-pad">
      <div className="pull-quote hk-quote">Architecture first: design every turn around prefix stability.</div>
      <div className="hk-rail">
        {["prompt order", "tool catalog", "model routing", "compaction"].map((x, i) => (
          <div key={x} className={`hk-node ${i < 4 ? "is-on" : ""}`}>{x}</div>
        ))}
      </div>
    </div>
  );
}
