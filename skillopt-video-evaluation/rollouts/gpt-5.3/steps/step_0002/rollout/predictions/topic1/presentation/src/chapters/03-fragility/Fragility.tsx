import type { ChapterStepProps } from "../../registry/types";
import "./Fragility.css";

const checks = ["freeze static prefix", "deterministic tool order", "stable schema signatures", "change via messages"];

export default function FragilityChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return <div className="fg-scene scene-pad"><h1 className="fg-title">Cache Fragility Map</h1><div className="fg-grid">{["timestamp", "tool order", "params"].map((x) => <div key={x} className="card fg-box">{x}</div>)}</div></div>;
  }

  if (step === 1) {
    return <div className="fg-scene scene-pad"><h2 className="fg-h2">Timestamp Drift</h2><div className="card fg-code"><pre>system_prompt
version: 12
generated_at: 2026-06-16T16:38:33Z</pre></div><div className="fg-bad">volatile value inside static prefix</div></div>;
  }

  if (step === 2) {
    return <div className="fg-scene scene-pad"><h2 className="fg-h2">Tool order diff</h2><div className="fg-diff"><div className="card fg-col"><b>A</b><p>search → read → patch</p></div><div className="card fg-col"><b>B</b><p>read → search → patch</p></div></div><div className="fg-bad">same tools, different sequence, different cache key</div></div>;
  }

  if (step === 3) {
    return <div className="fg-scene scene-pad"><h2 className="fg-h2">Schema / param drift</h2><div className="card fg-schema"><span>max_tokens: 1024</span><span>temperature: 0.2</span><span>tools: +1 provider</span></div><div className="fg-bad">prefix fingerprint changes</div></div>;
  }

  return (
    <div className="fg-scene scene-pad">
      <h2 className="fg-h2">Guardrails</h2>
      <div className="fg-checks">{checks.map((c) => <div key={c} className="card fg-check">✓ {c}</div>)}</div>
    </div>
  );
}
