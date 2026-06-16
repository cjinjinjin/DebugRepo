import type { ChapterStepProps } from "../../registry/types";
import "./OperatingModel.css";

const rules = [
  "prefix match is law",
  "updates via messages",
  "no mid-session tool/model churn",
  "monitor hit rate like uptime",
  "fork with shared prefix",
];

export default function OperatingModelChapter({ step }: ChapterStepProps) {
  if (step === 0) return <div className="om-scene scene-pad"><h1 className="om-title">Operating Model: Five Rules</h1><div className="om-rules">{rules.map((r, i) => <div className="card om-rule" key={r}><span className="hero-num om-n">0{i+1}</span>{r}</div>)}</div></div>;
  if (step === 1) return <div className="om-scene scene-pad"><h2 className="om-h2">Impact summary</h2><div className="om-impact"><div className="card"><b>latency</b><i style={{width:'66%'}} /></div><div className="card"><b>cost</b><i style={{width:'61%'}} /></div><div className="card"><b>session depth</b><i style={{width:'73%'}} /></div></div></div>;
  if (step === 2) return <div className="om-scene scene-pad"><h2 className="om-h2">Starter checklist</h2><div className="om-check">{["freeze prompt skeleton","stabilize tool order","set cache SLO alerts","define compaction fork path"].map((c)=> <div key={c} className="card om-item">□ {c}</div>)}</div></div>;
  return <div className="om-scene scene-pad"><div className="pull-quote om-close">Design prompts, tools, routing, and compaction around prefix stability from day one.</div></div>;
}
