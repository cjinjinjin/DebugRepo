import type { ChapterStepProps } from "../../registry/types";
import "./Compaction.css";

export default function CompactionChapter({ step }: ChapterStepProps) {
  if (step === 0) return <div className="cp-scene scene-pad"><h1 className="cp-title">Compaction Without Cache Collapse</h1><div className="cp-meter"><span>context used</span><div><i style={{width:'92%'}} /></div></div></div>;
  if (step === 1) return <div className="cp-scene scene-pad"><h2 className="cp-h2">Naive approach</h2><div className="card cp-trap">separate summarizer prompt + different tool set = zero prefix reuse</div></div>;
  if (step === 2) return <div className="cp-scene scene-pad"><h2 className="cp-h2">Divergence point</h2><div className="cp-strip">{Array.from({length:10}).map((_,i)=><span key={i} className={i===0?'is-break':''}>{i+1}</span>)}</div><p className="cp-caption">break at token 1</p></div>;
  if (step === 3) return <div className="cp-scene scene-pad"><h2 className="cp-h2">Cache-safe fork</h2><div className="cp-fork"><div className="card">parent prefix</div><div className="cp-forkline" /><div className="card">compaction request tail</div></div></div>;
  if (step === 4) return <div className="cp-scene scene-pad"><h2 className="cp-h2">Buffer budgeting</h2><div className="cp-budget"><div className="card">history window</div><div className="card">compaction prompt</div><div className="card">summary output reserve</div></div></div>;
  return <div className="cp-scene scene-pad"><h2 className="cp-h2">Cost outcome</h2><div className="cp-compare"><div className="card"><b>naive</b><i style={{width:'88%'}} /></div><div className="card"><b>cache-safe fork</b><i style={{width:'41%'}} /></div></div></div>;
}
