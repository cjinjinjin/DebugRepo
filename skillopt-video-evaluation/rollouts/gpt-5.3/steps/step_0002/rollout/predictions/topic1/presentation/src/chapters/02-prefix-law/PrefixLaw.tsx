import type { ChapterStepProps } from "../../registry/types";
import "./PrefixLaw.css";

const good = ["system + tools", "project context", "session context", "conversation"];
const bad = ["timestamp", "latest user text", "tool mutation", "system + tools"];

export default function PrefixLawChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="pl-scene scene-pad">
        <h1 className="pl-title">Prefix Match Mental Model</h1>
        <div className="pl-strip">
          {Array.from({ length: 12 }).map((_, i) => <span key={i} className={`pl-token ${i < 8 ? "is-hit" : ""}`} />)}
        </div>
        <p className="pl-note">Cache reuse stops at the first divergence.</p>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="pl-scene scene-pad">
        <h2 className="pl-h2">Good ordering</h2>
        <div className="pl-lane">{good.map((x, i) => <div key={x} className="card pl-item"><span className="hero-num pl-i">0{i + 1}</span><span>{x}</span></div>)}</div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="pl-scene scene-pad">
        <h2 className="pl-h2">Shared expensive prefix across sessions</h2>
        <div className="pl-sessions">
          {["session A", "session B", "session C"].map((s) => (
            <div key={s} className="pl-session"><span>{s}</span><div className="pl-bar"><i className="pl-hit" /><i className="pl-tail" /></div></div>
          ))}
        </div>
      </div>
    );
  }

  if (step === 3) {
    return (
      <div className="pl-scene scene-pad">
        <h2 className="pl-h2">Bad ordering</h2>
        <div className="pl-lane">{bad.map((x, i) => <div key={x} className={`card pl-item ${i===0?"is-bad":""}`}><span className="hero-num pl-i">0{i + 1}</span><span>{x}</span></div>)}</div>
        <div className="pl-break">cache breaks at token #1</div>
      </div>
    );
  }

  return (
    <div className="pl-scene scene-pad">
      <div className="card pl-rulecard">
        <div className="kicker">prefix law</div>
        <div className="pl-ruletext">Static First · Dynamic Last</div>
        <hr className="rule" />
        <p>Prompt order is architecture, not formatting style.</p>
      </div>
    </div>
  );
}
