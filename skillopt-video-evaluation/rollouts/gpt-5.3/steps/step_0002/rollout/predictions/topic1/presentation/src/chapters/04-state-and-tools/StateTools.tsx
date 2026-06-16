import type { ChapterStepProps } from "../../registry/types";
import "./StateTools.css";

export default function StateToolsChapter({ step }: ChapterStepProps) {
  if (step === 0) return <div className="st-scene scene-pad"><h1 className="st-title">Keep Prefix Stable While Behavior Changes</h1><div className="card st-callout">Rule: state update != system prompt rewrite</div></div>;
  if (step === 1) return <div className="st-scene scene-pad"><h2 className="st-h2">Message update pattern</h2><div className="st-flow"><div className="card st-node">turn N result</div><div className="st-arrow">→</div><div className="card st-node">system-reminder message</div><div className="st-arrow">→</div><div className="card st-node">turn N+1 user input</div></div></div>;
  if (step === 2) return <div className="st-scene scene-pad"><h2 className="st-h2">Anti-pattern</h2><div className="st-split"><div className="card st-bad">rewrite system prompt every state change</div><div className="card st-bad">add/remove tools in active session</div></div></div>;
  if (step === 3) return <div className="st-scene scene-pad"><h2 className="st-h2">Plan Mode design</h2><div className="st-tools"><div className="card st-tool">tool catalog = fixed</div><div className="card st-tool is-accent">EnterPlanMode</div><div className="card st-tool is-accent">ExitPlanMode</div></div></div>;
  if (step === 4) return <div className="st-scene scene-pad"><h2 className="st-h2">Deferred loading</h2><div className="card st-load"><div>stable stubs in prefix</div><div>resolve full schema on selection</div><div>keep ordering deterministic</div></div></div>;
  return <div className="st-scene scene-pad"><div className="pull-quote st-quote">Stable prefix + dynamic capability = scalable agent architecture.</div></div>;
}
