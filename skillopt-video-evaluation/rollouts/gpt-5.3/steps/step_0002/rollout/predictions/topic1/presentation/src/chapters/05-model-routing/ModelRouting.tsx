import type { ChapterStepProps } from "../../registry/types";
import "./ModelRouting.css";

export default function ModelRoutingChapter({ step }: ChapterStepProps) {
  if (step === 0) return <div className="mr-scene scene-pad"><h1 className="mr-title">Model-Specific Cache Boundaries</h1><div className="mr-boundary"><div className="card mr-model">Opus cache</div><div className="card mr-model">Haiku cache</div></div></div>;
  if (step === 1) return <div className="mr-scene scene-pad"><h2 className="mr-h2">Cost paradox</h2><div className="mr-bars"><div className="mr-bar"><span>stay on Opus</span><i style={{width:'52%'}} /></div><div className="mr-bar"><span>late switch to Haiku</span><i style={{width:'86%'}} /></div></div></div>;
  if (step === 2) return <div className="mr-scene scene-pad"><h2 className="mr-h2">Subagent handoff pattern</h2><div className="mr-flow"><div className="card">deep thread on Opus</div><div className="mr-arr">→</div><div className="card">compact summary packet</div><div className="mr-arr">→</div><div className="card">route to Haiku</div></div></div>;
  return <div className="mr-scene scene-pad"><div className="card mr-rule"><div className="kicker">routing rule</div><h3>Switch early or hand off compact.</h3><p>Do not cold-start a giant prefix on a new model midstream.</p></div></div>;
}
