import type { ChapterStepProps } from "../../registry/types";
import "./PrimitivesMemory.css";

const cards = [
  { t: "Scheduling", d: "Cadence driver" },
  { t: "Worktrees", d: "Safe parallelism" },
  { t: "Skills", d: "Persistent knowledge" },
  { t: "Connectors", d: "External reach" },
  { t: "Sub-agents", d: "Maker/checker split" },
  { t: "Memory", d: "Durable state spine" },
];

export default function PrimitivesMemoryChapter({ step }: ChapterStepProps) {
  const active = cards[step] ?? cards[cards.length - 1];
  return (
    <div className="pm-scene scene-pad">
      <div className="kicker">Building blocks + memory</div>
      <div className="pm-layout">
        <div className="pm-grid">
          {cards.map((c, i) => (
            <article key={c.t} className={`card pm-card ${i === step ? "is-on" : ""}`}>
              <h3>{c.t}</h3>
              <p>{c.d}</p>
            </article>
          ))}
        </div>
        <aside className="pm-focus card">
          <div className="hero-num pm-num">{String(step + 1).padStart(2, "0")}</div>
          <h2>{active.t}</h2>
          <p>{active.d}</p>
        </aside>
      </div>
    </div>
  );
}

