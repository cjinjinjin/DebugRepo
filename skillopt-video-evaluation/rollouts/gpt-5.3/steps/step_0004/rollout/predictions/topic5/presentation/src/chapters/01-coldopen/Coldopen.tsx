import type { ChapterStepProps } from "../../registry/types";
import "./Coldopen.css";

export default function ColdopenChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="co-scene scene-pad">
        <div className="co-grid">
          <div>
            <div className="kicker">CLAUDE DESIGN / 2026-04-17</div>
            <h1 className="co-h1 serif-cn">它不是 AI 版 Figma</h1>
            <p className="co-sub">发布日 + 市场波动，让设计工具叙事瞬间改写。</p>
            <div className="co-bars">
              <span style={{ height: "42%" }} />
              <span style={{ height: "68%" }} />
              <span style={{ height: "53%" }} />
              <span style={{ height: "88%" }} />
              <span style={{ height: "36%" }} />
            </div>
          </div>
          <img className="co-img" src={`${import.meta.env.BASE_URL}assets/wechat-01.jpg`} alt="wechat" />
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="co-scene scene-pad">
        <div className="kicker">ROLE FLIP</div>
        <div className="co-split">
          <div className="card co-card">
            <div className="label-mono">传统工具</div>
            <div className="co-role">人主操作 · AI 辅助</div>
          </div>
          <div className="co-arrow">→</div>
          <div className="card co-card co-card-accent">
            <div className="label-mono">Claude Design</div>
            <div className="co-role">AI 主生成 · 人主审阅</div>
          </div>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="co-scene scene-pad">
        <div className="kicker">RUNNABLE OUTPUT</div>
        <div className="co-code card">
          <div className="co-tabs"><span>link</span><span>tab</span><span>diff</span></div>
          <pre className="co-pre">{`<button class=\"accent\">Try</button>\n<tabs active=\"A\" />\n+ CTA text: \"Start\"`}</pre>
        </div>
      </div>
    );
  }

  return (
    <div className="co-scene scene-pad">
      <div className="kicker">ITERATION GAP</div>
      <div className="co-compare">
        <div className="co-col">
          <div className="co-num hero-num">20+</div>
          <div className="co-label">传统复杂交互轮次</div>
        </div>
        <div className="co-vs">VS</div>
        <div className="co-col">
          <div className="co-num hero-num">2</div>
          <div className="co-label">Claude Design 轮次</div>
        </div>
      </div>
    </div>
  );
}
