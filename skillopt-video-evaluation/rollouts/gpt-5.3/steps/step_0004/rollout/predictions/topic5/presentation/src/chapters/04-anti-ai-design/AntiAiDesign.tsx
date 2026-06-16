import type { ChapterStepProps } from "../../registry/types";
import "./AntiAiDesign.css";

export default function AntiAiDesignChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="aa-scene scene-pad">
        <div className="kicker">ANTI-AI-SMELL CHECKLIST</div>
        <ul className="aa-list card">
          <li>过度渐变背景</li>
          <li>emoji 当图标</li>
          <li>彩色边框圆角卡片</li>
          <li>假数据堆砌</li>
        </ul>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="aa-scene scene-pad">
        <div className="aa-fonts">
          <div className="card aa-bad">模板味字体<br/>Inter / Roboto / Arial</div>
          <div className="aa-arrow">→</div>
          <div className="card aa-good">更有质感的替代<br/>Sora / Newsreader / Space Grotesk</div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="aa-scene scene-pad">
        <div className="kicker">COLOR POLICY</div>
        <div className="aa-flow card">
          <div>品牌色</div><span>→</span><div>oklch 派生</div><span>→</span><div>禁凭空造色</div>
        </div>
      </div>
    );
  }
  return (
    <div className="aa-scene scene-pad">
      <div className="aa-contrast">
        <div className="card aa-box">
          <div className="label-mono">HSL</div>
          <div className="aa-bar aa-hsl" />
          <p>看起来忽亮忽暗</p>
        </div>
        <div className="card aa-box">
          <div className="label-mono">OKLCH</div>
          <div className="aa-bar aa-oklch" />
          <p>亮度感知更一致</p>
        </div>
      </div>
    </div>
  );
}
