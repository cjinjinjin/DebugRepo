import type { ChapterStepProps } from "../../registry/types";
import "./SkillRollout.css";

export default function SkillRolloutChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="sr-scene scene-pad">
        <div className="kicker">WHY EXTRACT A SKILL</div>
        <div className="sr-pain">
          <div className="card">访问门槛高</div>
          <div className="card">无 API 接入</div>
          <div className="card">行为不可定制</div>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="sr-scene scene-pad">
        <div className="sr-pipe card">
          <div>Prompt Engineering</div><span>→</span><div>Skill 封装</div><span>→</span><div>Codex / Cursor / Agent</div>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="sr-scene scene-pad">
        <img className="sr-img" src={`${import.meta.env.BASE_URL}assets/wechat-03.jpg`} alt="wechat" />
        <div className="sr-dual">
          <div className="card"><div className="label-mono">增强 A</div><p>先宣告设计系统</p></div>
          <div className="card"><div className="label-mono">增强 B</div><p>尽早交付 v0 半成品</p></div>
        </div>
      </div>
    );
  }
  return (
    <div className="sr-scene scene-pad">
      <div className="sr-end">
        <div className="sr-quote pull-quote">真正可迁移的，不是某个模型，而是可复用的方法。</div>
        <div className="sr-next card">
          <div className="label-mono">NEXT</div>
          <ul>
            <li>继续补真实素材</li>
            <li>提取 narrations 音频段</li>
            <li>开启 auto=1 一镜录屏</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
