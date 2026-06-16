import type { ChapterStepProps } from "../../registry/types";
import "./SkillSystem.css";

export default function SkillSystemChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="t6s-scene scene-pad">
        <div className="t6s-kicker">WHY SKILL</div>
        <div className="t6s-contrast">
          <div className="card t6s-cell"><h3>随手 prompt</h3><p>结果波动大、难复现</p></div>
          <div className="card t6s-cell t6s-cell-accent"><h3>工程化流程</h3><p>稳定、可迭代、可团队协作</p></div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="t6s-scene scene-pad">
        <div className="t6s-kicker">PIPELINE</div>
        <div className="t6s-flow">
          <span>识别模式</span><span>需求分类</span><span>选模板</span><span>填参数</span><span>渲染 prompt</span><span>执行生图</span>
        </div>
      </div>
    );
  }

  if (step === 2) {
    return (
      <div className="t6s-scene scene-pad">
        <div className="t6s-kicker">TEMPLATE SCALE</div>
        <div className="t6s-stats">
          <div><b className="hero-num">18</b><small>大类</small></div>
          <div><b className="hero-num">79</b><small>模板</small></div>
          <div><b className="hero-num">100+</b><small>可复用案例</small></div>
        </div>
      </div>
    );
  }

  return (
    <div className="t6s-scene scene-pad">
      <div className="t6s-kicker">MODES A / B / C</div>
      <div className="t6s-modes">
        <div className="card t6s-mode"><h4>Mode A</h4><p>有 API Key：全自动出图</p></div>
        <div className="card t6s-mode"><h4>Mode B</h4><p>宿主原生生图：Skill 负责模板 + prompt</p></div>
        <div className="card t6s-mode"><h4>Mode C</h4><p>无工具无 Key：输出高质量提示词</p></div>
      </div>
    </div>
  );
}
