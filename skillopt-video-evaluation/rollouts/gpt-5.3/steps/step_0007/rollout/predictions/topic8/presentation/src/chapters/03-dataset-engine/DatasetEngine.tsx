import type { ChapterStepProps } from "../../registry/types";
import "./DatasetEngine.css";

export default function DatasetEngineChapter({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="de-scene scene-pad">
        <div className="kicker">EASY DATASET 1.7</div>
        <div className="de-hero card">
          <div className="hero-num">1.7.0</div>
          <p>自动化生成测试集 + 可视化评估</p>
        </div>
      </div>
    );
  }
  if (step === 1) {
    return (
      <div className="de-scene scene-pad">
        <div className="kicker">SOURCE TO QUESTION</div>
        <div className="de-flow card">
          <span>PDF / DOCX</span><b>→</b><span>Chunk</span><b>→</b><span>Prompt</span><b>→</b><span>题目生成</span>
        </div>
      </div>
    );
  }
  if (step === 2) {
    return (
      <div className="de-scene scene-pad">
        <div className="kicker">QUESTION COVERAGE</div>
        <div className="de-matrix card">
          <div>判断</div><div>单选</div><div>多选</div><div>简答</div><div>开放</div>
        </div>
      </div>
    );
  }
  return (
    <div className="de-scene scene-pad">
      <div className="kicker">DATASET OPS</div>
      <div className="de-ops">
        <div className="card de-op"><b>导入</b><span>JSON / XLS / XLSX</span></div>
        <div className="card de-op"><b>变体</b><span>训练集转评估集</span></div>
        <div className="card de-op"><b>导出</b><span>跨系统复用</span></div>
      </div>
    </div>
  );
}
