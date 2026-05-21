#!/usr/bin/env python3
"""
诊断 TRT fallback 比例
=====================
验证假设 H1: torch.compile(backend='torch_tensorrt') 是否大量 fallback 到 PyTorch eager。

输出:
  - TRT 实际编译的子图数量和算子数
  - Fallback 到 PyTorch 的算子数
  - 每个子图的算子列表
"""

import sys
import os
import io
import logging
import time
import torch

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

def setup_trt_logging():
    """开启 torch_tensorrt 的 debug 日志，捕获到字符串"""
    import torch_tensorrt

    # 尝试不同版本的 API 设置日志级别
    try:
        torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
    except AttributeError:
        try:
            torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.DEBUG)
        except AttributeError:
            pass
    # 尝试新版 API
    try:
        torch_tensorrt.logging.set_log_level(torch_tensorrt.logging.Level.Debug)
    except AttributeError:
        pass
    try:
        torch_tensorrt.logging.set_log_level(torch_tensorrt.logging.Level.DEBUG)
    except AttributeError:
        pass

    print(f"  torch_tensorrt.logging 可用属性: {[x for x in dir(torch_tensorrt.logging) if not x.startswith('_')]}")

    # 捕获 Python logging
    log_capture = io.StringIO()

    for logger_name in ["torch_tensorrt", "torch._dynamo", "torch._inductor",
                        "torch_tensorrt.dynamo", "torch_tensorrt.dynamo.conversion"]:
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        lg.addHandler(handler)

    return log_capture


def analyze_compilation_log(log_text):
    """分析编译日志，统计 TRT vs fallback"""
    lines = log_text.split('\n')

    trt_compiled = []
    fallback = []
    graph_breaks = []
    unsupported_ops = []

    for line in lines:
        lower = line.lower()
        if 'falling back' in lower or 'fallback' in lower:
            fallback.append(line.strip())
        if 'graph break' in lower or 'graph_break' in lower:
            graph_breaks.append(line.strip())
        if 'unsupported' in lower or 'not supported' in lower:
            unsupported_ops.append(line.strip())
        if 'trt engine' in lower or 'tensorrt engine' in lower or 'compiled successfully' in lower:
            trt_compiled.append(line.strip())

    return {
        'trt_compiled': trt_compiled,
        'fallback': fallback,
        'graph_breaks': graph_breaks,
        'unsupported_ops': unsupported_ops,
    }


def diagnose_with_torch_compile():
    """方法1: 用 torch.compile + 日志分析"""
    print("=" * 70)
    print("方法1: torch.compile(backend='torch_tensorrt') 日志分析")
    print("=" * 70)

    import torch_tensorrt
    print(f"  torch_tensorrt: {torch_tensorrt.__version__}")

    log_capture = setup_trt_logging()

    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    # 加载 TRT-compatible transformer
    orig_state = pipe.transformer.state_dict()
    orig_config = pipe.transformer.config

    sys.path.insert(0, "/tmp")
    from transformer_z_image_trt import ZImageTransformer2DModel as TRTModel
    trt_transformer = TRTModel(**orig_config)
    trt_transformer.load_state_dict(orig_state, strict=True)
    trt_transformer.to(device=DEVICE, dtype=DTYPE)
    trt_transformer.eval()

    # 用 torch.compile 编译
    print("\n  编译中 (这会比较慢)...")
    compiled = torch.compile(trt_transformer, backend="torch_tensorrt")
    pipe.transformer = compiled

    # 跑一次触发编译
    print("  运行一次推理触发编译...")
    gen = torch.Generator(DEVICE).manual_seed(42)
    t0 = time.time()
    _ = pipe(
        prompt="A red apple on a white table",
        height=768, width=1344,
        guidance_scale=0, num_inference_steps=1,
        generator=gen,
    )
    t1 = time.time()
    print(f"  编译+推理完成: {t1-t0:.1f}s")

    # 分析日志
    log_text = log_capture.getvalue()
    with open("/tmp/trt_debug_log.txt", "w") as f:
        f.write(log_text)
    print(f"\n  完整日志已保存: /tmp/trt_debug_log.txt ({len(log_text)} chars, {log_text.count(chr(10))} lines)")

    results = analyze_compilation_log(log_text)

    print(f"\n  === 分析结果 ===")
    print(f"  TRT 编译成功的引擎/子图: {len(results['trt_compiled'])}")
    print(f"  Fallback 到 PyTorch:     {len(results['fallback'])}")
    print(f"  Graph breaks:            {len(results['graph_breaks'])}")
    print(f"  不支持的算子:            {len(results['unsupported_ops'])}")

    if results['fallback']:
        print(f"\n  --- Fallback 详情 (前20条) ---")
        for line in results['fallback'][:20]:
            print(f"    {line[:120]}")

    if results['unsupported_ops']:
        print(f"\n  --- 不支持的算子 (前20条) ---")
        for line in results['unsupported_ops'][:20]:
            print(f"    {line[:120]}")

    if results['graph_breaks']:
        print(f"\n  --- Graph breaks (前10条) ---")
        for line in results['graph_breaks'][:10]:
            print(f"    {line[:120]}")

    return log_text


def diagnose_graph_structure():
    """方法2: 用 torch._dynamo.explain 分析 graph breaks"""
    print("\n" + "=" * 70)
    print("方法2: torch._dynamo.explain() 分析 graph breaks")
    print("=" * 70)

    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    orig_state = pipe.transformer.state_dict()
    orig_config = pipe.transformer.config

    sys.path.insert(0, "/tmp")
    from transformer_z_image_trt import ZImageTransformer2DModel as TRTModel
    trt_transformer = TRTModel(**orig_config)
    trt_transformer.load_state_dict(orig_state, strict=True)
    trt_transformer.to(device=DEVICE, dtype=DTYPE)
    trt_transformer.eval()

    # 构造 dummy inputs
    print("  构造 dummy inputs...")
    batch = 1
    seq_len = 256  # text tokens
    img_seq_len = 1008  # 1344/14 * 768/14 ≈ 96*54.8 ≈ 5260... let's use actual
    # ZImage: height=768, width=1344, patch_size=2, in_channels=64
    # hidden_size from config
    hidden_size = orig_config.get('inner_dim', 3072)

    # 尝试用 explain
    try:
        explanation = torch._dynamo.explain(trt_transformer)
        print(f"\n  === Dynamo Explain ===")
        print(f"  Graph count: {explanation.graph_count}")
        print(f"  Graph break count: {explanation.graph_break_count}")
        print(f"  Op count (per graph): {explanation.ops_per_graph}")

        if explanation.break_reasons:
            print(f"\n  --- Break reasons ---")
            for i, reason in enumerate(explanation.break_reasons[:10]):
                print(f"    [{i}] {str(reason)[:200]}")
    except Exception as e:
        print(f"  explain() 需要实际输入，跳过: {type(e).__name__}: {str(e)[:100]}")
        print("  (explain 需要实际调用 transformer forward，需要正确的输入 tensor)")


def diagnose_trt_partitioning():
    """方法3: 直接查看 TRT 分区信息"""
    print("\n" + "=" * 70)
    print("方法3: TRT partitioning 分析")
    print("=" * 70)

    try:
        import torch_tensorrt
        from torch_tensorrt.dynamo import partitioning

        print("  检查 torch_tensorrt.dynamo.partitioning 模块...")
        members = [m for m in dir(partitioning) if not m.startswith('_')]
        print(f"  可用方法: {members[:20]}")
    except Exception as e:
        print(f"  无法导入 partitioning: {e}")

    # 检查环境变量控制
    print("\n  === 有用的环境变量 ===")
    print("  TORCH_TENSORRT_LOG_LEVEL=DEBUG  — 启用 C++ 层 TRT 日志")
    print("  TORCH_COMPILE_DEBUG=1           — 输出 dynamo 编译调试信息到 /tmp/torchinductor_*/")
    print("  TORCHDYNAMO_VERBOSE=1           — 显示 graph break 原因")


if __name__ == "__main__":
    print("TensorRT Fallback 诊断")
    print("目标: 验证 H1 — TRT 是否大量 fallback 到 PyTorch eager")
    print()

    # 方法1: 最重要 — 实际编译并分析日志
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    log_text = diagnose_with_torch_compile()

    # 释放显存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 方法3: 查看可用的分析工具
    diagnose_trt_partitioning()

    # 如果日志太少，建议手动方法
    print("\n" + "=" * 70)
    print("手动诊断建议")
    print("=" * 70)
    print("""
  如果上面的自动分析日志不够，请手动运行:

  1. 查看完整 TRT debug 日志:
     cat /tmp/trt_debug_log.txt | grep -i "fallback\\|unsupported\\|graph.break" | head -50

  2. 用 TORCH_COMPILE_DEBUG 获取更详细的编译信息:
     TORCH_COMPILE_DEBUG=1 python -c "
     import torch, sys
     sys.path.insert(0, '/tmp')
     from transformer_z_image_trt import ZImageTransformer2DModel
     # ... 编译并运行
     "
     然后检查 /tmp/torchinductor_*/ 目录下的 debug 文件

  3. 关键指标:
     - 如果 fallback 数量 >> TRT compiled 数量 → H1 确认，TRT 基本没用
     - 如果 graph breaks > 10 → H3 确认，计算图碎片化严重
     - 如果 TRT 编译了大部分算子但仍无加速 → H5 确认，硬件瓶颈
""")
