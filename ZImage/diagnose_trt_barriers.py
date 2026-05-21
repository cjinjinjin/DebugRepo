#!/usr/bin/env python3
"""
ZImage TensorRT 障碍全量诊断脚本
=================================
目标：收集 ZImage transformer 通往 TensorRT 的所有障碍，不修复，只记录。

运行方式（在 A6000 Docker 内）：
    python /tmp/diagnose_trt_barriers.py

前置条件：
    pip install torch-tensorrt  # 如果还没装的话
"""

import sys
import os
import time
import traceback
import json
from pathlib import Path

# ── UTF-8 stdout ──
if sys.platform == "win32":
    import io
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import torch.nn as nn

DIVIDER = "=" * 70
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

barriers = []  # 收集所有障碍


def log_barrier(phase, description, error_msg, severity="BLOCKER"):
    """记录一个障碍"""
    entry = {
        "phase": phase,
        "severity": severity,
        "description": description,
        "error": str(error_msg)[:500],
    }
    barriers.append(entry)
    print(f"\n  [BARRIER-{len(barriers):02d}] ({severity}) {description}")
    print(f"    Error: {str(error_msg)[:200]}")


def phase1_inspect_inputs():
    """Phase 1: 加载模型，做一次真实 forward，记录所有输入的 type/shape/dtype"""
    print(f"\n{DIVIDER}")
    print("Phase 1: Inspect ZImage transformer inputs")
    print(DIVIDER)

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)
    transformer = pipe.transformer

    # 记录 forward 签名
    import inspect
    sig = inspect.signature(transformer.forward)
    print(f"\n  forward signature: {sig}")
    print(f"\n  Parameters:")
    for name, param in sig.parameters.items():
        print(f"    {name}: {param.annotation} (default={param.default})")

    # Hook transformer.forward 来捕获真实输入
    captured_inputs = {}

    original_forward = transformer.forward

    def capturing_forward(*args, **kwargs):
        # 记录所有参数
        sig_params = list(inspect.signature(original_forward).parameters.keys())
        all_args = {}
        for i, arg in enumerate(args):
            if i < len(sig_params):
                all_args[sig_params[i]] = arg
        all_args.update(kwargs)

        for name, val in all_args.items():
            captured_inputs[name] = describe_value(val)

        return original_forward(*args, **kwargs)

    transformer.forward = capturing_forward

    # 做一次真实推理
    print("\n  Running one inference to capture real inputs...")
    gen = torch.Generator(DEVICE).manual_seed(42)
    _ = pipe(
        prompt="A red apple on a white table",
        height=768, width=1344,
        guidance_scale=0, num_inference_steps=2,  # 只跑2步，够了
        generator=gen,
    )

    transformer.forward = original_forward  # 恢复

    print("\n  Captured transformer inputs:")
    for name, desc in captured_inputs.items():
        print(f"    {name}: {desc}")

    return pipe, transformer, captured_inputs


def describe_value(val, depth=0):
    """递归描述一个值的类型/shape/dtype"""
    indent = "  " * depth
    if isinstance(val, torch.Tensor):
        return f"Tensor(shape={list(val.shape)}, dtype={val.dtype}, device={val.device})"
    elif isinstance(val, list):
        items = [describe_value(v, depth + 1) for v in val]
        return f"list[{len(val)}] = [{', '.join(items)}]"
    elif isinstance(val, tuple):
        items = [describe_value(v, depth + 1) for v in val]
        return f"tuple({len(val)}) = ({', '.join(items)})"
    elif isinstance(val, dict):
        items = {k: describe_value(v, depth + 1) for k, v in val.items()}
        return f"dict({items})"
    elif val is None:
        return "None"
    elif isinstance(val, bool):
        return f"bool={val}"
    elif isinstance(val, (int, float)):
        return f"{type(val).__name__}={val}"
    else:
        return f"{type(val).__name__}"


def phase2_check_complex64(transformer):
    """Phase 2: 检查 complex64 使用"""
    print(f"\n{DIVIDER}")
    print("Phase 2: Scan for complex64 operations")
    print(DIVIDER)

    import diffusers.models.transformers.transformer_z_image as zt
    source_file = inspect.getfile(zt)
    print(f"  Source: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        source = f.read()

    complex_patterns = [
        "complex64", "complex128", "torch.polar", "view_as_complex",
        "view_as_real", "torch.complex",
    ]

    found = []
    for i, line in enumerate(source.split("\n"), 1):
        for pat in complex_patterns:
            if pat in line and not line.strip().startswith("#"):
                found.append((i, pat, line.strip()))

    if found:
        print(f"\n  Found {len(found)} complex-related lines:")
        for lineno, pat, line in found:
            print(f"    L{lineno} [{pat}]: {line[:100]}")
        log_barrier("complex64", f"{len(found)} lines use complex operations (RoPE)",
                    "torch.polar / view_as_complex / view_as_real in RoPE embedding",
                    severity="BLOCKER - needs Real RoPE patch")
    else:
        print("  No complex64 found (Real RoPE may already be applied)")


def phase3_check_scatter(transformer):
    """Phase 3: 定位所有可能产生 aten.scatter 的 index assignment"""
    print(f"\n{DIVIDER}")
    print("Phase 3: Scan for index assignment patterns (→ aten.scatter)")
    print(DIVIDER)

    import diffusers.models.transformers.transformer_z_image as zt
    source_file = inspect.getfile(zt)

    with open(source_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 找 tensor[...] = value 模式
    import re
    scatter_patterns = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # 匹配: xxx[yyy] = zzz 或 xxx[yyy, :zzz] = 等
        if re.search(r'\w+\[.*\]\s*=\s*(?!.*==)', stripped):
            # 排除 dict 赋值和普通变量赋值
            if not re.match(r'^\w+\s*=', stripped):  # 不是 var = 开头
                scatter_patterns.append((i, stripped))
            elif "[" in stripped.split("=")[0]:  # 左边有 [
                scatter_patterns.append((i, stripped))

    if scatter_patterns:
        print(f"\n  Found {len(scatter_patterns)} potential scatter-inducing lines:")
        for lineno, line in scatter_patterns:
            print(f"    L{lineno}: {line[:120]}")
        log_barrier("scatter", f"{len(scatter_patterns)} index assignments may become aten.scatter",
                    "tensor[i, :seq_len] = 1 patterns in _prepare_sequence/_build_unified_sequence",
                    severity="BLOCKER - needs vectorized rewrite")
    else:
        print("  No index assignment patterns found")


def phase4_check_dynamic_control_flow(transformer):
    """Phase 4: 检查动态控制流（for 循环、if 依赖 tensor 值）"""
    print(f"\n{DIVIDER}")
    print("Phase 4: Scan for dynamic control flow")
    print(DIVIDER)

    import diffusers.models.transformers.transformer_z_image as zt
    source_file = inspect.getfile(zt)

    with open(source_file, "r", encoding="utf-8") as f:
        source = f.read()

    issues = []

    # for 循环遍历动态长度
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "for " in stripped and "enumerate" in stripped and ("seqlen" in stripped or "feats" in stripped):
            issues.append((i, "dynamic-for", stripped))
        if "if all(" in stripped or "if any(" in stripped:
            issues.append((i, "data-dependent-branch", stripped))
        if "with torch.device" in stripped:
            issues.append((i, "torch.device-context", stripped))

    if issues:
        print(f"\n  Found {len(issues)} potential control flow issues:")
        for lineno, kind, line in issues:
            print(f"    L{lineno} [{kind}]: {line[:120]}")
        for lineno, kind, line in issues:
            log_barrier("control_flow", f"{kind} at L{lineno}",
                        line[:200], severity="WARNING - may need static rewrite")
    else:
        print("  No dynamic control flow issues found")


def phase5_try_torch_export(pipe, transformer):
    """Phase 5: 尝试 torch.export（strict + non-strict），收集所有错误"""
    print(f"\n{DIVIDER}")
    print("Phase 5: Attempt torch.export")
    print(DIVIDER)

    # 构造一个简化的输入
    # 先做一次 forward 来获取真实输入
    print("\n  Preparing real inputs via pipeline encode...")

    try:
        # 用 pipeline 的 encode_prompt 获取 text embeddings
        prompt_embeds = pipe.encode_prompt(
            prompt="A red apple",
            do_classifier_free_guidance=False,
            device=DEVICE,
        )
        if isinstance(prompt_embeds, tuple):
            cap_feats = prompt_embeds[0]  # 通常是 prompt_embeds
        else:
            cap_feats = prompt_embeds
        print(f"    cap_feats type: {type(cap_feats)}, {describe_value(cap_feats)}")
    except Exception as e:
        print(f"    encode_prompt failed: {e}")
        log_barrier("torch_export", "Cannot prepare inputs via encode_prompt", e)
        return

    # 由于输入是嵌套 list，torch.export 大概率直接失败
    # 先尝试 non-strict
    print("\n  5a. torch.export (non-strict)...")
    try:
        # 我们需要知道真实的调用方式
        # 但由于输入是 list of tensors，直接尝试
        exported = torch.export.export(
            transformer,
            args=(),
            kwargs={},  # 空的，肯定失败，但看错误信息
            strict=False,
        )
        print("    Unexpected success!")
    except TypeError as e:
        print(f"    TypeError (expected - need real inputs): {str(e)[:200]}")
        log_barrier("torch_export", "torch.export needs flat tensor inputs, ZImage uses nested lists",
                    e, severity="BLOCKER - needs forward signature rewrite")
    except Exception as e:
        print(f"    Failed: {type(e).__name__}: {str(e)[:300]}")
        log_barrier("torch_export", f"torch.export failed: {type(e).__name__}", e)


def phase6_try_torch_compile_trt(pipe, transformer):
    """Phase 6: 尝试 torch.compile(backend='torch_tensorrt')"""
    print(f"\n{DIVIDER}")
    print("Phase 6: Attempt torch.compile with torch_tensorrt backend")
    print(DIVIDER)

    # 检查 torch_tensorrt 是否可用
    try:
        import torch_tensorrt
        print(f"  torch_tensorrt version: {torch_tensorrt.__version__}")
    except ImportError:
        log_barrier("torch_tensorrt", "torch_tensorrt not installed",
                    "pip install torch-tensorrt", severity="SETUP")
        print("  torch_tensorrt not installed, skipping")
        return

    # 检查 TensorRT runtime
    try:
        import tensorrt as trt
        print(f"  TensorRT version: {trt.__version__}")
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        if builder is None:
            log_barrier("tensorrt", "trt.Builder() returned None", "TRT runtime issue")
        else:
            print(f"  trt.Builder OK")
    except Exception as e:
        print(f"  TensorRT runtime issue: {e}")
        log_barrier("tensorrt", "TensorRT runtime check failed", e, severity="SETUP")

    # 尝试 compile
    print("\n  6a. torch.compile(backend='torch_tensorrt', fullgraph=False)...")
    compiled = torch.compile(transformer, backend="torch_tensorrt")

    print("  6b. Running inference with compiled model...")
    try:
        gen = torch.Generator(DEVICE).manual_seed(42)
        _ = pipe(
            prompt="A red apple on a white table",
            height=768, width=1344,
            guidance_scale=0, num_inference_steps=1,
            generator=gen,
        )
        print("    SUCCESS! (unexpected)")
    except Exception as e:
        full_tb = traceback.format_exc()
        print(f"    Failed: {type(e).__name__}")
        print(f"    {str(e)[:500]}")

        # 分析错误类型
        error_str = str(e) + full_tb
        if "complex" in error_str.lower():
            log_barrier("trt_compile", "complex64 dtype not supported by TRT", e)
        if "scatter" in error_str.lower():
            log_barrier("trt_compile", "aten.scatter with mixed types", e)
        if "unsupported" in error_str.lower():
            log_barrier("trt_compile", "Unsupported operation in TRT", e)
        if "dtype" in error_str.lower() or "type" in error_str.lower():
            log_barrier("trt_compile", "Type mismatch in TRT conversion", e)

        # 保存完整 traceback
        tb_path = "/tmp/trt_compile_traceback.txt"
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(full_tb)
        print(f"    Full traceback saved to {tb_path}")

    # 恢复原始 transformer
    pipe.transformer = transformer


def phase7_try_compile_trt_fullgraph(pipe, transformer):
    """Phase 7: fullgraph=True 测试"""
    print(f"\n{DIVIDER}")
    print("Phase 7: torch.compile(backend='torch_tensorrt', fullgraph=True)")
    print(DIVIDER)

    try:
        import torch_tensorrt
    except ImportError:
        print("  Skipped (torch_tensorrt not installed)")
        return

    compiled = torch.compile(transformer, backend="torch_tensorrt",
                             options={"truncate_double": True},
                             fullgraph=True)
    pipe.transformer = compiled

    try:
        gen = torch.Generator(DEVICE).manual_seed(42)
        _ = pipe(
            prompt="A red apple",
            height=768, width=1344,
            guidance_scale=0, num_inference_steps=1,
            generator=gen,
        )
        print("    SUCCESS!")
    except Exception as e:
        full_tb = traceback.format_exc()
        print(f"    Failed: {type(e).__name__}: {str(e)[:500]}")
        log_barrier("trt_fullgraph", f"fullgraph=True failed: {type(e).__name__}", e)

        tb_path = "/tmp/trt_fullgraph_traceback.txt"
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(full_tb)
        print(f"    Full traceback saved to {tb_path}")

    pipe.transformer = transformer


def phase8_flux_baseline():
    """Phase 8: Flux transformer 做同样的 torch.compile TRT 测试，确认工具链正常"""
    print(f"\n{DIVIDER}")
    print("Phase 8: Flux baseline (verify TRT toolchain works)")
    print(DIVIDER)

    try:
        import torch_tensorrt
    except ImportError:
        print("  Skipped (torch_tensorrt not installed)")
        return

    try:
        from diffusers import FluxPipeline
        print("  Loading Flux pipeline (this may take a while)...")
        flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=DTYPE,
        )
        flux_pipe.to(DEVICE)
    except Exception as e:
        print(f"  Cannot load Flux: {e}")
        print("  Skipping Flux baseline (model not available)")
        return

    flux_transformer = flux_pipe.transformer
    compiled = torch.compile(flux_transformer, backend="torch_tensorrt")
    flux_pipe.transformer = compiled

    try:
        gen = torch.Generator(DEVICE).manual_seed(42)
        _ = flux_pipe(
            prompt="A red apple",
            height=256, width=256,  # 小分辨率快速测试
            guidance_scale=0, num_inference_steps=1,
            generator=gen,
        )
        print("    Flux + TRT: SUCCESS! (toolchain confirmed working)")
    except Exception as e:
        print(f"    Flux + TRT also failed: {type(e).__name__}: {str(e)[:300]}")
        print("    (This means the issue may be TRT toolchain, not ZImage-specific)")
        log_barrier("flux_baseline", "Flux + TRT also failed — toolchain issue", e,
                    severity="INFO")
    finally:
        del flux_pipe
        torch.cuda.empty_cache()


def phase9_submodule_isolation(transformer):
    """Phase 9: 逐子模块测试 TRT 兼容性"""
    print(f"\n{DIVIDER}")
    print("Phase 9: Sub-module TRT compatibility scan")
    print(DIVIDER)

    try:
        import torch_tensorrt
    except ImportError:
        print("  Skipped (torch_tensorrt not installed)")
        return

    # 测试单个 transformer block
    block = transformer.layers[0]
    print(f"\n  Testing single ZImageTransformerBlock...")
    print(f"    Block type: {type(block).__name__}")
    print(f"    Block params: {sum(p.numel() for p in block.parameters()) / 1e6:.1f}M")

    # 构造 block 的输入
    # ZImageTransformerBlock.forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None)
    seq_len = 100
    hidden_dim = transformer.config.get("inner_dim", 3840) if hasattr(transformer, "config") else 3840

    try:
        # 尝试获取 hidden_dim
        if hasattr(transformer, "inner_dim"):
            hidden_dim = transformer.inner_dim
        elif hasattr(transformer.layers[0], "attention"):
            attn = transformer.layers[0].attention
            if hasattr(attn, "inner_dim"):
                hidden_dim = attn.inner_dim
        print(f"    hidden_dim: {hidden_dim}")
    except:
        pass

    x_test = torch.randn(1, seq_len, hidden_dim, dtype=DTYPE, device=DEVICE)
    c_test = torch.randn(1, 1, hidden_dim, dtype=DTYPE, device=DEVICE)

    # 先确认 eager forward 工作
    print("    Eager forward...")
    try:
        with torch.no_grad():
            out = block(x_test, c=c_test)
        print(f"    Eager OK, output: {describe_value(out)}")
    except Exception as e:
        print(f"    Eager forward failed: {e}")
        log_barrier("block_test", "Single block eager forward failed", e)
        return

    # torch.compile inductor（确认 compile 本身可以工作）
    print("\n    torch.compile(backend='inductor')...")
    try:
        compiled_block = torch.compile(block, backend="inductor")
        with torch.no_grad():
            out = compiled_block(x_test, c=c_test)
        print(f"    Inductor OK")
    except Exception as e:
        print(f"    Inductor failed: {e}")
        log_barrier("block_inductor", "Single block inductor compile failed", e)

    # torch.compile TRT
    print("\n    torch.compile(backend='torch_tensorrt')...")
    try:
        compiled_block = torch.compile(block, backend="torch_tensorrt")
        with torch.no_grad():
            out = compiled_block(x_test, c=c_test)
        print(f"    TRT block compile: SUCCESS!")
    except Exception as e:
        full_tb = traceback.format_exc()
        print(f"    TRT failed: {type(e).__name__}: {str(e)[:500]}")
        log_barrier("block_trt", f"Single block TRT failed: {type(e).__name__}", e)

        tb_path = "/tmp/trt_block_traceback.txt"
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(full_tb)
        print(f"    Full traceback saved to {tb_path}")

    # 测试 attention 子模块
    print("\n    Testing attention sub-module only...")
    attn = block.attention
    try:
        compiled_attn = torch.compile(attn, backend="torch_tensorrt")
        # 需要知道 attn 的输入格式
        print(f"    Attention type: {type(attn).__name__}")
        # Attention.forward 需要 hidden_states + encoder_hidden_states 等
    except Exception as e:
        print(f"    Attention TRT: {type(e).__name__}: {str(e)[:200]}")

    # 测试 FFN 子模块
    print("\n    Testing feed_forward sub-module only...")
    ffn = block.feed_forward
    try:
        compiled_ffn = torch.compile(ffn, backend="torch_tensorrt")
        with torch.no_grad():
            ffn_out = compiled_ffn(x_test)
        print(f"    FFN TRT: SUCCESS!")
    except Exception as e:
        print(f"    FFN TRT: {type(e).__name__}: {str(e)[:200]}")
        log_barrier("ffn_trt", f"FFN TRT failed", e, severity="WARNING")


def print_summary():
    """打印所有障碍汇总"""
    print(f"\n{DIVIDER}")
    print("SUMMARY: All TensorRT Barriers")
    print(DIVIDER)

    if not barriers:
        print("\n  No barriers found! TRT should work.")
        return

    blockers = [b for b in barriers if "BLOCKER" in b["severity"]]
    warnings = [b for b in barriers if "WARNING" in b["severity"]]
    info = [b for b in barriers if b["severity"] in ("INFO", "SETUP")]

    print(f"\n  Total: {len(barriers)} barriers")
    print(f"    BLOCKERS: {len(blockers)}")
    print(f"    WARNINGS: {len(warnings)}")
    print(f"    INFO/SETUP: {len(info)}")

    if blockers:
        print(f"\n  {'─' * 50}")
        print("  BLOCKERS (must fix):")
        for i, b in enumerate(blockers, 1):
            print(f"\n    {i}. [{b['phase']}] {b['description']}")
            print(f"       Severity: {b['severity']}")
            print(f"       Error: {b['error'][:150]}")

    if warnings:
        print(f"\n  {'─' * 50}")
        print("  WARNINGS (may need fix):")
        for i, b in enumerate(warnings, 1):
            print(f"\n    {i}. [{b['phase']}] {b['description']}")
            print(f"       Error: {b['error'][:150]}")

    if info:
        print(f"\n  {'─' * 50}")
        print("  INFO:")
        for i, b in enumerate(info, 1):
            print(f"    {i}. [{b['phase']}] {b['description']}")

    # 保存 JSON
    report_path = "/tmp/trt_barriers_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(barriers, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved to {report_path}")


if __name__ == "__main__":
    import inspect

    print(DIVIDER)
    print("ZImage TensorRT Barrier Diagnosis")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(DIVIDER)

    # Phase 1: 加载模型，记录输入
    pipe, transformer, captured_inputs = phase1_inspect_inputs()

    # Phase 2: complex64 检查
    phase2_check_complex64(transformer)

    # Phase 3: scatter 检查
    phase3_check_scatter(transformer)

    # Phase 4: 动态控制流检查
    phase4_check_dynamic_control_flow(transformer)

    # Phase 5: torch.export 尝试
    phase5_try_torch_export(pipe, transformer)

    # Phase 6: torch.compile TRT（fullgraph=False）
    phase6_try_torch_compile_trt(pipe, transformer)

    # Phase 7: torch.compile TRT（fullgraph=True）
    phase7_try_compile_trt_fullgraph(pipe, transformer)

    # Phase 8: Flux baseline（可选，验证工具链）
    # 注释掉以节省时间，需要下载 Flux 模型
    # phase8_flux_baseline()

    # Phase 9: 子模块隔离测试
    phase9_submodule_isolation(transformer)

    # 汇总
    print_summary()

    print(f"\n{DIVIDER}")
    print("Diagnosis complete.")
    print(f"Total barriers: {len(barriers)}")
    print(f"Barrier report: /tmp/trt_barriers_report.json")
    if any("BLOCKER" in b["severity"] for b in barriers):
        print("Traceback files: /tmp/trt_*_traceback.txt")
    print(DIVIDER)
