"""
Test ONNX Runtime for ZImage transformer acceleration
Step 1: Check environment
Step 2: Export transformer to ONNX
Step 3: Benchmark with ORT
"""
import sys, time, torch, gc, os
import numpy as np

print("=" * 60)
print("ZImage ONNX Runtime Test")
print("=" * 60)

# === Step 1: Environment Check ===
print("\n[Step 1] Environment Check")
print(f"  torch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

try:
    import onnx
    print(f"  onnx: {onnx.__version__}")
except ImportError:
    print("  onnx: NOT INSTALLED")
    print("  Installing onnx...")
    os.system("pip install onnx")
    import onnx
    print(f"  onnx: {onnx.__version__}")

try:
    import onnxruntime as ort
    print(f"  onnxruntime: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"  ORT providers: {providers}")
except ImportError:
    print("  onnxruntime: NOT INSTALLED")
    print("  Installing onnxruntime-gpu...")
    os.system("pip install onnxruntime-gpu")
    import onnxruntime as ort
    print(f"  onnxruntime: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"  ORT providers: {providers}")

# === Step 2: Load pipeline and try ONNX export of transformer ===
print("\n[Step 2] Load ZImage pipeline")
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/home/lixinqian/jinjin/Z-Image",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe = pipe.to("cuda")

transformer = pipe.transformer
transformer.eval()

print(f"  Transformer type: {type(transformer).__name__}")
print(f"  Transformer dtype: {next(transformer.parameters()).dtype}")

# === Step 2b: Try torch.onnx.export ===
print("\n[Step 2b] Attempting ONNX export of transformer...")
print("  First, doing a test forward pass to understand input shapes...")

# Do a single inference to capture transformer inputs
PROMPT = "a futuristic cityscape at sunset, highly detailed, 8k"
NEG_PROMPT = "blurry, low quality"

# We need to trace what inputs the transformer receives
# Let's use a hook to capture inputs
captured_inputs = {}

original_forward = transformer.forward.__func__ if hasattr(transformer.forward, '__func__') else None

class InputCapture:
    def __init__(self, model):
        self.model = model
        self.captured_args = None
        self.captured_kwargs = None
        self.original_forward = model.forward

    def hook_forward(self, *args, **kwargs):
        self.captured_args = args
        self.captured_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(f"  Captured {len(args)} positional args, {len(self.captured_kwargs)} kwargs")
        for k, v in self.captured_kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"    {k}: type={type(v).__name__}, value={v}")
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                print(f"    arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        return self.original_forward(*args, **kwargs)

    def install(self):
        self.model.forward = self.hook_forward

    def restore(self):
        self.model.forward = self.original_forward

capture = InputCapture(transformer)
capture.install()

print("  Running single inference to capture transformer inputs...")
try:
    with torch.no_grad():
        result = pipe(
            PROMPT,
            negative_prompt=NEG_PROMPT,
            height=768, width=1344,
            num_inference_steps=2,  # minimal steps just to capture
            guidance_scale=4.0,
            generator=torch.Generator("cuda").manual_seed(42),
        )
    print("  ✅ Captured transformer inputs successfully")
except Exception as e:
    print(f"  ❌ Failed to capture: {e}")

capture.restore()

# === Step 2c: Try ONNX export ===
if capture.captured_kwargs:
    print("\n[Step 2c] Attempting ONNX export...")

    # Create dummy inputs matching captured shapes
    dummy_inputs = {}
    input_names = []
    dynamic_axes = {}

    # Process captured kwargs
    for k, v in capture.captured_kwargs.items():
        if isinstance(v, torch.Tensor):
            dummy_inputs[k] = v.clone()
            input_names.append(k)
            # First dim is usually batch
            dynamic_axes[k] = {0: 'batch'}

    print(f"  Input names: {input_names}")

    onnx_path = "/tmp/zimage_transformer.onnx"

    try:
        # Convert to float32 for ONNX export (BF16 not well supported)
        transformer_f32 = transformer.float()
        dummy_inputs_f32 = {k: v.float() if v.dtype == torch.bfloat16 else v
                           for k, v in dummy_inputs.items()}

        print("  Exporting with torch.onnx.export (dynamo=False)...")
        torch.onnx.export(
            transformer_f32,
            (),  # args
            onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            kwargs=dummy_inputs_f32,
        )
        print(f"  ✅ ONNX export succeeded! File: {onnx_path}")

        # Check file size
        fsize = os.path.getsize(onnx_path) / (1024**3)
        print(f"  ONNX file size: {fsize:.2f} GB")

    except Exception as e:
        print(f"  ❌ Standard ONNX export failed: {e}")
        print()
        print("  Trying torch.onnx.dynamo_export (torch 2.x)...")
        try:
            export_output = torch.onnx.dynamo_export(
                transformer_f32,
                **dummy_inputs_f32,
            )
            export_output.save(onnx_path)
            print(f"  ✅ Dynamo ONNX export succeeded! File: {onnx_path}")
            fsize = os.path.getsize(onnx_path) / (1024**3)
            print(f"  ONNX file size: {fsize:.2f} GB")
        except Exception as e2:
            print(f"  ❌ Dynamo ONNX export also failed: {e2}")
            print("\n  === Alternative: Try torch.compile with onnxrt backend ===")
            print("  This avoids explicit export — uses ORT as a torch.compile backend")

            try:
                # Restore to bfloat16
                transformer_bf16 = transformer.to(torch.bfloat16)
                pipe.transformer = torch.compile(
                    transformer_bf16,
                    backend="onnxrt",
                    mode="reduce-overhead",
                )
                print("  ✅ torch.compile(backend='onnxrt') succeeded!")

                # Benchmark
                print("\n  Benchmarking ORT backend...")
                GEN_KWARGS = dict(
                    height=768, width=1344,
                    num_inference_steps=9,
                    guidance_scale=4.0,
                )

                # Warmup
                for i in range(3):
                    with torch.no_grad():
                        _ = pipe(
                            PROMPT, negative_prompt=NEG_PROMPT,
                            generator=torch.Generator("cuda").manual_seed(42),
                            **GEN_KWARGS,
                        )
                    print(f"    Warmup {i+1}/3 done")

                # Benchmark
                times = []
                for i in range(3):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = pipe(
                            PROMPT, negative_prompt=NEG_PROMPT,
                            generator=torch.Generator("cuda").manual_seed(42),
                            **GEN_KWARGS,
                        )
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 1000
                    times.append(ms)
                    print(f"    ORT run {i+1}: {ms:.0f}ms")

                avg = sum(times) / len(times)
                print(f"    ORT avg: {avg:.0f}ms")

            except Exception as e3:
                print(f"  ❌ torch.compile(backend='onnxrt') failed: {e3}")
                print("\n  All ONNX approaches exhausted.")

# === Step 3: If ONNX export succeeded, benchmark with ORT session ===
onnx_path = "/tmp/zimage_transformer.onnx"
if os.path.exists(onnx_path):
    print("\n[Step 3] Benchmarking with ONNX Runtime session...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers_to_try = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers_to_try.append('CUDAExecutionProvider')
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers_to_try.append('TensorrtExecutionProvider')
        providers_to_try.append('CPUExecutionProvider')

        print(f"  Using providers: {providers_to_try}")
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers_to_try)

        print(f"  Session created successfully")
        print(f"  Input names: {[i.name for i in session.get_inputs()]}")
        print(f"  Output names: {[o.name for o in session.get_outputs()]}")

    except Exception as e:
        print(f"  ❌ ORT session creation failed: {e}")

print("\n" + "=" * 60)
print("ONNX Runtime test complete")
print("=" * 60)
