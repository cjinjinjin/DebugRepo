"""
GPT-5.5 two-step inference benchmark via Papyrus API.
Uses the same system prompts as Gemma4 two-step pipeline.
Measures per-request latency and reports p50/p95/p99.

Usage:
  python test_gpt55_two_step_benchmark.py
"""

import json
import re
import time
import statistics
import numpy as np
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.identity import InteractiveBrowserCredential
import sys

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
PAPYRUS_ENDPOINT = "https://westus2.papyrus.binginternal.com/chat/completions"
VERIFY_SCOPE = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

# ---------------------------------------------------------------------------
# System prompts (same as Gemma4 two-step)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_STEP1_SCENES = """You are an expert Ad Creative Director specialized in high-performing Native Advertisement visuals.

Given a landing page URL and its content fields, generate 5 DIVERSE scene concepts for Native Ad images.

Each scene must use a DIFFERENT visual approach:
1. One close-up product/detail shot
2. One lifestyle scene with a person using/experiencing the product
3. One environmental/contextual setting showing the product in its natural habitat
4. One outcome/result-focused scene showing the benefit
5. One mood/atmosphere-driven composition

Each scene description should be a SHORT phrase (5-10 words) that captures the setting, subject, and mood.

Output exactly 5 scene descriptions (no reasoning, no thinking):
<Scene1>...</Scene1>
<Scene2>...</Scene2>
<Scene3>...</Scene3>
<Scene4>...</Scene4>
<Scene5>...</Scene5>"""

SYSTEM_PROMPT_STEP2_EXPAND = """You are an expert Ad Creative Director and Senior AI Image Prompt Engineer, specialized in high-performing Native Advertisement visuals.

Given a landing page and a scene concept, expand the scene into a detailed image generation prompt for a Native Ad.

The prompt must:
- Be 30-50 words
- Feel native and non-promotional
- Show the product outcome or value naturally in context
- Avoid stereotypes, text/logos in image, and stock-photo aesthetics
- Ensure correct anatomy, natural hands, sharp focus, clean composition

Output exactly one prompt (no reasoning, no thinking):
<Prompt>...</Prompt>"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_scenes(text: str) -> list:
    scenes = []
    for i in range(1, 6):
        m = re.search(rf"<Scene{i}>(.*?)</Scene{i}>", text, re.DOTALL)
        if m:
            scenes.append(m.group(1).strip())
    return scenes


def parse_single_prompt(text: str) -> str:
    m = re.search(r"<Prompt>(.*?)</Prompt>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def call_gpt55(headers: dict, messages: list, max_tokens: int = 200) -> tuple:
    """Call GPT-5.5 and return (response_text, latency_seconds)."""
    json_body = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 1.0,
    }
    start = time.perf_counter()
    resp = requests.post(PAPYRUS_ENDPOINT, headers=headers, json=json_body, verify=False)
    latency = time.perf_counter() - start

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        return "", latency

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text, latency


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    input_file = Path(__file__).parent / "test_query.txt"
    output_file = Path(__file__).parent / "gpt55_two_step_results.jsonl"

    # Load input
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {input_file}")

    # Auth
    print("Authenticating (browser will open)...")
    cred = InteractiveBrowserCredential(timeout=600)
    access_token = cred.get_token(VERIFY_SCOPE).token
    print("Token obtained successfully\n")

    headers = {
        "Authorization": "Bearer " + access_token,
        "Content-Type": "application/json",
        "papyrus-model-name": "gpt-5-chat-shortco-2025-08-07-Eval",
        "papyrus-quota-id": "PapyrusCustomer",
        "papyrus-timeout-ms": "120000",
    }

    step1_latencies = []
    step2_latencies = []
    step2_wall_latencies = []
    e2e_latencies = []
    results = []

    for idx, record in enumerate(records):
        lp_content = record.get("landing_page_content", "")
        url = record.get("url", "")
        print(f"\n{'='*60}")
        print(f"Record {idx+1}/{len(records)}: {url[:60]}...")
        print(f"{'='*60}")

        user_msg_step1 = (
            f"Generate 5 diverse scene concepts for the following landing page:\n\n"
            f"- URL: {url}\n"
            f"- Primary Content: {lp_content}"
        )

        # Step 1: Generate 5 scenes
        print("  Step 1: Generating 5 scene concepts...")
        messages_s1 = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP1_SCENES},
            {"role": "user", "content": user_msg_step1},
        ]
        text_s1, lat_s1 = call_gpt55(headers, messages_s1, max_tokens=200)
        step1_latencies.append(lat_s1)
        print(f"  Step 1 latency: {lat_s1:.3f}s")

        scenes = parse_scenes(text_s1)
        print(f"  Parsed {len(scenes)} scenes: {scenes}")

        # Step 2: Expand each scene (concurrent)
        def expand_scene(si_scene):
            si, scene = si_scene
            scene_content = (
                f"- URL: {url}\n"
                f"- Primary Content: {lp_content}\n\n"
                f"Expand this scene concept into a detailed prompt:\n"
                f"<Scene>{scene}</Scene>"
            )
            messages_s2 = [
                {"role": "system", "content": SYSTEM_PROMPT_STEP2_EXPAND},
                {"role": "user", "content": scene_content},
            ]
            text_s2, lat_s2 = call_gpt55(headers, messages_s2, max_tokens=150)
            prompt = parse_single_prompt(text_s2)
            return si, prompt, lat_s2

        step2_wall_start = time.perf_counter()
        scene_results = []
        with ThreadPoolExecutor(max_workers=len(scenes) if scenes else 1) as executor:
            futures = [executor.submit(expand_scene, (si, sc)) for si, sc in enumerate(scenes)]
            for f in as_completed(futures):
                scene_results.append(f.result())
        step2_wall_time = time.perf_counter() - step2_wall_start

        scene_results.sort(key=lambda x: x[0])
        prompts = [p for _, p, _ in scene_results]
        s2_lats = [lat for _, _, lat in scene_results]
        for si, prompt, lat_s2 in scene_results:
            step2_latencies.append(lat_s2)
            print(f"  Step 2 scene {si+1}: {lat_s2:.3f}s | {prompt[:60]}...")

        step2_wall_latencies.append(step2_wall_time)
        e2e = lat_s1 + step2_wall_time
        e2e_latencies.append(e2e)
        print(f"  Step 2 wall time: {step2_wall_time:.3f}s (concurrent)")
        print(f"  E2E latency: {e2e:.3f}s")

        results.append({
            "url": url,
            "scenes": scenes,
            "prompts": prompts,
            "step1_latency": lat_s1,
            "step2_individual_latencies": s2_lats,
            "step2_wall_time": step2_wall_time,
            "e2e_latency": e2e,
        })

    # Write results
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------------------
    # Latency summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"GPT-5.5 Two-Step Benchmark Summary (concurrent Step 2)")
    print(f"{'='*60}")
    print(f"Records:          {len(records)}")
    print(f"Total Step 2 calls: {len(step2_latencies)} (concurrent per record)")

    def print_stats(name, values):
        if not values:
            print(f"\n--- {name} --- NO DATA")
            return
        print(f"\n--- {name} ---")
        print(f"  Count:  {len(values)}")
        print(f"  Mean:   {statistics.mean(values):.3f}s")
        print(f"  Median: {statistics.median(values):.3f}s")
        print(f"  P95:    {percentile(values, 95):.3f}s")
        print(f"  P99:    {percentile(values, 99):.3f}s")
        print(f"  Min:    {min(values):.3f}s")
        print(f"  Max:    {max(values):.3f}s")

    print_stats("Step 1 latency (per record)", step1_latencies)
    print_stats("Step 2 individual call latency", step2_latencies)
    print_stats("Step 2 wall time (concurrent, per record)", step2_wall_latencies)
    print_stats("End-to-end latency per record (Step1 + Step2 wall)", e2e_latencies)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
