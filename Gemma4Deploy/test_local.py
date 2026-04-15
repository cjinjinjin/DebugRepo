"""
Gemma4 DLIS 本地验证脚本 (不需要 GPU/vLLM)
Mock vLLM 引擎输出，验证 preprocess → build_step2_prompts → postprocess 流程
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dlis_inter import PreAndPostProcessor


def mock_step1_output():
    """模拟 vLLM Step 1 输出: 5 个 scene concepts"""
    return (
        "<Scene1>Close-up of hiking boots on rocky trail</Scene1>\n"
        "<Scene2>Woman smiling while adjusting backpack at mountain summit</Scene2>\n"
        "<Scene3>Camping gear arranged beside a forest stream</Scene3>\n"
        "<Scene4>Hiker reaching peak with panoramic valley view</Scene4>\n"
        "<Scene5>Golden hour light filtering through pine forest canopy</Scene5>"
    )


def mock_step2_outputs():
    """模拟 vLLM Step 2 输出: 5 个 expanded prompts"""
    return [
        "<Prompt>Rugged leather hiking boots planted firmly on wet granite, morning dew on laces, shallow depth of field with blurred alpine meadow background, natural lighting, editorial outdoor photography style</Prompt>",
        "<Prompt>Young woman in lightweight fleece smiling genuinely while adjusting her ultralight backpack straps at a misty mountain summit, candid moment, soft diffused daylight, documentary travel photography</Prompt>",
        "<Prompt>Neatly arranged camping essentials beside a clear forest stream, compact tent, titanium cookset, and trail map, overhead flat lay composition, warm afternoon light filtering through birch trees</Prompt>",
        "<Prompt>Solo hiker standing at granite peak edge overlooking vast green valley below, arms relaxed at sides, sense of accomplishment and freedom, golden hour backlighting, wide angle landscape perspective</Prompt>",
        "<Prompt>Warm golden sunlight streaming through tall pine canopy onto mossy forest floor, single hiking trail disappearing into soft focus distance, peaceful atmospheric mood, cinematic nature photography</Prompt>",
    ]


def test_full_pipeline():
    """测试完整流程: preprocess → build_step2_prompts → postprocess"""
    print("=" * 60)
    print("测试 1: 完整流程 (正常输入)")
    print("=" * 60)

    processor = PreAndPostProcessor()

    # --- preprocess ---
    input_data = {
        "landing_page_content": "Welcome to TrailMaster Outdoor Gear. Premium hiking boots, ultralight backpacks, and camping essentials for your next adventure.",
        "url": "https://trailmaster.example.com",
        "num_prompts": 5,
    }

    (step1_prompts, metadata) = processor.preprocess(input_data)

    assert isinstance(step1_prompts, list), f"step1_prompts should be list, got {type(step1_prompts)}"
    assert len(step1_prompts) == 1, f"step1_prompts should have 1 item, got {len(step1_prompts)}"
    assert "prompt" in step1_prompts[0], "step1_prompts[0] should have 'prompt' key"
    assert "<start_of_turn>" in step1_prompts[0]["prompt"], "prompt should use Gemma chat template"
    assert isinstance(metadata, list), f"metadata should be list, got {type(metadata)}"
    assert "user_message" in metadata[0], "metadata should contain user_message for Step 2"

    print(f"  [OK] preprocess: 返回 {len(step1_prompts)} 个 prompt, metadata 包含 user_message")
    print(f"  [OK] prompt 长度: {len(step1_prompts[0]['prompt'])} chars")
    print(f"  [OK] Gemma 模板格式: <start_of_turn>user...  <end_of_turn>  <start_of_turn>model")

    # --- build_step2_prompts ---
    step1_out = mock_step1_output()
    (step2_prompts, step2_meta) = processor.build_step2_prompts([step1_out], metadata)

    assert isinstance(step2_prompts, list), f"step2_prompts should be list, got {type(step2_prompts)}"
    assert len(step2_prompts) == 5, f"step2_prompts should have 5 items, got {len(step2_prompts)}"
    for i, p in enumerate(step2_prompts):
        assert "prompt" in p, f"step2_prompts[{i}] should have 'prompt' key"
        assert "<start_of_turn>" in p["prompt"], f"step2_prompts[{i}] should use Gemma chat template"

    assert "scenes" in step2_meta[0], "step2_meta should contain parsed scenes"
    assert len(step2_meta[0]["scenes"]) == 5, f"should have 5 scenes, got {len(step2_meta[0]['scenes'])}"

    print(f"  [OK] build_step2_prompts: 解析出 {len(step2_meta[0]['scenes'])} 个 scenes")
    print(f"  [OK] 生成 {len(step2_prompts)} 个 Step 2 prompts")
    for i, scene in enumerate(step2_meta[0]["scenes"]):
        print(f"       Scene {i+1}: {scene[:60]}...")

    # --- postprocess ---
    step2_outs = mock_step2_outputs()
    result = processor.postprocess(step2_outs, step2_meta)

    assert isinstance(result, dict), f"result should be dict, got {type(result)}"
    assert result["Status"] == "Success", f"Status should be Success, got {result['Status']}"
    assert len(result["generated_prompts"]) == 5, f"should have 5 prompts, got {len(result['generated_prompts'])}"
    assert result["format_compliant"] is True, f"format_compliant should be True, got {result['format_compliant']}"
    assert len(result["scenes"]) == 5, f"should have 5 scenes in result"

    print(f"  [OK] postprocess: Status={result['Status']}, format_compliant={result['format_compliant']}")
    print(f"  [OK] 生成 {len(result['generated_prompts'])} 个 prompts:")
    for i, p in enumerate(result["generated_prompts"]):
        print(f"       Prompt {i+1}: {p[:80]}...")

    print("\n  PASSED\n")
    return result


def test_truncated_step1():
    """测试 Step 1 输出被截断 (stop at </Scene5>)"""
    print("=" * 60)
    print("测试 2: Step 1 输出截断 (缺少 </Scene5> 闭合)")
    print("=" * 60)

    processor = PreAndPostProcessor()

    input_data = {
        "landing_page_content": "Test page content",
        "num_prompts": 5,
    }
    (_, metadata) = processor.preprocess(input_data)

    # 模拟 stop string 截断: 没有 </Scene5>
    truncated = (
        "<Scene1>Product close-up shot</Scene1>\n"
        "<Scene2>Lifestyle usage scene</Scene2>\n"
        "<Scene3>Environmental setting</Scene3>\n"
        "<Scene4>Benefit outcome shot</Scene4>\n"
        "<Scene5>Atmospheric mood composition"  # 没有闭合标签
    )

    (step2_prompts, step2_meta) = processor.build_step2_prompts([truncated], metadata)
    assert len(step2_meta[0]["scenes"]) == 5, f"should recover 5 scenes, got {len(step2_meta[0]['scenes'])}"
    print(f"  [OK] 截断恢复: 解析出 {len(step2_meta[0]['scenes'])} 个 scenes")
    print(f"       Scene 5: {step2_meta[0]['scenes'][4]}")
    print("\n  PASSED\n")


def test_partial_step2():
    """测试 Step 2 部分输出为空"""
    print("=" * 60)
    print("测试 3: Step 2 部分输出为空/格式异常")
    print("=" * 60)

    processor = PreAndPostProcessor()

    input_data = {"landing_page_content": "Test", "num_prompts": 5}
    (_, metadata) = processor.preprocess(input_data)
    (_, step2_meta) = processor.build_step2_prompts([mock_step1_output()], metadata)

    # 模拟: 第 3 个输出没有 <Prompt> 标签
    step2_outs = [
        "<Prompt>Good prompt one</Prompt>",
        "<Prompt>Good prompt two</Prompt>",
        "This output has no prompt tags at all",  # 无标签, 会 fallback 到 text.strip()
        "<Prompt>Good prompt four</Prompt>",
        "<Prompt>Good prompt five</Prompt>",
    ]

    result = processor.postprocess(step2_outs, step2_meta)
    assert result["Status"] == "Success", "should still be Success with 5 non-empty results"
    assert len(result["generated_prompts"]) == 5
    # 第 3 个应该 fallback 到 strip 后的原文
    assert "no prompt tags" in result["generated_prompts"][2]
    print(f"  [OK] 无标签 fallback: Prompt 3 = '{result['generated_prompts'][2][:50]}...'")
    print(f"  [OK] Status={result['Status']}, 共 {len(result['generated_prompts'])} prompts")
    print("\n  PASSED\n")


def test_string_input():
    """测试 preprocess 接受 JSON string 输入"""
    print("=" * 60)
    print("测试 4: preprocess 接受 JSON string 输入")
    print("=" * 60)

    processor = PreAndPostProcessor()
    input_str = json.dumps({
        "landing_page_content": "Test LP content as string input",
        "url": "https://example.com",
    })

    (step1_prompts, metadata) = processor.preprocess(input_str)
    assert len(step1_prompts) == 1
    assert "prompt" in step1_prompts[0]
    print(f"  [OK] JSON string 输入正常解析")
    print("\n  PASSED\n")


def test_lp_truncation():
    """测试 LP content 超长截断"""
    print("=" * 60)
    print("测试 5: LP content 超长截断")
    print("=" * 60)

    processor = PreAndPostProcessor()
    long_content = "A" * 10000
    input_data = {
        "landing_page_content": long_content,
        "max_lp_chars": 100,
    }

    (step1_prompts, metadata) = processor.preprocess(input_data)
    prompt_text = step1_prompts[0]["prompt"]
    # 应该包含截断标记
    assert "... [truncated]" in prompt_text, "长内容应被截断"
    # 不应该包含完整的 10000 个 A
    assert "A" * 10000 not in prompt_text
    print(f"  [OK] 内容截断至 100 chars + '... [truncated]'")
    print("\n  PASSED\n")


def test_thinking_mode_output():
    """测试 Gemma4 thinking 模式输出的处理"""
    print("=" * 60)
    print("测试 6: Thinking 模式输出过滤")
    print("=" * 60)

    processor = PreAndPostProcessor()

    input_data = {"landing_page_content": "Test", "num_prompts": 5}
    (_, metadata) = processor.preprocess(input_data)

    # 模拟 Step 1 输出带 thinking 前缀
    step1_with_thinking = (
        "thought\n"
        "<Scene1>Close-up of hiking boots on rocky trail</Scene1>\n"
        "<Scene2>Woman smiling while adjusting backpack</Scene2>\n"
        "<Scene3>Camping gear beside a forest stream</Scene3>\n"
        "<Scene4>Hiker reaching peak with valley view</Scene4>\n"
        "<Scene5>Golden hour light through pine canopy</Scene5>"
    )

    (step2_prompts, step2_meta) = processor.build_step2_prompts([step1_with_thinking], metadata)
    assert len(step2_meta[0]["scenes"]) == 5, f"should parse 5 scenes, got {len(step2_meta[0]['scenes'])}"
    print(f"  [OK] Step 1 thinking 前缀正确过滤, 解析出 {len(step2_meta[0]['scenes'])} 个 scenes")

    # 模拟 Step 2 输出带 thinking 前缀
    step2_with_thinking = [
        "thought\n<Prompt>Good prompt one without thinking noise</Prompt>",
        "<think>Let me think about this scene...</think>\n<Prompt>Good prompt two after think block</Prompt>",
        "thought\n- analysis line 1\n- analysis line 2\n<Prompt>Good prompt three</Prompt>",
        "<Prompt>Good prompt four no thinking</Prompt>",
        "thought\n<Prompt>Good prompt five</Prompt>",
    ]

    result = processor.postprocess(step2_with_thinking, step2_meta)
    assert result["Status"] == "Success"
    for i, p in enumerate(result["generated_prompts"]):
        assert "thought" not in p.lower(), f"Prompt {i+1} should not contain 'thought': {p[:50]}"
        assert "<think>" not in p, f"Prompt {i+1} should not contain '<think>'"
    print(f"  [OK] Step 2 thinking 内容正确过滤")
    print(f"  [OK] 5 个 prompts 均不含 thinking 内容")
    print("\n  PASSED\n")


if __name__ == "__main__":
    print("\nGemma4 DLIS 本地验证\n")

    test_full_pipeline()
    test_truncated_step1()
    test_partial_step2()
    test_string_input()
    test_lp_truncation()
    test_thinking_mode_output()

    print("=" * 60)
    print("全部测试通过!")
    print("=" * 60)
