#!/usr/bin/env python3
"""
Creative Ads Image Generation Agent

Usage:
    python main.py "https://example.com/roofing-services"
    python main.py "https://example.com/tax-software" --eval --output results.json
    python main.py "https://example.com" --llm-backend azure

Environment variables:
    OPENAI_API_KEY      Your OpenAI or Azure OpenAI API key
    OPENAI_API_BASE     API base URL (default: https://api.openai.com/v1)
    LLM_MODEL           Model name (default: gpt-4o)
    EMBED_MODEL         Embedding model (default: text-embedding-3-small)
"""
import argparse
import os
import sys

# Ensure CreativeAdsAgent/ is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from config import Config
from orchestrator import PipelineOrchestrator
from utils.llm_client import set_config
from utils.stream_printer import print_banner, print_final_results


def main():
    parser = argparse.ArgumentParser(
        description="Creative Ads Image Generation Agent — URL → 5 refined image prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="Landing page URL to process")
    parser.add_argument(
        "--eval", action="store_true",
        help="Also generate 12 verification questions per refined prompt",
    )
    parser.add_argument(
        "--output", default="output.json",
        help="Path to save full pipeline state as JSON (default: output.json)",
    )
    parser.add_argument(
        "--llm-backend", choices=["openai", "azure"], default="openai",
        dest="llm_backend",
    )
    args = parser.parse_args()

    config = Config.from_env(llm_backend=args.llm_backend)

    if not config.llm_api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "       Set it in your shell or in a .env file."
        )
        sys.exit(1)

    set_config(config)
    print_banner()

    orchestrator = PipelineOrchestrator(config)
    state = orchestrator.run(url=args.url, eval_mode=args.eval)

    print_final_results(state)
    state.save(args.output)

    if state.errors:
        print(f"\n⚠️  Pipeline completed with {len(state.errors)} error(s). Check output JSON for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
