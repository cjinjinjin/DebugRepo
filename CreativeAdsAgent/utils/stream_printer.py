try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    _rich = True
except ImportError:
    _rich = False

_console = None
if _rich:
    _console = Console()


def _print(msg: str):
    if _rich:
        _console.print(msg)
    else:
        print(msg)


def print_banner():
    if _rich:
        _console.print(Panel.fit(
            "[bold cyan]Creative Ads Image Generation Agent[/bold cyan]\n"
            "[dim]LP Crawling → RAG → LPUnderstanding → Creator → Refiner[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 60)
        print("  Creative Ads Image Generation Agent")
        print("  LP Crawling → RAG → LPUnderstanding → Creator → Refiner")
        print("=" * 60)


def print_step(title: str, content=None):
    sep = "─" * 60
    if _rich:
        _console.print(f"\n[bold green]▶ {title}[/bold green]")
        _console.rule(style="green")
    else:
        print(f"\n{'=' * 60}")
        print(f"▶  {title}")
        print(sep)
    if content is not None:
        if isinstance(content, list):
            for i, item in enumerate(content, 1):
                _print(f"  [{i}] {str(item)[:200]}")
        elif hasattr(content, "__dict__"):
            for k, v in content.__dict__.items():
                if k != "raw_llm_output" and v:
                    _print(f"  {k}: {str(v)[:200]}")
        elif content:
            _print(f"  {str(content)[:400]}")


def print_final_results(state):
    if _rich:
        _console.print("\n")
        _console.print(Panel.fit("[bold yellow]✅ Pipeline Complete[/bold yellow]", border_style="yellow"))
    else:
        print("\n" + "=" * 60)
        print("✅  Pipeline Complete")
        print("=" * 60)

    print_step("Final Refined Prompts")
    for i, p in enumerate(state.refined_prompts, 1):
        _print(f"\n  [Prompt {i}]\n  {p}\n")

    if state.errors:
        _print(f"\n⚠️  Errors encountered: {len(state.errors)}")
        for e in state.errors:
            _print(f"  - {e}")
