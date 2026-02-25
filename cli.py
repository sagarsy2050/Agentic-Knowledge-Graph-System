import sys
import json
import argparse
from pathlib import Path

# Make sure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def cmd_research(args):
    from core.orchestrator import AgenticKGOrchestrator

    console.print(Panel.fit(f"Researching: [bold cyan]{args.topic}[/]"))

    def progress_cb(msg, pct):
        console.print(f"[dim][{pct:3d}%][/dim] {msg}")

    system = AgenticKGOrchestrator(on_progress=progress_cb)
    sources = [s.strip() for s in args.sources.split(",")]

    results = system.run_research_pipeline(
        topic=args.topic,
        max_papers=args.max_papers,
        sources=sources,
        local_dir=args.local_dir,
        year_from=args.year_from
    )

    stats = results.get("stats", {})
    table = Table(title="Pipeline Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    papers = results.get("papers", [])
    if papers:
        console.print(f"\n[bold]Papers Collected:[/bold]")
        for i, p in enumerate(papers[:10], 1):
            console.print(f"  {i}. {p.get('title', 'Unknown')[:70]} ({p.get('year', '')})")

    landscape = results.get("insights", {}).get("landscape", {})
    if landscape.get("overview"):
        console.print(Panel(landscape["overview"], title="Research Landscape"))

    gaps = results.get("insights", {}).get("research_gaps", [])
    if gaps:
        console.print("\n[bold]Research Gaps Found:[/bold]")
        for gap in gaps[:3]:
            console.print(f"  - {gap.get('description', '')}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\nResults saved to: {args.output}")


def cmd_ask(args):
    from core.orchestrator import AgenticKGOrchestrator
    system = AgenticKGOrchestrator()
    console.print(f"Question: [bold]{args.question}[/bold]\n")
    with console.status("Reasoning..."):
        result = system.ask(args.question)
    console.print(Panel(result["answer"], title="Answer", border_style="green"))
    if result.get("sources"):
        console.print(f"[dim]Used {len(result['sources'])} knowledge graph nodes[/dim]")


def cmd_hypotheses(args):
    from core.orchestrator import AgenticKGOrchestrator
    system = AgenticKGOrchestrator()
    console.print(f"Generating hypotheses for: [bold cyan]{args.topic}[/bold cyan]\n")
    with console.status("Thinking..."):
        hypotheses = system.generate_hypotheses(args.topic)
    for i, hyp in enumerate(hypotheses, 1):
        console.print(Panel(
            f"[bold]{hyp.get('hypothesis', '')}[/bold]\n\n"
            f"[green]Rationale:[/green] {hyp.get('rationale', '')}\n\n"
            f"[blue]Testing:[/blue] {hyp.get('testing_approach', '')}\n\n"
            f"[yellow]Impact:[/yellow] {hyp.get('impact', '')}",
            title=f"Hypothesis {i}",
            border_style="blue"
        ))


def cmd_review(args):
    from core.orchestrator import AgenticKGOrchestrator
    system = AgenticKGOrchestrator()
    console.print(f"Generating {args.style} literature review for: [bold]{args.topic}[/bold]\n")
    with console.status("Writing review... (may take a minute)"):
        review = system.write_literature_review(args.topic, style=args.style)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"# Literature Review: {args.topic}\n\n")
            f.write(review)
        console.print(f"Review saved to: {args.output}")
    else:
        console.print(Markdown(review))


def cmd_stats(args):
    from core.orchestrator import AgenticKGOrchestrator
    system = AgenticKGOrchestrator()
    stats = system.graph.get_stats()
    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Backend", stats.get("backend", "Unknown"))
    table.add_row("Total Nodes", str(stats.get("total_nodes", 0)))
    table.add_row("Total Edges", str(stats.get("total_edges", 0)))
    for ntype, count in stats.get("node_types", {}).items():
        table.add_row(f"  {ntype}", str(count))
    console.print(table)


def cmd_chat(args):
    from core.orchestrator import AgenticKGOrchestrator
    system = AgenticKGOrchestrator()
    console.print(Panel.fit("Research Assistant Chat\nType 'exit' to quit", border_style="blue"))
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            with console.status("Thinking..."):
                response = system.chat(user_input)
            console.print(f"\n[bold green]Assistant:[/bold green] {response}")
        except KeyboardInterrupt:
            break
    console.print("\nGoodbye!")


def cmd_setup(args):
    console.print(Panel.fit("System Setup & Check"))
    checks = []

    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        checks.append(("Ollama", "Running", f"Models: {', '.join(models[:3]) if models else 'None'}"))
    except:
        checks.append(("Ollama", "Not running", "Run: ollama serve"))

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
        driver.verify_connectivity()
        checks.append(("Neo4j", "Connected", "bolt://localhost:7687"))
        driver.close()
    except Exception as e:
        checks.append(("Neo4j", "Not connected", "Using in-memory fallback (OK)"))

    packages = ["langchain", "chromadb", "networkx", "streamlit", "httpx", "rich", "loguru"]
    for pkg in packages:
        try:
            __import__(pkg)
            checks.append((f"Python: {pkg}", "Installed", ""))
        except ImportError:
            checks.append((f"Python: {pkg}", "Missing", f"pip install {pkg}"))

    table = Table(title="System Check")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Notes")
    for comp, status, note in checks:
        icon = "OK" if "Not" not in status and "Missing" not in status else "!!"
        table.add_row(comp, f"{'[green]' if icon=='OK' else '[yellow]'}{status}", note)
    console.print(table)

    console.print("""
[bold]Quick Setup Commands:[/bold]

  1. Install Ollama:  https://ollama.ai
  2. Start Ollama:    ollama serve
  3. Pull a model:    ollama pull llama3.1:8b
  4. Install deps:    pip install -r requirements.txt
  5. Start Neo4j:     docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:latest
  6. Launch UI:       streamlit run ui/app.py
""")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Knowledge Graph System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py setup
  python cli.py research "transformer attention mechanisms" --max-papers 10
  python cli.py ask "What are the key challenges?"
  python cli.py hypotheses "protein folding"
  python cli.py review "CRISPR" --style survey -o review.md
  python cli.py chat
  python cli.py stats
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    r = subparsers.add_parser("research", help="Run research pipeline")
    r.add_argument("topic")
    r.add_argument("--max-papers", type=int, default=10, dest="max_papers")
    r.add_argument("--sources", default="arxiv")
    r.add_argument("--local-dir", default=None, dest="local_dir")
    r.add_argument("--year-from", type=int, default=None, dest="year_from")
    r.add_argument("-o", "--output", default=None)

    a = subparsers.add_parser("ask", help="Ask a question")
    a.add_argument("question")

    h = subparsers.add_parser("hypotheses", help="Generate hypotheses")
    h.add_argument("topic")

    rev = subparsers.add_parser("review", help="Generate literature review")
    rev.add_argument("topic")
    rev.add_argument("--style", default="academic",
                     choices=["academic", "survey", "executive", "technical"])
    rev.add_argument("-o", "--output", default=None)

    subparsers.add_parser("chat", help="Interactive chat")
    subparsers.add_parser("stats", help="Show graph stats")
    subparsers.add_parser("setup", help="Check system setup")

    args = parser.parse_args()

    commands = {
        "research": cmd_research,
        "ask": cmd_ask,
        "hypotheses": cmd_hypotheses,
        "review": cmd_review,
        "chat": cmd_chat,
        "stats": cmd_stats,
        "setup": cmd_setup,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
