"""
ingest.py
---------
CLI tool to index documents into the vector store.

Usage:
    python ingest.py                        # indexes data/docs/
    python ingest.py --docs path/to/docs/   # custom directory
    python ingest.py --reset                # wipe store first, then index
"""

import sys
import argparse
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.progress import track
from vectorstore import VectorStore, load_document
from retriever import HybridRetriever

console = Console()


def ingest(docs_dir: str = "data/docs", reset: bool = False):
    store = VectorStore()

    if reset:
        console.print("[yellow]Wiping existing vector store...[/yellow]")
        store.reset()

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        console.print(f"[red]Directory not found: {docs_dir}[/red]")
        sys.exit(1)

    supported = {".pdf", ".txt", ".md"}
    files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        console.print(f"[yellow]No supported files found in '{docs_dir}'[/yellow]")
        console.print("Supported formats: .pdf  .txt  .md")
        sys.exit(0)

    console.print(f"\nFound [bold]{len(files)}[/bold] file(s) to ingest\n")

    total_chunks = 0
    for file in track(files, description="Indexing..."):
        try:
            chunks = load_document(str(file))
            added = store.add_chunks(chunks)
            total_chunks += added
            console.print(f"  [green]✓[/green] {file.name} → {added} chunks")
        except Exception as e:
            console.print(f"  [red]✗[/red] {file.name} — {e}")

    console.print(f"\n[bold green]Done![/bold green] {total_chunks} chunks indexed.")
    console.print(f"Total chunks in store: [bold]{store.count()}[/bold]\n")

    # Pre-build BM25 index to verify it works
    retriever = HybridRetriever(vector_store=store)
    retriever.build_bm25_index()
    console.print("[dim]BM25 index built successfully.[/dim]")
    console.print("\nYou can now run: [bold]streamlit run app.py[/bold]")


def main():
    parser = argparse.ArgumentParser(description="Index documents into the RAG vector store")
    parser.add_argument("--docs", default="data/docs", help="Folder containing documents (default: data/docs)")
    parser.add_argument("--reset", action="store_true", help="Wipe the vector store before indexing")
    args = parser.parse_args()
    ingest(args.docs, reset=args.reset)


if __name__ == "__main__":
    main()
