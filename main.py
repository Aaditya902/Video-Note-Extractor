"""
main.py — CLI entry point for Video Note Extractor.

Usage:
    # From a YouTube URL
    python main.py --url "https://www.youtube.com/watch?v=..."

    # From a local video file (.mp4, .mkv, .mov, .avi, .webm, .m4v, .flv)
    python main.py --video lecture.mp4
    python main.py --video /path/to/meeting_recording.mkv

    # From a transcript file (.txt / .srt / .vtt)
    python main.py --file lecture.srt

    # Custom output directory
    python main.py --video meeting.mp4 --output ./notes

    # Whisper model override (larger = more accurate, slower)
    python main.py --video lecture.mp4 --whisper-model small
"""

import argparse
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

load_dotenv()
console = Console()


def check_env() -> None:
    """Fail fast if API key is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[bold red]✗ ANTHROPIC_API_KEY not set.[/bold red]")
        console.print("  Create a .env file from .env.example and add your key.")
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    check_env()

    # ── 1. Ingestion ──────────────────────────────────────────────────────────
    segments = []
    video_title = ""

    if args.url:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task("Downloading audio from YouTube...", total=None)
            from ingestion.youtube import download_audio
            audio_path, video_title = download_audio(args.url, output_dir=args.output)
            p.update(task, description=f"[green]✓ Downloaded:[/green] {video_title}")

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task(f"Transcribing with Whisper ({args.whisper_model})...", total=None)
            from transcription.whisper_engine import transcribe
            segments = transcribe(audio_path, model_size=args.whisper_model)
            p.update(task, description=f"[green]✓ Transcribed:[/green] {len(segments)} segments")

    elif args.video:
        from ingestion.local_video import extract_audio, _format_duration, SUPPORTED_EXTENSIONS

        # ── Probe & extract ───────────────────────────────────────────────────
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task(f"Probing video file: {Path(args.video).name}", total=None)
            audio_path, meta = extract_audio(args.video, output_dir=args.output)
            video_title = meta.title
            p.update(
                task,
                description=(
                    f"[green]✓ Audio extracted:[/green] {meta.title} "
                    f"[dim]({_format_duration(meta.duration_sec)} · "
                    f"{meta.video_codec}/{meta.audio_codec})[/dim]"
                ),
            )

        # ── Transcribe ────────────────────────────────────────────────────────
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task(
                f"Transcribing with Whisper ({args.whisper_model})...", total=None
            )
            from transcription.whisper_engine import transcribe
            segments = transcribe(audio_path, model_size=args.whisper_model)
            p.update(task, description=f"[green]✓ Transcribed:[/green] {len(segments)} segments")

    elif args.file:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task(f"Loading transcript: {args.file}", total=None)
            from ingestion.file_loader import load_file
            segments = load_file(args.file)
            video_title = Path(args.file).stem
            p.update(task, description=f"[green]✓ Loaded:[/green] {len(segments)} segments")

    else:
        console.print("[red]Provide one of: --url, --video, or --file[/red]")
        sys.exit(1)

    if not segments:
        console.print("[red]✗ No transcript segments found. Cannot continue.[/red]")
        sys.exit(1)

    # ── 2. Chunking ───────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Chunking transcript...", total=None)
        from processing.chunker import chunk_segments
        chunks = chunk_segments(segments, window_words=250, overlap_words=50)
        p.update(task, description=f"[green]✓ Chunked:[/green] {len(chunks)} chunks")

    # ── 3. Embedding + Vector Store ───────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Embedding chunks (sentence-transformers)...", total=None)
        from processing.vector_store import VectorStore
        store = VectorStore()
        store.add_chunks(chunks)
        p.update(task, description=f"[green]✓ Indexed:[/green] {store.count()} chunks in vector store")

    # ── 4. LLM Extraction ────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Extracting notes with Claude...", total=None)
        from llm.claude_extractor import extract_notes
        result = extract_notes(store, video_title=video_title)
        p.update(task, description="[green]✓ Notes extracted[/green]")

    # ── 5. Output ─────────────────────────────────────────────────────────────
    from output.formatter import save_markdown, save_json
    md_path = save_markdown(result, output_dir=args.output)
    json_path = save_json(result, output_dir=args.output)

    # ── Display results ───────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold white]{result.title}[/bold white]\n\n[dim]{result.summary}[/dim]",
        title="[bold green]◈ Extraction Complete[/bold green]",
        border_style="green",
    ))

    # Key Concepts
    console.print(f"\n[bold]Key Concepts:[/bold] {' · '.join(result.key_concepts)}\n")

    # Notes table
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim", width=8)
    table.add_column("Heading", style="bold", width=25)
    table.add_column("Content")

    for note in result.notes:
        table.add_row(
            note.timestamp or "—",
            note.heading,
            note.content,
        )
    console.print(table)

    # Action items
    console.print("[bold]Action Items:[/bold]")
    for item in result.action_items:
        console.print(f"  [cyan]▸[/cyan] {item}")

    # File paths
    console.print(f"\n[dim]Saved:[/dim]")
    console.print(f"  [green]→[/green] {md_path}")
    console.print(f"  [green]→[/green] {json_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video Note Extractor — Convert videos into structured notes using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url",   metavar="URL",  help="YouTube video URL")
    source.add_argument("--video", metavar="PATH", help="Local video file (.mp4, .mkv, .mov, .avi, .webm, .m4v, .flv)")
    source.add_argument("--file",  metavar="PATH", help="Transcript file (.txt, .srt, .vtt)")

    parser.add_argument(
        "--output", metavar="DIR", default="data",
        help="Output directory for notes (default: ./data)"
    )
    parser.add_argument(
        "--whisper-model",
        metavar="SIZE",
        default=os.getenv("WHISPER_MODEL", "base"),
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()