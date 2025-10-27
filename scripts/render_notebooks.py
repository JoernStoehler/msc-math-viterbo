"""
Render Jupytext notebooks to Markdown or single-file HTML.

Usage examples:

- Markdown (for GitHub viewing) into docs/notebooks:
    uv run python scripts/render_notebooks.py --to md --out docs/notebooks

- Single-file HTML into docs/notebooks/html:
    uv run python scripts/render_notebooks.py --to html --out docs/notebooks/html

Notes:
- Executes notebooks by default; use --no-exec to skip.
- Respects Jupytext percent-format notebooks under the input directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import jupytext
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat import NotebookNode


def _log(message: str) -> None:
    """Emit progress messages without violating lint rules."""
    sys.stderr.write(f"{message}\n")


def discover_notebooks(root: Path, pattern: str) -> list[Path]:
    """Return notebook files under ``root`` (recursively) matching ``pattern``."""
    pattern = pattern or "*.py"
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def load_notebook(path: Path) -> NotebookNode:
    """Load a Jupytext-managed notebook."""
    # jupytext auto-detects format from header (py:percent in this repo)
    return jupytext.read(path)


def execute_notebook(nb: NotebookNode, cwd: Path | None = None) -> NotebookNode:
    """Execute the notebook inside an ipykernel instance."""
    client = NotebookClient(
        nb, kernel_name="python3", resources={"metadata": {"path": str(cwd or ".")}}
    )
    client.execute()
    return nb


def export_markdown(nb: NotebookNode, out_dir: Path, stem: str, hide_input: bool = False) -> Path:
    """Render a notebook to Markdown with stable asset placement."""
    from nbconvert import MarkdownExporter
    from traitlets.config import Config

    c = Config()
    c.MarkdownExporter.exclude_input = hide_input
    # Ensure assets land under <stem>_files for stable relative links
    output_files_dir = f"{stem}_files"
    resources = {"output_files_dir": output_files_dir}
    body, resources = MarkdownExporter(config=c).from_notebook_node(nb, resources=resources)

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(body, encoding="utf-8")

    outputs = resources.get("outputs", {})
    if outputs:
        assets_dir = out_dir / output_files_dir
        assets_dir.mkdir(parents=True, exist_ok=True)
        for rel_name, data in outputs.items():
            # nbconvert may return just the filename; place under assets_dir
            asset_path = assets_dir / Path(rel_name).name
            mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
            with open(asset_path, mode) as f:
                f.write(data)
    return md_path


def export_html(nb: NotebookNode, out_dir: Path, stem: str, hide_input: bool = False) -> Path:
    """Render a notebook to a single self-contained HTML file."""
    from nbconvert import HTMLExporter
    from traitlets.config import Config

    c = Config()
    c.HTMLExporter.exclude_input = hide_input
    c.HTMLExporter.embed_images = True
    body, _ = HTMLExporter(config=c).from_notebook_node(nb)

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(body, encoding="utf-8")
    return html_path


def write_index(pages: Iterable[Path], out_index: Path, title: str) -> None:
    """Write a Markdown index that links to rendered notebook pages."""
    out_index.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "Generated from notebooks under `notebooks/`.", ""]
    for page in sorted(pages):
        # Make links relative to the index file location
        try:
            rel = page.relative_to(out_index.parent).as_posix()
        except ValueError:
            # If paths are on different drives or otherwise not relative, fallback to basename
            rel = page.name
        name = page.stem.replace("_", " ")
        lines.append(f"- [{name}]({rel})")
    out_index.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Render Jupytext notebooks")
    parser.add_argument(
        "--in", dest="inp", default="notebooks", help="Input directory with Jupytext notebooks"
    )
    parser.add_argument(
        "--pattern", default="*.py", help="Glob pattern under input dir (default: *.py)"
    )
    parser.add_argument("--to", choices=["md", "html"], default="md", help="Output format")
    parser.add_argument("--out", default="docs/notebooks", help="Output directory")
    parser.add_argument("--no-exec", action="store_true", help="Skip execution; convert as-is")
    parser.add_argument("--hide-input", action="store_true", help="Hide code cells in output")
    parser.add_argument("--index", default=None, help="Write an index.md alongside outputs (path)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")
    args = parser.parse_args()

    in_dir = Path(args.inp).resolve()
    out_dir = Path(args.out).resolve()

    # Optional: make sure CPU time cap is off for long runs
    if os.environ.get("VITERBO_CPU_LIMIT") == "":
        os.environ["VITERBO_CPU_LIMIT"] = "0"

    paths = discover_notebooks(in_dir, args.pattern)
    if not paths:
        _log(f"[render] No notebooks matched under {in_dir} with pattern '{args.pattern}'.")
        return 0

    rendered: list[Path] = []
    for path in paths:
        stem = path.stem
        _log(f"[render] Processing: {path}")
        try:
            nb = load_notebook(path)
            if not args.no_exec:
                execute_notebook(nb, cwd=path.parent)
            if args.to == "md":
                out_path = export_markdown(nb, out_dir, stem, hide_input=args.hide_input)
            else:
                out_path = export_html(nb, out_dir, stem, hide_input=args.hide_input)
            _log(f"[render] Wrote: {out_path}")
            rendered.append(out_path)
        except CellExecutionError as e:
            _log(f"[render] ERROR executing {path}: {e}")
            if args.fail_fast:
                return 2
        except Exception as e:  # noqa: BLE001
            _log(f"[render] ERROR processing {path}: {e}")
            if args.fail_fast:
                return 2

    if args.index:
        index_path = Path(args.index).resolve()
        title = "Rendered Notebooks (Executed)" + (
            " — HTML" if args.to == "html" else " — Markdown"
        )
        write_index(rendered, index_path, title=title)
        _log(f"[render] Index written: {index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
