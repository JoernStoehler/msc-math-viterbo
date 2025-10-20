"""
Fetch papers by arXiv ID or DOI, store text under docs/papers/, and emit an index snippet.

Usage examples:
  uv run python scripts/fetch_paper.py --arxiv 1712.03494
  uv run python scripts/fetch_paper.py --arxiv 2008.10111 --status useful
  uv run python scripts/fetch_paper.py --doi 10.1090/S0894-0347-00-00341-3 --status background

Notes:
- Uses only Python stdlib + `pdftotext` CLI.
- For DOI, queries OpenAlex for metadata and OA links (no API key needed).
- Writes `paper.md` with a small YAML header and outputs an index block to stdout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import shutil
import subprocess
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DOCS_DIR = Path("docs/papers")


def _slugify(text: str, maxlen: int = 60) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:maxlen].strip("-") or "paper"


def _authors_short(authors: list[str]) -> str:
    if not authors:
        return "anon"
    last = [a.split()[-1] for a in authors]
    if len(last) == 1:
        return last[0]
    if len(last) == 2:
        return f"{last[0]}-{last[1]}"
    return f"{last[0]}-etal"


def _pdftotext(pdf_path: Path) -> str:
    """Run pdftotext on a local PDF and return UTF-8 text or exit on failure."""
    try:
        out = subprocess.check_output(
            ["pdftotext", "-layout", "-nopgbrk", str(pdf_path), "-"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"pdftotext failed: {exc}")
    return out.decode("utf-8", errors="replace")


def _write_paper_md(outdir: Path, title: str, source: str, text: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    md = f"---\nsource: {source}\nfetched: {_dt.datetime.utcnow().date()}\n---\n# {title}\n\n{text}"
    path = outdir / "paper.md"
    path.write_text(md, encoding="utf-8")
    return path


def fetch_arxiv(arxiv_id: str, status: str | None = None) -> None:
    """Fetch an arXiv PDF, convert to Markdown, write to docs, and log index."""
    # Metadata
    api = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    req = urllib.request.Request(api, headers={"User-Agent": "fetch-paper/1.0"})
    with urllib.request.urlopen(req) as r:
        xml = r.read().decode("utf-8", errors="replace")
    title_match = re.search(r"<title>(.*?)</title>", xml, re.DOTALL)
    title = title_match.group(1).split("\n", 1)[0].strip() if title_match else f"arXiv:{arxiv_id}"
    # Extract authors (basic)
    authors = re.findall(r"<name>(.*?)</name>", xml)
    # Year
    pub_match = re.search(r"<published>(\d{4})-", xml)
    year = pub_match.group(1) if pub_match else _dt.datetime.utcnow().strftime("%Y")

    # PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "fetch-paper/1.0"})
        with urllib.request.urlopen(req) as r:
            tmp.write(r.read())
        pdf_path = Path(tmp.name)

    try:
        text = _pdftotext(pdf_path)
    finally:
        try:
            pdf_path.unlink()
        except FileNotFoundError:
            pass

    slug = f"{year}-{_authors_short(authors)}-{_slugify(title)}"
    outdir = DOCS_DIR / slug
    path = _write_paper_md(outdir, title, f"arXiv:{arxiv_id}", text)

    idx_block = textwrap.dedent(
        f"""
        - {", ".join(authors) if authors else "Unknown"} ({year}) — “{title}” — arXiv:{arxiv_id}
          - Local: `{outdir.as_posix()}/paper.md`
          - Status: {status or "useful"} — add as appropriate.
          - Takeaways:
            - …
        """
    ).strip()

    logging.info("Saved: %s", path)
    logging.info("\nIndex snippet:\n%s", idx_block)


def _crossref_meta(doi: str) -> tuple[str, str, list[str]]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-paper/1.0"})
    with urllib.request.urlopen(req) as r:
        msg = json.load(r).get("message", {})
    title = (msg.get("title") or [doi])[0]
    year = ""
    issued = msg.get("issued", {}).get("'date-parts", []) or msg.get("issued", {}).get(
        "date-parts", []
    )
    if issued and issued[0]:
        year = str(issued[0][0])
    authors: list[str] = []
    for a in msg.get("author", []) or []:
        name = " ".join([a.get("given", ""), a.get("family", "")]).strip() or a.get("name", "")
        if name:
            authors.append(name)
    return title, (year or _dt.datetime.utcnow().strftime("%Y")), authors


def fetch_doi(doi: str, status: str | None = None) -> None:
    """Fetch by DOI using OpenAlex metadata and OA links; log index snippet."""
    # Prefer OpenAlex for OA location and metadata
    data = None
    try:
        oa_url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
        req = urllib.request.Request(oa_url, headers={"User-Agent": "fetch-paper/1.0"})
        with urllib.request.urlopen(req) as r:
            data = json.load(r)
    except (urllib.error.URLError, json.JSONDecodeError):
        data = None

    title: str | None = None
    year: str | None = None
    authors: list[str] = []
    pdf_url: str | None = None

    if data:
        title = data.get("title")
        year = str(data.get("publication_year") or _dt.datetime.utcnow().year)
        authors = [a.get("author", {}).get("display_name", "") for a in data.get("authorships", [])]
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url:
            for loc in data.get("locations", []) or []:
                if loc.get("is_oa") and (loc.get("url_for_pdf") or loc.get("url")):
                    pdf_url = loc.get("url_for_pdf") or loc.get("url")
                    break

    if not title:
        try:
            title, year, authors = _crossref_meta(doi)
        except (urllib.error.URLError, json.JSONDecodeError):
            title = doi
            year = _dt.datetime.utcnow().strftime("%Y")
            authors = []

    # If no OA PDF, print an index snippet and exit gracefully
    if not pdf_url:
        slug = f"{year}-{_authors_short(authors)}-{_slugify(title)}"
        outdir = DOCS_DIR / slug
        idx_block = textwrap.dedent(
            f"""
            - {", ".join(authors) if authors else "Unknown"} ({year}) — “{title}” — doi:{doi}
              - Local: none (no OA link found via OpenAlex)
              - Status: {status or "background"} — add as appropriate.
              - Takeaways:
                - …
            """
        ).strip()
        logging.info("No OA PDF found; not saved.")
        logging.info("\nIndex snippet:\n%s", idx_block)
        return

    # Fetch and convert
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "fetch-paper/1.0"})
        with urllib.request.urlopen(req) as r:
            tmp.write(r.read())
        pdf_path = Path(tmp.name)
    try:
        text = _pdftotext(pdf_path)
    finally:
        try:
            pdf_path.unlink()
        except FileNotFoundError:
            pass

    slug = f"{year}-{_authors_short(authors)}-{_slugify(title)}"
    outdir = DOCS_DIR / slug
    path = _write_paper_md(outdir, title or doi, f"doi:{doi}", text)

    idx_block = textwrap.dedent(
        f"""
        - {", ".join(authors) if authors else "Unknown"} ({year}) — “{title or doi}” — doi:{doi}
          - Local: `{outdir.as_posix()}/paper.md`
          - Status: {status or "background"} — add as appropriate.
          - Takeaways:
            - …
        """
    ).strip()

    logging.info("Saved: %s", path)
    logging.info("\nIndex snippet:\n%s", idx_block)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for fetching papers by arXiv or DOI."""
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--arxiv", help="arXiv identifier, e.g., 1712.03494 or 2008.10111v2")
    g.add_argument("--doi", help="DOI, e.g., 10.1007/s00039-019-00486-4")
    parser.add_argument("--status", help="Initial status for index entry", default=None)
    args = parser.parse_args(argv)

    if shutil.which("pdftotext") is None:
        raise SystemExit("pdftotext not found; please install poppler-utils")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.arxiv:
        arxiv_id = args.arxiv.strip()
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("https://arxiv.org/abs/", "")
        fetch_arxiv(arxiv_id, status=args.status)
    elif args.doi:
        fetch_doi(args.doi.strip(), status=args.status)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
