"""Emit summarized metadata for skills."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_frontmatter(content: str) -> dict[str, str]:
    """Return a dict of frontmatter key/value pairs."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        msg = "missing opening --- delimiter"
        raise ValueError(msg)

    fm_lines: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        fm_lines.append(line.rstrip())
    else:
        msg = "missing closing --- delimiter"
        raise ValueError(msg)

    data: dict[str, str] = {}
    current_key: str | None = None
    for raw in fm_lines:
        if not raw.strip():
            continue

        if raw.startswith(" ") and current_key:
            # Continuation of a previous value (YAML folded style).
            data[current_key] = f"{data[current_key]} {raw.strip()}"
            continue

        if ":" not in raw:
            msg = f"invalid frontmatter entry: {raw!r}"
            raise ValueError(msg)

        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        data[key] = value
        current_key = key

    return data


def load_skill_metadata(path: Path) -> dict[str, str]:
    """Read and parse the metadata for a single skill file."""
    content = path.read_text(encoding="utf-8")
    metadata = parse_frontmatter(content)
    if "name" not in metadata:
        metadata["name"] = path.stem
    metadata.setdefault("description", "")
    return metadata


def format_metadata(metadata: dict[str, str]) -> str:
    """Collapse metadata into a single summary line."""
    name = metadata.get("name", "unknown")
    description = metadata.get("description", "").strip()
    suffix_parts: list[str] = []
    for optional in ("last-updated", "license"):
        value = metadata.get(optional)
        if value:
            suffix_parts.append(f"{optional}: {value}")

    suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
    if description:
        return f"{name}: {description}{suffix}"
    return f"{name}{suffix}"


def iter_skill_files(skills_dir: Path) -> list[Path]:
    """Return all Markdown files directly under the skills directory."""
    return sorted(path for path in skills_dir.glob("*.md") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    """Run the CLI that validates and summarizes skill metadata."""
    parser = argparse.ArgumentParser(
        description="Summarize skills metadata without emitting full instructions.",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path("skills"),
        help="Directory containing flat skill markdown files (default: skills/).",
    )
    args = parser.parse_args(argv)

    skills_dir: Path = args.skills_dir
    if not skills_dir.exists():
        sys.stderr.write(f"no skills directory at {skills_dir}\n")
        return 1

    skill_files = iter_skill_files(skills_dir)
    if not skill_files:
        sys.stderr.write("no skill metadata found\n")
        return 0

    errors: list[tuple[Path, Exception]] = []
    summaries: list[str] = []
    for path in skill_files:
        try:
            metadata = load_skill_metadata(path)
        except Exception as exc:  # noqa: BLE001 - surface metadata issues
            errors.append((path, exc))
            continue
        summaries.append(format_metadata(metadata))

    for summary in summaries:
        sys.stdout.write(f"{summary}\n")

    for path, exc in errors:
        sys.stderr.write(f"[warn] {path}: {exc}\n")

    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
