#!/usr/bin/env python3
"""Fail CI if any policy waivers are expired or malformed.

Waivers are tracked in `waivers.toml` at the repo root. Each entry must include:

- id (string)
- summary (string)
- owner (string)
- scope (string)  # e.g., file or module path
- created (YYYY-MM-DD)
- expires (YYYY-MM-DD)
- justification (string)
- removal_plan (string)

CI policy: any waiver with an `expires` date earlier than today fails the check.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    print("Python 3.11+ required for tomllib.", file=sys.stderr)
    sys.exit(2)


def _parse_date(value: str) -> _dt.date:
    return _dt.date.fromisoformat(value)


def main() -> int:
    waivers_file = Path("waivers.toml")
    if not waivers_file.exists():
        # No waivers → nothing to check.
        return 0

    data = tomllib.loads(waivers_file.read_text(encoding="utf-8"))
    waivers = data.get("waiver", [])
    if not isinstance(waivers, list):
        print("waivers.toml: `[ [waiver] ... ]` array expected", file=sys.stderr)
        return 2

    today = _dt.date.today()
    failures: list[str] = []

    required = {
        "id",
        "summary",
        "owner",
        "scope",
        "created",
        "expires",
        "justification",
        "removal_plan",
    }

    for idx, waiver in enumerate(waivers):
        if not isinstance(waiver, dict):
            failures.append(f"waivers[{idx}] must be a table")
            continue
        missing = required.difference(waiver.keys())
        if missing:
            failures.append(f"waiver '{waiver.get('id', f'#${idx}')}' missing fields: {sorted(missing)}")
            continue
        try:
            expires = _parse_date(str(waiver["expires"]))
            created = _parse_date(str(waiver["created"]))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"waiver '{waiver.get('id')}': invalid date format ({exc})")
            continue
        if created > expires:
            failures.append(f"waiver '{waiver['id']}': created after expires")
        if expires < today:
            failures.append(f"waiver '{waiver['id']}' expired on {expires.isoformat()}")

    if failures:
        print("Policy waiver check failed:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

