#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark oriented-edge spectrum with pruning modes")
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-call wall-clock timeout in seconds (sets VITERBO_SOLVER_TIMEOUT)",
    )
    p.add_argument(
        "--polys",
        type=str,
        default="hypercube_4d_unit,pentagon_product_counterexample",
        help="Comma-separated names from tests.polytopes.STANDARD_POLYTOPES_BY_NAME",
    )
    p.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "table"],
        help="Output format",
    )
    p.add_argument(
        "--include-constants",
        action="store_true",
        help="Also run budgets with cF_constant=1/(2π) and 1/(4π)",
    )
    return p.parse_args()


def _load_poly(name: str):
    from tests.polytopes import STANDARD_POLYTOPES_BY_NAME

    if name not in STANDARD_POLYTOPES_BY_NAME:
        raise SystemExit(f"Unknown polytope name: {name}")
    poly = STANDARD_POLYTOPES_BY_NAME[name]
    return poly.vertices, poly.normals, poly.offsets


def _faces_for(vertices, normals, offsets):
    import torch

    from viterbo.math.capacity_ehz.stubs import _enumerate_two_faces, _vertex_facet_incidence

    tol = max(float(torch.finfo(vertices.dtype).eps) ** 0.5, 1e-9)
    incid = _vertex_facet_incidence(vertices, normals, offsets, tol)
    return _enumerate_two_faces(vertices, normals, incid, tol)


def _measure(name: str, include_constants: bool) -> dict[str, Any]:
    from viterbo.math.capacity_ehz.stubs import (
        compute_cF_constant_certified,
        oriented_edge_spectrum_4d,
    )
    from viterbo.runtime import TimeBudgetExceededError

    V, N, C = _load_poly(name)
    faces = _faces_for(V, N, C)

    rows: list[dict[str, Any]] = []

    # 1) Time certified C*(X) builder
    s = time.perf_counter()
    cstar = compute_cF_constant_certified(N, C, faces)
    b = time.perf_counter()
    rows.append(
        {
            "poly": name,
            "mode": "builder",
            "result": cstar,
            "time_ms": (b - s) * 1000.0,
        }
    )

    # Helper to time a run
    def run(label: str, **kwargs: Any) -> None:
        try:
            s = time.perf_counter()
            val = oriented_edge_spectrum_4d(V, N, C, **kwargs)
            b = time.perf_counter()
            rows.append(
                {
                    "poly": name,
                    "mode": label,
                    "result": float(val.item()),
                    "time_ms": (b - s) * 1000.0,
                }
            )
        except TimeBudgetExceededError:
            rows.append(
                {
                    "poly": name,
                    "mode": label,
                    "result": "timeout",
                    "time_ms": float("inf"),
                }
            )
        except Exception as e:  # pragma: no cover - diagnostic
            rows.append(
                {
                    "poly": name,
                    "mode": label,
                    "result": f"error: {type(e).__name__}",
                    "time_ms": None,
                }
            )

    # 2) Baseline (rotation only), certified
    run("base", use_cF_budgets=False, use_memo=None)
    # 3) Budgets with certified C*
    run("budgets", use_cF_budgets=True, use_memo=None, cF_constant=cstar)
    # 4) Memo only (heuristics)
    # 5) Budgets + memo (memo requires budgets)
    run("budgets+memo", use_cF_budgets=True, use_memo=True, cF_constant=cstar)
    if include_constants:
        run("budgets_halfpi", use_cF_budgets=True, use_memo=None, cF_constant=1.0 / (2.0 * math.pi))
        run(
            "budgets_quarter", use_cF_budgets=True, use_memo=None, cF_constant=1.0 / (4.0 * math.pi)
        )

    return {"rows": rows}


def _print_rows(rows: list[dict[str, Any]], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return
    if fmt == "csv":
        print("poly,mode,result,time_ms")
        for r in rows:
            print(f"{r['poly']},{r['mode']},{r['result']},{r['time_ms']}")
        return
    # Pretty table
    # Group by poly and print rows
    by_poly: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_poly.setdefault(r["poly"], []).append(r)
    for poly, items in by_poly.items():
        print(f"\n== {poly} ==")
        for r in items:
            print(f"- {r['mode']}: result={r['result']} time_ms={r['time_ms']}")


def main() -> None:
    args = _parse_args()
    # Ensure timeout is picked up by the decorated function at import time
    os.environ["VITERBO_SOLVER_TIMEOUT"] = str(args.timeout)
    # Delay imports until after env set
    polys = [s.strip() for s in args.polys.split(",") if s.strip()]
    all_rows: list[dict[str, Any]] = []
    for name in polys:
        result = _measure(name, include_constants=args.include_constants)
        all_rows.extend(result["rows"])
        sys.stdout.flush()
    _print_rows(all_rows, args.format)


if __name__ == "__main__":
    main()
