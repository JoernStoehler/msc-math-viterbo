# Gap Matrix — 2025-10-21

Minimal audit of documentation, tests, and navigation claims against the current
`src/` implementation surface.

## Docs ↔ Code

| Doc page | Doc claim | Reality in `src/` | Action |
| --- | --- | --- | --- |
| `docs/math/similarity.md` | Presents similarity metrics module without front-of-page status | `src/viterbo/math/similarity.py` only exposes stubs that raise `NotImplementedError` | Add STATUS banner pointing to milestone plan; hide nav entry until implementation lands |
| `docs/math/polar.md` | Presents polar/Mahler helpers without status warning | `src/viterbo/math/polar.py` is stub-only (`NotImplementedError`) | Add STATUS banner pointing to milestone plan; hide nav entry until implementation lands |

## Tests ↔ Code

| Test | Gap | Action |
| --- | --- | --- |
| `tests/math/test_similarity.py` | No coverage documenting stubbed similarity helpers | Track planned behaviour via `xfail` against current `NotImplementedError` until implementation | 
| `tests/math/test_polar.py` | Same for polar helpers | Track via `xfail` expecting `NotImplementedError` |

## MkDocs Navigation

| Nav entry | Gap | Action |
| --- | --- | --- |
| `Math API -> similarity`, `Math API -> polar` | Visible in nav despite modules being stub-only | Remove from nav for now; link remains reachable directly from gap matrix |

