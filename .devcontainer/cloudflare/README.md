Cloudflare Workers — VibeKanban Hygiene and UI Polish

Overview
- Two Workers run in front of VibeKanban:
  - API sanitizer: `wrangler-sanitizer.toml` routes `/api/*` and escapes only intraword underscores to prevent snake_case → italics, while preserving star-based italics.
  - Font/CSS injector: `wrangler.toml` routes `/*` and injects Inter (or other) for readability. No functional changes to Markdown.

Key Decisions (2025-10-18)
- Keep `*` → italics; mitigate only `_` → italics via pre-render sanitization at the API edge.
- Escape only intraword underscores (regex-wise: a_b becomes a\_b when both neighbors are word characters).
- Preserve underscores inside:
  - Inline code: `` `...` ``
  - Fenced code blocks: ```...``` or ~~~...~~~
  - Markdown link destinations: `[label](dest_here)`
  - Angle autolinks: `<https://...>`
  - Bare URLs: `http(s)://…`, `ftp://…`, `mailto:`
- Field allowlist sanitized: `title`, `description`, `comment`, `body`, `text`.
- HTML remains sanitized/stripped by VK; no changes at the edge.
- No request/response body logging at the edge.
- We are not forking or modifying upstream VK — this is overlay only.

Rationale
- Underscores in identifiers (snake_case, x_i, ENV_VARS) frequently collide with Markdown italics. Disabling only intraword underscores at ingress preserves readability while keeping `*` italics and headings/bullets.
- Preserving link destinations and URLs avoids broken links from escaping.
- Separation of concerns: API sanitizer on `/api/*` is content-only; font injector on `/*` adjusts UI without interfering with API.

Deploy
- Prerequisites (in devcontainer once):
  - `wrangler login` (interactive) — persisted via container volume per docs.
- One-shot deploy (recommended):
  - Inside container: `just cf`
  - From host: `bash .devcontainer/bin/admin cf`
- Individual deploys (if needed):
  - API sanitizer: `wrangler -c .devcontainer/cloudflare/wrangler-sanitizer.toml deploy`
  - Font/CSS: `wrangler -c .devcontainer/cloudflare/wrangler.toml deploy`

Auto-deploy option
- Set `CF_AUTO_DEPLOY=1` in the host environment to have `admin start` deploy both Workers automatically after services start. Default is off to avoid surprising deploys.

Justfile helpers (inside container)
- `just cf` — deploy both Workers (sanitizer + font)
- `just cf-deploy-sanitizer` — deploy API sanitizer
- `just cf-deploy-font` — deploy font injector
- `just cf-tail` — tail both Workers
- `just cf-tail-sanitizer` — tail sanitizer logs
- `just cf-tail-font` — tail font logs

Admin helpers (host)
- `bash .devcontainer/bin/admin cf` — deploy both Workers
- `bash .devcontainer/bin/admin cf-tail` — tail both Workers
- Also available: `cf-deploy-sanitizer`, `cf-deploy-font`, `cf-tail-sanitizer`, `cf-tail-font`

Troubleshooting
- Sanitizer not effective:
  - Confirm the API route matches your domain in `wrangler-sanitizer.toml`.
  - Use `cf-tail-sanitizer` to ensure deploy is active; re-auth with `wrangler login` if needed.
  - Post a minimal VK update: "alpha_beta and `gamma_delta`" — expect alpha_beta literal, gamma_delta unchanged.
- Links broken or escaped:
  - The sanitizer preserves URLs and link destinations by design. If a repro exists, capture the exact text snippet.
- Font not applying:
  - Confirm `wrangler.toml` route and that HTMLRewriter injects CSS on the base document.

Update Strategy (when VK changes)
- If VK adds new Markdown-bearing fields, add them to the allowlist in `worker-vk-sanitizer.js`.
- Keep the intraword underscore rule — safe and minimally invasive.
- If Markdown rules in VK change (e.g., intrinsic intraword underscore disabled upstream), the sanitizer remains harmless.

Change Log
- 2025-10-18: Initial deployment with intraword-underscore sanitizer and Inter font injector.
