// Cloudflare Worker: VibeKanban outbound text sanitizer (optional)
// Acts as a reverse proxy in front of VK, intercepting JSON bodies
// and escaping underscores in fields that are rendered as Markdown.

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Only sanitize JSON on POST/PUT/PATCH to VK API paths; proxy others.
    const method = request.method.toUpperCase();
    const isJson = request.headers.get("content-type")?.includes("application/json");
    const shouldSanitize = ["POST", "PUT", "PATCH"].includes(method) && isJson && url.pathname.startsWith("/api/");

    if (!shouldSanitize) {
      return fetch(request);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      // Non-JSON; just proxy through.
      return fetch(request);
    }

    const sanitizeString = (s) => {
      // Escape intraword underscores only (a_b -> a\_b), but do NOT modify:
      // - inline code spans (backticks)
      // - fenced code blocks (``` or ~~~)
      // - Markdown link destinations [...](here)
      // - angle-bracket autolinks <https://...>
      // - bare URLs like http(s)://... or mailto:

      // 1) Split by fenced code blocks to avoid touching them.
      const fenceBlockRe = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
      const split = s.split(fenceBlockRe);
      const chunks = split.map(seg => ({ code: seg.startsWith('```') || seg.startsWith('~~~'), text: seg }));

      const processNonCode = (text) => {
        // Split by inline code spans
        const parts = text.split(/(`[^`]*`)/);

        const escapeOutsideLinksAndUrls = (frag) => {
          // Build preservation ranges for link destinations, angle autolinks, bare URLs
          const ranges = [];

          // Markdown links/images: capture (...) after ]
          const linkRe = /!\?\[[^\]]*\]\(([^)]+)\)/g;
          for (const match of frag.matchAll(linkRe)) {
            const url = match[1];
            // compute start/end of just the destination inside frag
            const full = match[0];
            const idx = match.index;
            if (idx == null) continue;
            const destStart = idx + full.indexOf("(") + 1;
            const destEnd = destStart + url.length;
            ranges.push([destStart, destEnd]);
          }

          // Angle-bracket autolinks: <...>
          const angleRe = /<[^>]+>/g;
          for (const match of frag.matchAll(angleRe)) {
            const start = match.index;
            if (start == null) continue;
            const end = start + match[0].length;
            ranges.push([start + 1, end - 1]); // inside <...>
          }

          // Bare URLs: http(s)://..., ftp://..., mailto:
          const urlRe = /\b(?:https?:\/\/|ftp:\/\/|mailto:)[^\s)]+/g;
          for (const match of frag.matchAll(urlRe)) {
            const start = match.index;
            if (start == null) continue;
            const end = start + match[0].length;
            ranges.push([start, end]);
          }

          // Merge overlapping ranges
          ranges.sort((a, b) => a[0] - b[0]);
          const merged = [];
          for (const r of ranges) {
            if (!merged.length || r[0] > merged[merged.length - 1][1]) merged.push(r);
            else merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], r[1]);
          }

          // Escape intraword underscores outside preserved ranges
          let out = "";
          for (let i = 0; i < frag.length; i++) {
            const ch = frag[i];
            const inPreserved = merged.some(([s, e]) => i >= s && i < e);
            if (ch === "_" && !inPreserved) {
              const prev = frag[i - 1] || "";
              const next = frag[i + 1] || "";
              if (/\w/.test(prev) && /\w/.test(next)) {
                out += "\\_";
                continue;
              }
            }
            out += ch;
          }
          return out;
        };

        return parts
          .map((part) => {
            if (part.startsWith("`") && part.endsWith("`")) return part;
            return escapeOutsideLinksAndUrls(part);
          })
          .join("");
      };

      return chunks
        .map((c) => (c.code ? c.text : processNonCode(c.text)))
        .join("");
    };

    const targetKeys = new Set(["title", "description", "comment", "body", "text"]);
    const maybeSanitize = (value, key) => {
      if (typeof value === "string" && (!key || targetKeys.has(key))) {
        return sanitizeString(value);
      }
      if (Array.isArray(value)) {
        return value.map((v) => maybeSanitize(v));
      }
      if (value && typeof value === "object") {
        const out = {};
        for (const [k, v] of Object.entries(value)) {
          out[k] = maybeSanitize(v, k);
        }
        return out;
      }
      return value;
    };

    const sanitized = maybeSanitize(body);

    const headers = new Headers(request.headers);
    headers.set("content-type", "application/json");
    const init = { method, headers, body: JSON.stringify(sanitized) };
    return fetch(new Request(request, init));
  },
};
