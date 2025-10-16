// Cloudflare Worker: Inject Inter font into VibeKanban pages without modifying upstream.
// Deploy with `wrangler` using the wrangler.toml in this folder.
// Route: vibekanban.joernstoehler.com/*

export default {
  async fetch(request, env, ctx) {
    const response = await fetch(request);

    // Only rewrite HTML responses
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/html")) {
      return response;
    }

    const rewriter = new HTMLRewriter().on("head", {
      element(el) {
        el.append(
          `<link rel="preconnect" href="https://fonts.googleapis.com">`,
          { html: true },
        );
        el.append(
          `<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>`,
          { html: true },
        );
        el.append(
          `<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">`,
          { html: true },
        );
        // Override upstream body font (Chivo Mono) without touching their bundle.
        el.append(
          `<style>html,body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji" !important;}</style>`,
          { html: true },
        );
      },
    });
    return rewriter.transform(response);
  },
};
