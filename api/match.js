/**
 * Vercel Edge proxy: browser POST /api/match → Railway POST /api/v1/match with API key.
 * Env: REVIEWERMATCH_API_URL (no trailing slash), REVIEWERMATCH_API_KEY
 */
export const config = { runtime: "edge" };

const DAILY_LIMIT = 3;
const WINDOW_MS = 24 * 60 * 60 * 1000;

/** @type {Map<string, { count: number, resetAt: number }>} */
const ipHits = new Map();

function clientIp(req) {
  const xf = req.headers.get("x-forwarded-for");
  if (xf) return xf.split(",")[0].trim();
  return req.headers.get("x-real-ip") || "unknown";
}

export default async function handler(req) {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  const base = process.env.REVIEWERMATCH_API_URL;
  const apiKey = process.env.REVIEWERMATCH_API_KEY;
  if (!base || !apiKey) {
    return new Response(
      JSON.stringify({
        error: "Server misconfigured",
        detail: "Set REVIEWERMATCH_API_URL and REVIEWERMATCH_API_KEY on Vercel.",
      }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }

  const ip = clientIp(req);
  const now = Date.now();
  let rec = ipHits.get(ip);
  if (!rec || now > rec.resetAt) {
    rec = { count: 0, resetAt: now + WINDOW_MS };
  }
  if (rec.count >= DAILY_LIMIT) {
    return new Response(
      JSON.stringify({ error: "Daily free limit reached" }),
      { status: 429, headers: { "Content-Type": "application/json" } }
    );
  }
  rec.count += 1;
  ipHits.set(ip, rec);

  const body = await req.text();
  const target = `${base.replace(/\/$/, "")}/api/v1/match`;

  const apiRes = await fetch(target, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
    },
    body,
  });

  const text = await apiRes.text();
  return new Response(text, {
    status: apiRes.status,
    headers: {
      "Content-Type": apiRes.headers.get("content-type") || "application/json",
    },
  });
}
