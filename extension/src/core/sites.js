// In a browser the trained `app` question is answered over open tabs and well-known sites:
// open_app "YouTube" switches to a YouTube tab, else opens youtube.com; quit_app closes that tab (always confirmed).

import { rankApps } from "./questions.js";

export const SITES = {
  Gmail: "https://mail.google.com/",
  YouTube: "https://www.youtube.com/",
  Google: "https://www.google.com/",
  "Google Docs": "https://docs.google.com/",
  "Google Drive": "https://drive.google.com/",
  "Google Calendar": "https://calendar.google.com/",
  "Google Maps": "https://maps.google.com/",
  Wikipedia: "https://en.wikipedia.org/",
  GitHub: "https://github.com/",
  Reddit: "https://www.reddit.com/",
  "X (Twitter)": "https://x.com/",
  Amazon: "https://www.amazon.com/",
  Netflix: "https://www.netflix.com/",
  Spotify: "https://open.spotify.com/",
  Slack: "https://app.slack.com/",
  Notion: "https://www.notion.so/",
  LinkedIn: "https://www.linkedin.com/",
  Outlook: "https://outlook.live.com/",
  "Hacker News": "https://news.ycombinator.com/",
  "Stack Overflow": "https://stackoverflow.com/",
  Twitch: "https://www.twitch.tv/",
  Figma: "https://www.figma.com/",
};

const BRANDS = {
  "mail.google.com": "Gmail", "youtube.com": "YouTube", "docs.google.com": "Google Docs", "drive.google.com": "Google Drive",
  "calendar.google.com": "Google Calendar", "maps.google.com": "Google Maps", "github.com": "GitHub", "reddit.com": "Reddit",
  "x.com": "X (Twitter)", "twitter.com": "X (Twitter)", "wikipedia.org": "Wikipedia", "amazon.com": "Amazon", "netflix.com": "Netflix",
  "spotify.com": "Spotify", "slack.com": "Slack", "notion.so": "Notion", "linkedin.com": "LinkedIn", "news.ycombinator.com": "Hacker News",
  "stackoverflow.com": "Stack Overflow", "twitch.tv": "Twitch", "figma.com": "Figma", "google.com": "Google",
};

export function brandOf(url) {
  try {
    const h = new URL(url).hostname.replace(/^www\./, "");
    for (const [d, b] of Object.entries(BRANDS)) if (h === d || h.endsWith("." + d)) return b;
    const parts = h.split(".");
    const core = parts.length > 1 ? parts[parts.length - 2] : parts[0];
    return core ? core[0].toUpperCase() + core.slice(1) : null;
  } catch {
    return null;
  }
}

/** Short tab name: the site brand, else the first segment of the title. */
export function tabName(tab) {
  const b = tab.url ? brandOf(tab.url) : null;
  if (b) return b;
  const t = (tab.title || "").split(/ [-|–—] /)[0].trim();
  return t.slice(0, 40) || null;
}

/** tabs: [{id, title, url, active}] -> {names, resolve(name) -> {tabId}|{url}|null} */
export function appCatalog(tabs, tailText, maxN = 24) {
  const byName = new Map();
  for (const t of tabs) {
    const n = tabName(t);
    if (n && !byName.has(n)) byName.set(n, { tabId: t.id });
  }
  for (const [n, url] of Object.entries(SITES)) if (!byName.has(n)) byName.set(n, { url });
  const tabNames = [...byName.entries()].filter(([, v]) => v.tabId !== undefined).map(([n]) => n);
  const names = rankApps(tailText, [...byName.keys()], tabNames, maxN);
  return { names, resolve: (name) => byName.get(name) || null };
}
