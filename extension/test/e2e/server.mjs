// Static server for the e2e tests: the test site, the model release assets (to exercise download + cache +
// checksum), and the speech test clips. Sends COEP/CORP headers so extension pages under COEP may load them.
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";

const ROOT = join(import.meta.dirname, "..", "..");
const MOUNTS = { "/site/": join(ROOT, "test/e2e/site"), "/assets/": join(ROOT, "release-assets"), "/audio/": join(ROOT, "test/fixtures/audio"), "/fixtures/": join(ROOT, "test/fixtures"), "/shots/": join(ROOT, "dist/store/screenshots") };
const TYPES = { ".html": "text/html", ".json": "application/json", ".onnx": "application/octet-stream", ".wav": "audio/wav", ".png": "image/png", ".js": "text/javascript" };

export function startServer(port = 0) {
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    const mount = Object.keys(MOUNTS).find((m) => url.pathname.startsWith(m));
    if (!mount) { res.writeHead(404); res.end(); return; }
    const p = normalize(join(MOUNTS[mount], decodeURIComponent(url.pathname.slice(mount.length))));
    if (!p.startsWith(MOUNTS[mount]) || !existsSync(p) || statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, {
      "content-type": TYPES[extname(p)] || "application/octet-stream", "content-length": statSync(p).size,
      "access-control-allow-origin": "*", "cross-origin-resource-policy": "cross-origin", "cache-control": "no-store",
    });
    if (req.method === "HEAD") { res.end(); return; }
    createReadStream(p).pipe(res);
  });
  return new Promise((r) => server.listen(port, "127.0.0.1", () => r({ server, port: server.address().port })));
}

if (process.argv[1] === import.meta.filename) {
  const { port } = await startServer(Number(process.argv[2] || 8737));
  console.log(`serving on http://127.0.0.1:${port}/site/shop.html`);
}
