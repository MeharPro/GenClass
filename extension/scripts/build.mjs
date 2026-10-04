// Build GenClass into dist/genclass/ (load it with chrome://extensions -> "Load unpacked").
//   node scripts/build.mjs                 build
//   node scripts/build.mjs --zip           build + dist/genclass-<version>.zip (Web Store upload / sideload)
//   node scripts/build.mjs --bundle-model  also copy release-assets/genclass-*.onnx into the extension (offline use, tests)
//   node scripts/build.mjs --release       build + zip + dist/release/ (GitHub release assets, store listing, privacy policy)
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { cpSync, createReadStream, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

const ROOT = join(import.meta.dirname, "..");
const DIST = join(ROOT, "dist");
const OUT = join(DIST, "genclass");
const args = new Set(process.argv.slice(2));
const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const release = args.has("--release");

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const common = { bundle: true, minify: !args.has("--dev"), sourcemap: false, target: ["chrome116"], legalComments: "none", logLevel: "warning" };
await build({ ...common, entryPoints: [join(ROOT, "src/background/sw.js")], outfile: join(OUT, "sw.js"), format: "esm" });
await build({ ...common, entryPoints: [join(ROOT, "src/content/content.js")], outfile: join(OUT, "content.js"), format: "iife" });
await build({ ...common, entryPoints: [join(ROOT, "src/sidepanel/panel.js")], outfile: join(OUT, "panel.js"), format: "esm" });
await build({ ...common, entryPoints: [join(ROOT, "src/sidepanel/welcome.js")], outfile: join(OUT, "welcome.js"), format: "esm" });
// One onnxruntime-web instance for the decision model and transformers.js; its wasm loader is fetched from ort/.
await build({
  ...common, entryPoints: [join(ROOT, "src/offscreen/offscreen.js")], outfile: join(OUT, "offscreen.js"), format: "esm",
  conditions: ["onnxruntime-web-use-extern-wasm"], mainFields: ["browser", "module", "main"], platform: "browser",
});

// transformers.js defaults its wasm path to a CDN; GenClass ships the files in ort/ (MV3: no remotely hosted code).
{
  const f = join(OUT, "offscreen.js");
  const src = readFileSync(f, "utf8");
  const fixed = src.replace(/https:\/\/cdn\.jsdelivr\.net\/npm\/onnxruntime-web@\$\{[^}]+\}\/dist\//g, "/ort/");
  if (/cdn\.jsdelivr\.net/.test(fixed)) throw new Error("remote code URL left in offscreen.js");
  writeFileSync(f, fixed);
}

cpSync(join(ROOT, "static"), OUT, { recursive: true });
rmSync(join(OUT, "icons", "store-icon-128.png"), { force: true });
const manifest = JSON.parse(readFileSync(join(ROOT, "static/manifest.json"), "utf8"));
manifest.version = pkg.version;
writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));

mkdirSync(join(OUT, "ort"), { recursive: true });
const ORT = join(ROOT, "node_modules/onnxruntime-web/dist");
for (const f of ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) cpSync(join(ORT, f), join(OUT, "ort", f));

mkdirSync(join(OUT, "model"), { recursive: true });
for (const f of ["model.json", "tokenizer.json", "calibration.json", "meta.json"]) cpSync(join(ROOT, "src/model", f), join(OUT, "model", f));
if (args.has("--bundle-model")) {
  for (const v of ["q8", "fp16"]) {
    const f = join(ROOT, "release-assets", `genclass-${v}.onnx`);
    if (existsSync(f)) cpSync(f, join(OUT, "model", `genclass-${v}.onnx`));
  }
}
cpSync(join(ROOT, "LICENSE"), join(OUT, "LICENSE"));
cpSync(join(ROOT, "NOTICE"), join(OUT, "NOTICE"));

const size = (p) => statSync(p).size;
console.log(`built ${OUT}`);

async function sha256(path) {
  const h = createHash("sha256");
  await new Promise((res, rej) => createReadStream(path).on("data", (d) => h.update(d)).on("end", res).on("error", rej));
  return h.digest("hex");
}

if (args.has("--zip") || release) {
  const zip = join(DIST, `genclass-${pkg.version}.zip`);
  rmSync(zip, { force: true });
  execFileSync("zip", ["-q", "-r", "-X", zip, "."], { cwd: OUT });
  console.log(`zip ${zip} (${(size(zip) / 1e6).toFixed(1)} MB)`);
}

if (release) {
  const R = join(DIST, "release");
  rmSync(R, { recursive: true, force: true });
  mkdirSync(R, { recursive: true });
  const card = JSON.parse(readFileSync(join(ROOT, "src/model/model.json"), "utf8"));
  const assets = [];
  for (const v of Object.values(card.variants)) {
    const src = join(ROOT, "release-assets", v.file);
    if (!existsSync(src)) { console.warn(`missing ${src}`); continue; }
    cpSync(src, join(R, v.file));
    const h = await sha256(src);
    if (h !== v.sha256) throw new Error(`sha256 of ${v.file} does not match src/model/model.json`);
    assets.push({ name: v.file, bytes: size(src), sha256: h });
  }
  for (const f of ["tokenizer.json", "calibration.json", "meta.json", "model.json"]) {
    cpSync(join(ROOT, "src/model", f), join(R, f));
    assets.push({ name: f, bytes: size(join(R, f)), sha256: await sha256(join(R, f)) });
  }
  const zipName = `genclass-${pkg.version}.zip`;
  cpSync(join(DIST, zipName), join(R, zipName));
  assets.push({ name: zipName, bytes: size(join(R, zipName)), sha256: await sha256(join(R, zipName)) });
  const speech = {
    "onnx-community/moonshine-base-ONNX": { webgpu: ["onnx/encoder_model.onnx (80.8 MB)", "onnx/decoder_model_merged_q4.onnx (72.8 MB)"], wasm: ["onnx/encoder_model_quantized.onnx (20.5 MB)", "onnx/decoder_model_merged_quantized.onnx (42.5 MB)"] },
    "onnx-community/whisper-base.en": { webgpu: ["onnx/encoder_model.onnx (82.5 MB)", "onnx/decoder_model_merged_q4.onnx (123.6 MB)"], wasm: ["onnx/encoder_model_quantized.onnx (23.2 MB)", "onnx/decoder_model_merged_quantized.onnx (53.7 MB)"] },
    "onnx-community/whisper-large-v3-turbo": { webgpu_f16: ["onnx/encoder_model_q4f16.onnx (370.0 MB)", "onnx/decoder_model_merged_q4f16.onnx (193.5 MB)"], webgpu: ["onnx/encoder_model_q4.onnx (424.9 MB)", "onnx/decoder_model_merged_q4.onnx (334.1 MB)"] },
  };
  writeFileSync(join(R, "ASSETS.json"), JSON.stringify({ tag: card.release_tag, repo: "https://github.com/MeharPro/GenClass",
    download_base: card.default_base_url, assets, speech_models_fetched_at_runtime_from_huggingface: speech }, null, 2));
  cpSync(join(ROOT, "store"), join(DIST, "store"), { recursive: true });
  writeFileSync(join(DIST, "store", "privacy.html"), mdToHtml(readFileSync(join(ROOT, "store", "privacy.md"), "utf8"), "GenClass privacy policy"));
  console.log(`release assets in ${R}`);
}

/** Minimal Markdown -> HTML for the privacy page (headings, lists, tables, links, bold, italics, paragraphs). */
function mdToHtml(md, title) {
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (t) => esc(t).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/_([^_]+)_/g, "<em>$1</em>")
    .replace(/(https?:\/\/[^\s)|]+)/g, '<a href="$1">$1</a>');
  const out = [];
  const lines = md.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (/^#{1,3} /.test(l)) { const n = l.match(/^#+/)[0].length; out.push(`<h${n}>${inline(l.slice(n + 1))}</h${n}>`); }
    else if (/^- /.test(l)) { const items = []; while (i < lines.length && /^- /.test(lines[i])) items.push(`<li>${inline(lines[i++].slice(2))}</li>`); i--; out.push(`<ul>${items.join("")}</ul>`); }
    else if (/^\|/.test(l)) {
      const rows = []; while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++]); i--;
      const cells = (r) => r.split("|").slice(1, -1).map((c) => c.trim());
      out.push(`<table><thead><tr>${cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.slice(2).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    } else if (l.trim()) out.push(`<p>${inline(l)}</p>`);
  }
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${title}</title>
<style>:root{color-scheme:light dark}body{font:16px/1.55 system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #8884;padding:6px 8px;text-align:left;vertical-align:top}</style></head><body>${out.join("\n")}</body></html>`;
}
