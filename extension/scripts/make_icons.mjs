// Render the GenClass icon (rounded square, blue->violet gradient, white voice-wave bars with a cursor dot)
// to PNGs with no image libraries: signed-distance shapes, 4x4 supersampling, zlib PNG encoding.
import { mkdirSync, writeFileSync } from "node:fs";
import { deflateSync } from "node:zlib";
import { join } from "node:path";

const OUT = join(import.meta.dirname, "..", "static", "icons");
mkdirSync(OUT, { recursive: true });

function crc32(buf) {
  let c;
  const t = crc32.t || (crc32.t = Array.from({ length: 256 }, (_, n) => { c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; }));
  let crc = 0xffffffff;
  for (const b of buf) crc = t[(crc ^ b) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
export function png(w, h, rgba) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 4 + 1)] = 0; rgba.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4); }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw, { level: 9 })), chunk("IEND", Buffer.alloc(0))]);
}

const sdRoundBox = (x, y, cx, cy, hw, hh, r) => {
  const qx = Math.abs(x - cx) - hw + r, qy = Math.abs(y - cy) - hh + r;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
};

/** u, v in [0,1] -> [r,g,b,a] in 0..1 */
function shade(u, v, pad) {
  const inner = 0.5 - pad;
  const dBg = sdRoundBox(u, v, 0.5, 0.5, inner, inner, inner * 0.44);
  if (dBg > 0) return [0, 0, 0, 0];
  const t = Math.min(1, Math.max(0, (u + v) / 2));
  let col = [0.15 + 0.33 * t, 0.39 - 0.2 * t, 0.92 - 0.02 * t];
  const heights = [0.16, 0.3, 0.42, 0.26, 0.12];
  for (let i = 0; i < 5; i++) {
    const cx = 0.25 + i * 0.125;
    const d = sdRoundBox(u, v, cx, 0.5, 0.036, heights[i], 0.036);
    if (d < 0) col = [1, 1, 1];
  }
  const dDot = Math.hypot(u - 0.78, v - 0.78) - 0.065;
  if (dDot < 0) col = [1, 0.8, 0.25];
  return [...col, 1];
}

function render(size, pad = 0.04, ss = 4) {
  const buf = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let r = 0, g = 0, b = 0, a = 0;
    for (let sy = 0; sy < ss; sy++) for (let sx = 0; sx < ss; sx++) {
      const [cr, cg, cb, ca] = shade((x + (sx + 0.5) / ss) / size, (y + (sy + 0.5) / ss) / size, pad);
      r += cr * ca; g += cg * ca; b += cb * ca; a += ca;
    }
    const i = (y * size + x) * 4;
    const n = ss * ss;
    buf[i] = a ? Math.round((255 * r) / a) : 0; buf[i + 1] = a ? Math.round((255 * g) / a) : 0;
    buf[i + 2] = a ? Math.round((255 * b) / a) : 0; buf[i + 3] = Math.round((255 * a) / n);
  }
  return png(size, size, buf);
}

for (const s of [16, 32, 48, 128]) writeFileSync(join(OUT, `icon-${s}.png`), render(s, s <= 32 ? 0.02 : 0.06));
// Web Store icon: 128x128 with a 96x96 artwork area (16 px transparent padding), per the store guidelines.
writeFileSync(join(OUT, "store-icon-128.png"), render(128, 0.125));
console.log("icons written to", OUT);
