// Speech sources for GenClass. Each emits {kind: speech_start|partial|final|speech_end|error, uid, text, t}.
//
// - local engines (Moonshine, Whisper base.en, Whisper Turbo) run on-device with transformers.js: an energy VAD
//   cuts utterances; while one is open the whole utterance so far is re-transcribed every `stepMs` and emitted
//   as a partial (streaming by re-decoding); the end of speech emits the final.
// - "chrome" uses webkitSpeechRecognition with interim results. Audio is sent to Google's servers by Chrome.

export const SPEECH_ENGINES = {
  moonshine: {
    label: "Moonshine base (local, fastest)", model: "onnx-community/moonshine-base-ONNX", local: true,
    webgpu: { dtype: { encoder_model: "fp32", decoder_model_merged: "q4" }, mb: 154 },
    wasm: { dtype: { encoder_model: "q8", decoder_model_merged: "q8" }, mb: 63 },
    stepMs: 250, padTo30s: false,
  },
  chrome: { label: "Chrome built-in (cloud: audio goes to Google)", local: false, mb: 0 },
  "whisper-base": {
    label: "Whisper base.en (local)", model: "onnx-community/whisper-base.en", local: true,
    webgpu: { dtype: { encoder_model: "fp32", decoder_model_merged: "q4" }, mb: 206 },
    wasm: { dtype: { encoder_model: "q8", decoder_model_merged: "q8" }, mb: 77 },
    stepMs: 500, padTo30s: true,
  },
  "whisper-turbo": {
    label: "Whisper large-v3 Turbo (local, most accurate; WebGPU only)", model: "onnx-community/whisper-large-v3-turbo", local: true,
    requiresWebGPU: true,
    webgpu: { dtype: { encoder_model: "q4", decoder_model_merged: "q4" }, mb: 759 },
    webgpuF16: { dtype: { encoder_model: "q4f16", decoder_model_merged: "q4f16" }, mb: 564 },
    stepMs: 900, padTo30s: true, generate: { language: "english", task: "transcribe" },
  },
};

export const SAMPLE_RATE = 16000;

/** WebGPU availability and fp16 shader support. */
export async function gpuInfo() {
  try {
    if (!("gpu" in navigator)) return { webgpu: false, f16: false, why: "navigator.gpu is missing" };
    const a = await navigator.gpu.requestAdapter();
    if (!a) return { webgpu: false, f16: false, why: "no WebGPU adapter" };
    return { webgpu: true, f16: a.features.has("shader-f16"), adapter: (a.info && (a.info.vendor + " " + a.info.architecture)) || "" };
  } catch (e) {
    return { webgpu: false, f16: false, why: String(e) };
  }
}

/** Resolve device + dtype for an engine; throws a user-facing Error when the engine cannot run here. */
export function speechPlan(engine, gpu, pref = "auto") {
  const e = SPEECH_ENGINES[engine];
  if (!e) throw new Error(`unknown speech engine ${engine}`);
  if (!e.local) return { engine, local: false, mb: 0 };
  const useGpu = gpu.webgpu && pref !== "wasm";
  if (e.requiresWebGPU && !useGpu) {
    throw new Error(`${e.label} needs WebGPU, which is not available here (${gpu.why || "disabled in settings"}). Use Moonshine or Whisper base.en instead.`);
  }
  const v = useGpu ? (gpu.f16 && e.webgpuF16 ? e.webgpuF16 : e.webgpu) : e.wasm;
  return { engine, local: true, model: e.model, device: useGpu ? "webgpu" : "wasm", dtype: v.dtype, mb: v.mb, stepMs: e.stepMs, generate: e.generate || null };
}

// ------------------------------------------------------------------ Chrome built-in (Web Speech API)

export class WebSpeechSource {
  constructor(onEvent, { lang = "en-US" } = {}) {
    this.onEvent = onEvent;
    this.lang = lang;
    this.rec = null;
    this.on = false;
    this.session = 0;
    this.cadenceMs = null;
    this.stats = { firstPartialMs: [], utterances: 0 };
  }

  start() {
    const SR = globalThis.webkitSpeechRecognition || globalThis.SpeechRecognition;
    if (!SR) throw new Error("Web Speech API is not available in this context");
    this.on = true;
    this.session++;
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = this.lang;
    rec.maxAlternatives = 1;
    const sess = this.session;
    let speechT = null;
    const firstSeen = new Set();
    rec.onspeechstart = () => { speechT = performance.now(); this.onEvent({ kind: "speech_start", uid: `ws${sess}`, text: "" }); };
    rec.onresult = (ev) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        const text = r[0].transcript.trim();
        const uid = `ws${sess}-${i}`;
        if (text && !firstSeen.has(uid) && speechT !== null) {
          firstSeen.add(uid);
          this.stats.firstPartialMs.push(performance.now() - speechT);
          speechT = null;
        }
        if (r.isFinal) {
          this.stats.utterances++;
          this.onEvent({ kind: "final", uid, text });
        } else if (text) {
          this.onEvent({ kind: "partial", uid, text });
        }
      }
    };
    rec.onerror = (e) => {
      if (e.error === "no-speech" || e.error === "aborted") return;
      this.onEvent({ kind: "error", uid: "", text: "", error: e.error === "not-allowed" ? "mic_denied" : e.error });
      if (e.error === "not-allowed" || e.error === "service-not-allowed") this.on = false;
    };
    rec.onend = () => {
      this.onEvent({ kind: "speech_end", uid: `ws${sess}`, text: "" });
      if (this.on && this.rec === rec) setTimeout(() => { if (this.on && this.rec === rec) { this.rec = null; this.start(); } }, 50);
    };
    this.rec = rec;
    rec.start();
  }

  stop() {
    this.on = false;
    if (this.rec) { try { this.rec.abort(); } catch { /* already stopped */ } }
    this.rec = null;
  }
}

// ------------------------------------------------------------------ local ASR (transformers.js)

export class LocalAsr {
  /** transformers: the @huggingface/transformers module. */
  constructor(transformers, plan) {
    this.tf = transformers;
    this.plan = plan;
    this.pipe = null;
  }

  async load(onProgress) {
    const { pipeline } = this.tf;
    this.pipe = await pipeline("automatic-speech-recognition", this.plan.model, {
      device: this.plan.device,
      dtype: this.plan.dtype,
      progress_callback: (p) => onProgress && onProgress(p),
    });
    // Warm up (compiles WebGPU shaders) on a short silence.
    await this.transcribe(new Float32Array(SAMPLE_RATE / 2));
    return this;
  }

  async transcribe(audio) {
    const opts = this.plan.generate ? { ...this.plan.generate } : {};
    const out = await this.pipe(audio, opts);
    return clean((out && out.text) || "");
  }
}

function clean(t) {
  // Whisper/Moonshine hallucinations on silence and bracketed non-speech tags.
  const s = t.replace(/\[[^\]]*\]|\([^)]*\)/g, " ").replace(/\s+/g, " ").trim();
  if (/^(thank you\.?|thanks for watching!?|you|\.|bye\.?)$/i.test(s)) return "";
  return s;
}

/**
 * Microphone (or injected audio) -> VAD -> re-decoded partials -> finals.
 * feed(chunk) accepts 16 kHz mono Float32 chunks; startMic() wires getUserMedia + an AudioWorklet to it.
 */
export class LocalSpeechSource {
  constructor(onEvent, asr, { stepMs = 250, endSilenceMs = 600, maxUtteranceS = 25, preRollMs = 300 } = {}) {
    this.onEvent = onEvent;
    this.asr = asr;
    this.stepMs = stepMs;
    this.endSilenceMs = endSilenceMs;
    this.maxSamples = maxUtteranceS * SAMPLE_RATE;
    this.preRoll = Math.round((preRollMs / 1000) * SAMPLE_RATE);
    this.ring = [];
    this.ringLen = 0;
    this.utt = null; // {uid, chunks, len, startT, lastVoiceT, lastText, busy, firstPartialDone}
    this.n = 0;
    this.noise = 0.004;
    this.on = false;
    this.cadenceMs = stepMs;
    this.stats = { firstPartialMs: [], rtf: [], decodeMs: [], finalLagMs: [], utterances: 0 };
    this.timer = null;
    this.ctx = null;
    this.stream = null;
  }

  async startMic() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this.ctx.audioWorklet.addModule(chrome.runtime.getURL("audio-worklet.js"));
    const src = this.ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.ctx, "genclass-capture");
    node.port.onmessage = (e) => this.feed(e.data);
    src.connect(node);
    this.node = node;
    this.start();
  }

  start() {
    this.on = true;
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), Math.max(50, this.stepMs / 2));
  }

  async stop() {
    this.on = false;
    clearInterval(this.timer);
    if (this.utt) await this.finish();
    if (this.node) this.node.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) await this.ctx.close().catch(() => {});
    this.node = this.stream = this.ctx = null;
  }

  /** 16 kHz mono chunk (any length, ~100 ms works best). */
  feed(chunk) {
    if (!this.on) return;
    const now = performance.now();
    let e = 0;
    for (let i = 0; i < chunk.length; i++) e += chunk[i] * chunk[i];
    const rms = Math.sqrt(e / Math.max(1, chunk.length));
    const voiced = rms > Math.max(0.012, this.noise * 3.5);
    if (!voiced) this.noise = 0.95 * this.noise + 0.05 * Math.min(rms, 0.05);
    if (!this.utt) {
      this.ring.push(chunk);
      this.ringLen += chunk.length;
      while (this.ringLen - this.ring[0].length > this.preRoll) this.ringLen -= this.ring.shift().length;
      if (voiced) {
        const uid = `ls${++this.n}`;
        this.utt = { uid, chunks: [...this.ring], len: this.ringLen, startT: now, lastVoiceT: now, lastText: "", busy: false, lastDecodeLen: 0, firstPartialDone: false };
        this.ring = [];
        this.ringLen = 0;
        this.onEvent({ kind: "speech_start", uid, text: "" });
      }
      return;
    }
    const u = this.utt;
    u.chunks.push(chunk);
    u.len += chunk.length;
    if (voiced) u.lastVoiceT = now;
    if (now - u.lastVoiceT >= this.endSilenceMs || u.len >= this.maxSamples) this.finish();
  }

  audio() {
    const u = this.utt;
    const out = new Float32Array(u.len);
    let o = 0;
    for (const c of u.chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  async tick() {
    const u = this.utt;
    if (!u || u.busy || u.ending) return;
    if (u.len - u.lastDecodeLen < SAMPLE_RATE * (this.stepMs / 1000) * 0.8 && u.lastDecodeLen) return;
    u.busy = true;
    const audio = this.audio();
    u.lastDecodeLen = audio.length;
    const t0 = performance.now();
    let text = "";
    try {
      text = await this.asr.transcribe(audio);
    } catch (e) {
      this.onEvent({ kind: "error", uid: u.uid, text: "", error: `asr: ${e.message || e}` });
    }
    const ms = performance.now() - t0;
    this.stats.decodeMs.push(ms);
    this.stats.rtf.push(ms / ((audio.length / SAMPLE_RATE) * 1000));
    this.cadenceMs = Math.max(this.stepMs, Math.round(ms));
    u.busy = false;
    if (this.utt !== u || u.ending) return;
    if (text && text !== u.lastText) {
      if (!u.firstPartialDone) {
        u.firstPartialDone = true;
        this.stats.firstPartialMs.push(performance.now() - u.startT);
      }
      u.lastText = text;
      this.onEvent({ kind: "partial", uid: u.uid, text });
    }
  }

  async finish() {
    const u = this.utt;
    if (!u || u.ending) return;
    u.ending = true;
    const endT = performance.now();
    while (u.busy) await new Promise((r) => setTimeout(r, 10));
    const audio = this.audio();
    this.utt = null;
    let text = u.lastText;
    try {
      const t0 = performance.now();
      text = await this.asr.transcribe(audio);
      const ms = performance.now() - t0;
      this.stats.decodeMs.push(ms);
      this.stats.rtf.push(ms / ((audio.length / SAMPLE_RATE) * 1000));
    } catch (e) {
      this.onEvent({ kind: "error", uid: u.uid, text: "", error: `asr: ${e.message || e}` });
    }
    this.stats.finalLagMs.push(performance.now() - endT);
    this.stats.utterances++;
    if (text) this.onEvent({ kind: "final", uid: u.uid, text });
    this.onEvent({ kind: "speech_end", uid: u.uid, text: "" });
  }

  /** Play a whole clip through the pipeline at real-time pace (benchmarks, tests). */
  async playClip(samples, chunkMs = 100) {
    if (!this.on) this.start();
    const n = Math.round((chunkMs / 1000) * SAMPLE_RATE);
    const t0 = performance.now();
    for (let i = 0, k = 0; i < samples.length; i += n, k++) {
      this.feed(samples.subarray(i, Math.min(samples.length, i + n)));
      const due = t0 + (k + 1) * chunkMs;
      const wait = due - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
    // trailing silence so the VAD closes the utterance
    for (let k = 0; k < Math.ceil((this.endSilenceMs + 300) / chunkMs); k++) {
      this.feed(new Float32Array(n));
      await new Promise((r) => setTimeout(r, chunkMs));
    }
    while (this.utt) await new Promise((r) => setTimeout(r, 20));
  }
}

export function summarizeStats(s) {
  const med = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : null);
  return {
    firstPartialMs: med(s.firstPartialMs), decodeMsP50: med(s.decodeMs || []), rtfP50: med(s.rtf || []),
    finalLagMs: med(s.finalLagMs || []), utterances: s.utterances,
  };
}
