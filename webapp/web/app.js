/* Neat DINO Studio — page logic (plain JS, no build step). */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const store = {
  get(k, d) { try { const v = localStorage.getItem("dino-" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("dino-" + k, JSON.stringify(v)); } catch (e) { /* storage blocked */ } },
};

const EXAMPLES = [
  "person . dog . cat .", "cup . bottle . laptop . phone .", "car . bus . traffic light . bicycle .",
  "chair . table . plant . lamp .", "red apple . banana . orange .", "person wearing glasses .",
];
const TIMING_PARTS = [
  ["decode", "Decode", "#7a828d"], ["text", "Text", "#b36bff"], ["backbone", "Backbone", "#3a86ec"],
  ["backbone_cpu", "Backbone CPU", "#8f9bff"],
  ["encoder_mla", "Encoder MLA", "#16c8a6"], ["encoder_cpu", "Encoder CPU", "#4bb54a"],
  ["dec_values", "Values", "#1fc3e8"], ["topk", "Top-K", "#a9c81c"],
  ["decoder_mla", "Decoder MLA", "#f2801d"], ["decoder_cpu", "Decoder CPU", "#ffc21a"],
];

const S = {
  status: null, result: null, imageBlob: null, imageId: null, imageSent: null, sourceW: 0, sourceH: 0,
  hidden: new Set(), busy: false, cam: null, live: false, facing: "environment", devices: [],
  hover: -1, lastFrameT: 0, fps: [], pending: false,
};

// --------------------------------------------------------------------- api
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((body && body.error) || `${r.status} ${r.statusText}`);
  return body;
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

function toast(msg, err = false, ms = 3800) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", err);
  t.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.hidden = true), ms);
}

function confirmBox(text, ok = "OK") {
  return new Promise((res) => {
    const d = $("#confirm");
    $("#confirm-text").textContent = text;
    $("#confirm-yes").textContent = ok;
    d.onclose = () => res(d.returnValue === "yes");
    d.showModal();
  });
}

const fmtBytes = (b) => (b == null ? "?" : b >= 1 << 30 ? (b / (1 << 30)).toFixed(1) + " GB" : Math.round(b / (1 << 20)) + " MB");

// --------------------------------------------------------------------- status / model pill
async function refreshStatus() {
  try {
    S.status = await api("/api/status");
  } catch (e) {
    S.status = null;
    $("#hero-model-text").textContent = "Studio server not answering";
    $("#hero-model").className = "hero-model error";
  }
  renderStatus();
  clearTimeout(refreshStatus.t);
  const vs = S.status && S.status.voice;
  const loading = S.status && (S.status.state === "loading" || (vs && (vs.state === "loading" || (vs.models || []).some((m) => m.job))));
  refreshStatus.t = setTimeout(refreshStatus, loading ? 800 : 4000);
}

function renderStatus() {
  const st = S.status;
  const dot = $("#model-dot"), name = $("#model-name"), ms = $("#model-ms"), banner = $("#banner");
  dot.className = "dot";
  ms.textContent = "";
  if (!st) {
    name.textContent = "Offline";
    dot.classList.add("error");
    showBanner("The studio server is not answering.", true);
    return;
  }
  $("#assist-row").hidden = !(st.assist && st.assist.available);
  const info = st.model_info;
  const narrow = matchMedia("(max-width: 600px)").matches;
  $("#model-pill").title = info ? `${info.title}\n${info.details}` : "Models";
  $("#det-model").textContent = info && st.state === "ready" ? info.title : "";
  $("#chip-model").textContent = info && st.state === "ready" ? info.short || info.title : "";
  if (st.state === "ready") {
    dot.classList.add("ready");
    name.textContent = narrow ? info.short : info.title;
    if (st.last_ms) ms.textContent = `${Math.round(st.last_ms)} ms`;
    banner.hidden = true;
  } else if (st.state === "loading" || st.state === "unloading") {
    dot.classList.add("loading");
    const what = info ? info.short : "model";
    name.textContent = st.state === "loading" ? `Loading ${what}…` : `Unloading ${what}…`;
    showProgressBanner(st);
  } else if (st.state === "error") {
    dot.classList.add("error");
    name.textContent = "Load Failed";
    showBanner(st.error || "Loading the model failed.", true, "Open Models");
  } else {
    name.textContent = "No Model";
    showBanner("No model is loaded. Load one in Settings → Models to start detecting.", false, "Open Models");
  }
  const hm = $("#hero-model"), hmt = $("#hero-model-text");
  hm.className = "hero-model " + ({ ready: "ready", loading: "loading", unloading: "loading", error: "error" }[st.state] || "");
  hmt.textContent = st.state === "ready" ? `${narrow ? info.short : info.title} · Ready`
    : st.state === "loading" ? `Loading ${info ? info.short : "the model"}…`
    : st.state === "error" ? "Model failed to load · Open Models" : "No model loaded · Open Models";
  $("#about-status").textContent = `Server ${st.version}, backend ${st.backend}` + (st.size ? `, input ${st.size} px` : "") +
    (st.load_seconds ? `, loaded in ${st.load_seconds} s` : "");
  renderVoice();
}

// Model loading / unloading: what is happening, with a bar that moves.
function showProgressBanner(st) {
  const b = $("#banner");
  const p = st.progress || {};
  const info = st.model_info;
  b.hidden = false;
  b.classList.remove("err");
  b.classList.add("progress-banner");
  b.innerHTML = `<span class="spinner"></span><div class="pb-text"><div class="pb-title"></div><div class="pb-detail muted small"></div>
    <div class="pbar"><span></span></div></div><span class="pb-time muted small"></span>`;
  b.querySelector(".pb-title").textContent =
    `${st.state === "loading" ? "Loading" : "Unloading"} ${info ? info.title : "the model"} — ${p.label || "starting"}`;
  b.querySelector(".pb-detail").textContent = p.detail || (info ? info.details : "");
  setBar(b.querySelector(".pbar"), p.fraction);
  if (st.loading_seconds != null) b.querySelector(".pb-time").textContent = `${Math.round(st.loading_seconds)} s`;
}

function setBar(bar, fraction) {
  const span = bar.querySelector("span");
  if (fraction == null) {
    bar.classList.add("indeterminate");
    span.style.width = "";
  } else {
    bar.classList.remove("indeterminate");
    span.style.width = `${Math.max(2, Math.min(100, fraction * 100)).toFixed(1)}%`;
  }
}

function showBanner(text, err = false, action = null) {
  const b = $("#banner");
  b.hidden = false;
  b.classList.remove("progress-banner");
  b.classList.toggle("err", err);
  b.innerHTML = "";
  const s = document.createElement("span");
  s.textContent = text;
  b.append(s);
  if (action) {
    const btn = document.createElement("button");
    btn.className = "btn subtle";
    btn.textContent = action;
    btn.onclick = () => openSettings("models");
    b.append(btn);
  }
}

function prettyModel(n) {
  if (!n) return "No Model";
  const t = /dino-tiny-(\d+)-(\w+)/.exec(n);
  if (t) return `Grounding DINO Tiny · ${t[1]} px · ${t[2]}`;
  const m = /dino-(\d+)-(\w+)/.exec(n);
  if (!m) return n;
  return `Grounding DINO Base · ${m[1] === "768" ? "Accurate" : m[1] === "576" ? "Fast" : ""} ${m[1]} px · ${m[2]}`;
}

// --------------------------------------------------------------------- prompt
const promptEl = () => $("#prompt");
let parseTimer = null;
function onPromptInput() {
  clearTimeout(parseTimer);
  parseTimer = setTimeout(parsePrompt, 250);
}
async function parsePrompt() {
  const text = promptEl().value;
  store.set("prompt", text);
  if (!text.trim()) {
    $("#tokens").textContent = "";
    $("#phrase-chips").innerHTML = "";
    return;
  }
  try {
    const p = await post("/api/prompt/parse", { text });
    $("#tokens").textContent = `${p.tokens}/${p.max_tokens} Tokens`;
    $("#tokens").classList.toggle("full", p.truncated.length > 0);
    renderPhraseChips(p.phrases, null);
    const hint = $("#prompt-hint");
    if (p.truncated.length) {
      hint.textContent = `Too long: ${p.truncated.join(", ")} won't be searched. Split it into two runs.`;
      hint.classList.add("warn");
    } else {
      hint.textContent = "Separate things with a period. Each phrase gets its own color.";
      hint.classList.remove("warn");
    }
  } catch (e) { /* no model or tokenizer yet: fine */ }
}

function renderPhraseChips(phrases, counts) {
  const box = $("#phrase-chips");
  box.innerHTML = "";
  phrases.forEach((p, i) => {
    const c = document.createElement("button");
    c.className = "chip" + (S.hidden.has(p.text) ? " off" : "");
    c.innerHTML = `<span class="sw"></span><span></span>`;
    c.querySelector(".sw").style.background = p.color;
    c.children[1].textContent = p.text;
    if (counts) {
      const n = document.createElement("span");
      n.className = "n";
      n.textContent = counts[i] || 0;
      c.append(n);
    }
    c.title = "Show or hide this phrase";
    c.onclick = () => {
      S.hidden.has(p.text) ? S.hidden.delete(p.text) : S.hidden.add(p.text);
      draw();
    };
    box.append(c);
  });
}

// Landing-page samples: COCO val2017 photos under CC BY 2.0 (credits in samples.json and About).
async function renderSamples() {
  let list = [];
  try {
    list = await (await fetch("/static/samples/samples.json")).json();
  } catch (e) {
    return;
  }
  const box = $("#samples");
  box.innerHTML = "";
  for (const s of list) {
    const b = document.createElement("button");
    b.className = "sample";
    b.title = `Find ${s.prompt}`;
    b.innerHTML = `<img alt="" loading="lazy" decoding="async"><span class="s-cap"><b></b><code></code></span>`;
    b.querySelector("img").src = s.thumb;
    b.querySelector("b").textContent = s.title;
    b.querySelector("code").textContent = s.prompt;
    b.onclick = () => useSample(s);
    box.append(b);
  }
  const cred = $("#sample-credits");
  cred.innerHTML = "Sample photos: COCO val2017 (Flickr), " + list.map((s) =>
    `<a href="${esc(s.source)}" target="_blank" rel="noopener">${esc(s.title)}</a>`).join(", ") +
    ` — <a href="https://creativecommons.org/licenses/by/2.0/" target="_blank" rel="noopener">CC BY 2.0</a>, resized.`;
}

async function useSample(s) {
  try {
    const blob = await (await fetch(s.image)).blob();
    promptEl().value = s.prompt;
    parsePrompt();
    await useFile(new File([blob], s.key + ".jpg", { type: "image/jpeg" }));
  } catch (e) {
    toast(`Could not open the sample: ${e.message}`, true);
  }
}

function renderExamples() {
  const box = $("#examples");
  const hist = store.get("history", []);
  const items = [...hist.slice(0, 4), ...EXAMPLES.filter((e) => !hist.includes(e))].slice(0, 8);
  box.innerHTML = "";
  for (const t of items) {
    const c = document.createElement("button");
    c.className = "chip example";
    c.textContent = t;
    c.onclick = () => {
      promptEl().value = t;
      parsePrompt();
      if (S.imageBlob || S.cam) detect();
    };
    box.append(c);
  }
}

function remember(text) {
  const hist = store.get("history", []).filter((h) => h !== text);
  hist.unshift(text);
  store.set("history", hist.slice(0, 12));
  renderExamples();
}

// --------------------------------------------------------------------- images
function showViewer(kind) {
  $("#drop").hidden = true;
  $("#viewer").hidden = false;
  $("#photo").hidden = kind !== "photo";
  $("#video").hidden = kind !== "video";
}

async function useFile(file) {
  if (!file || !file.type.startsWith("image/")) return toast("That is not an image.", true);
  stopCam();
  const url = URL.createObjectURL(file);
  const img = $("#photo");
  img.onload = () => {
    S.sourceW = img.naturalWidth;
    S.sourceH = img.naturalHeight;
    URL.revokeObjectURL(url);
    fitOverlay();
  };
  img.src = url;
  showViewer("photo");
  S.imageBlob = await downscale(file);
  S.imageId = null;
  S.result = null;
  draw();
  if (promptEl().value.trim()) detect();
}

// Send the server what the model will see anyway (long side = model input), not 12 MP.
async function downscale(file) {
  const size = (S.status && S.status.size) || 768;
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const s = Math.min(1, (size * 1.25) / Math.max(bmp.width, bmp.height));
    if (s >= 1 && file.size < 3e6) {
      S.sentScale = 1;
      return file;
    }
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * s);
    c.height = Math.round(bmp.height * s);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    S.sentScale = 1 / s;
    return await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
  } catch (e) {
    S.sentScale = 1;
    return file;
  }
}

// --------------------------------------------------------------------- detect
async function detect(opts = {}) {
  const text = promptEl().value.trim();
  if (!text) return toast("Type what to find first, e.g. “person . dog .”");
  if (!S.status || S.status.state !== "ready") return toast("Load a model first (Settings → Models).", true);
  let blob = opts.blob || null;
  if (!blob && !S.cam) {
    if (!S.imageBlob) return toast("Add a photo or open the webcam first.");
    blob = S.imageId ? null : S.imageBlob;
  }
  if (!blob && S.cam) blob = await grabFrame();
  if (S.busy && !opts.live) return;
  S.busy = true;
  setBusy(true);
  const fd = new FormData();
  fd.append("prompt", text);
  if ($("#nms").checked) fd.append("nms", "0.5");
  if (blob) fd.append("image", blob, "image.jpg");
  else fd.append("image_id", S.imageId);
  try {
    const r = await api("/api/detect", { method: "POST", body: fd });
    if (opts.live) {
      // two frames are in flight: show results in order, drop any that arrive late
      if (!S.live || opts.seq < S.liveShown) return;
      S.liveShown = opts.seq;
      const now = performance.now();
      S.fps.push(now - S.lastShownT);
      S.lastShownT = now;
      if (S.fps.length > 10) S.fps.shift();
      const avg = S.fps.reduce((a, b) => a + b, 0) / S.fps.length;
      $("#fps").innerHTML = `${(1000 / avg).toFixed(1)}<small>FPS</small>`;
      $("#mi-fps").title = `${Math.round(avg)} ms between results, ${Math.round(r.timing.server_total)} ms per frame on the board`;
    }
    S.result = r;
    if (!S.cam) S.imageId = r.image_id;
    if (!opts.live) remember(text);
    draw();
    renderTiming(r.timing);
    if (S.status) S.status.last_ms = r.timing.server_total;
    renderStatus();
  } catch (e) {
    if (/gone from the board/.test(e.message) && S.imageBlob) {
      S.imageId = null;
      S.busy = false;
      setBusy(false);
      return detect();
    }
    if (!opts.live) toast(e.message, true);
    else stopLive(e.message);
  } finally {
    S.busy = false;
    setBusy(false);
  }
}

function setBusy(on) {
  $("#busy").hidden = !on || S.live;
  $("#detect-btn").disabled = on && !S.live;
}

// Top metric strip: total, then the time split into the model's main parts.
const PART_KEYS = {
  text: ["text"], backbone: ["backbone", "backbone_cpu"], encoder: ["encoder_mla", "encoder_cpu"],
  decoder: ["dec_values", "topk", "decoder_mla", "decoder_cpu"],
};
function renderTiming(t) {
  $("#chipbar").hidden = false;
  const total = t.server_total || t.total || 0;
  const label = Object.fromEntries(TIMING_PARTS.map(([k, l]) => [k, l]));
  const fmt = (ms) => `${Math.round(ms)}<small>ms</small>`;
  let shown = 0;
  for (const [part, keys] of Object.entries(PART_KEYS)) {
    const ms = keys.reduce((a, k) => a + (t[k] || 0), 0);
    shown += ms;
    const el = $(`#mp-${part}`);
    el.classList.toggle("idle", !ms);
    el.querySelector(".metric-value").innerHTML = ms ? fmt(ms) : part === "text" ? "cached" : "—";
    el.title = el.title.split("\n")[0] + (ms ? "\n" + keys.filter((k) => t[k]).map((k) => `${label[k]}: ${t[k].toFixed(1)} ms`).join("\n") : "");
  }
  const other = Math.max(0, total - shown);
  $("#mp-other .metric-value").innerHTML = total ? fmt(other) : "—";
  $("#m-total").innerHTML = total ? fmt(total) : "—";
  if (!S.live) flash($("#m-total").parentElement);
}
function flash(el) {
  el.classList.add("flash");
  clearTimeout(el._flash);
  el._flash = setTimeout(() => el.classList.remove("flash"), 600);
}

// --------------------------------------------------------------------- drawing
function visibleDetections() {
  const r = S.result;
  if (!r) return [];
  const bt = +$("#box-thr").value, tt = +$("#text-thr").value;
  return r.detections.filter((d) => d.score >= bt && d.phrase >= 0 && d.phrase_scores[d.phrase] >= tt &&
    !S.hidden.has(r.phrases[d.phrase].text));
}

function fitOverlay() {
  const media = S.cam ? $("#video") : $("#photo");
  // Scale the photo or video to fill the stage (up or down), keeping its aspect.
  const box = $("#viewer").getBoundingClientRect();
  const nw = S.cam ? media.videoWidth : media.naturalWidth, nh = S.cam ? media.videoHeight : media.naturalHeight;
  if (nw && nh && box.width && box.height) {
    const k = Math.min(box.width / nw, box.height / nh);
    media.style.width = `${Math.floor(nw * k)}px`;
    media.style.height = `${Math.floor(nh * k)}px`;
  }
  const cv = $("#overlay"), stage = box, m = media.getBoundingClientRect();
  cv.style.left = `${m.left - stage.left}px`;
  cv.style.top = `${m.top - stage.top}px`;
  cv.style.width = `${m.width}px`;
  cv.style.height = `${m.height}px`;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(m.width * dpr);
  cv.height = Math.round(m.height * dpr);
  draw();
}

function draw() {
  const cv = $("#overlay"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const r = S.result;
  const dets = visibleDetections();
  renderList(dets);
  if (!r || !cv.width) return;
  const mirror = S.cam && $("#video").classList.contains("mirror");
  const sx = cv.width / r.width, sy = cv.height / r.height;
  const dpr = window.devicePixelRatio || 1;
  const lw = +$("#line-w").value * dpr, fs = +$("#label-size").value * dpr;
  const fill = $("#fill-boxes").checked, scores = $("#show-scores").checked;
  ctx.font = `600 ${fs}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  dets.forEach((d, i) => {
    const color = r.phrases[d.phrase].color;
    let [x0, y0, x1, y1] = d.box;
    if (mirror) [x0, x1] = [r.width - x1, r.width - x0];
    const X = x0 * sx, Y = y0 * sy, W = (x1 - x0) * sx, H = (y1 - y0) * sy;
    const hl = i === S.hover;
    ctx.lineWidth = hl ? lw * 1.8 : lw;
    ctx.strokeStyle = color;
    if (fill || hl) {
      ctx.fillStyle = color + (hl ? "33" : "1f");
      ctx.fillRect(X, Y, W, H);
    }
    ctx.strokeRect(X, Y, W, H);
    const label = r.phrases[d.phrase].text + (scores ? ` ${d.score.toFixed(2)}` : "");
    const tw = ctx.measureText(label).width + 10 * dpr, th = fs + 6 * dpr;
    const ly = Y - th >= 0 ? Y - th : Y;
    ctx.fillStyle = color;
    ctx.fillRect(X - lw / 2, ly, tw, th);
    ctx.fillStyle = "#0c1016";
    ctx.fillText(label, X + 5 * dpr - lw / 2, ly + 3 * dpr);
  });
}

function renderList(dets) {
  const r = S.result, list = $("#det-list"), legend = $("#legend");
  list.innerHTML = "";
  legend.innerHTML = "";
  if (!r) {
    list.innerHTML = '<li class="empty">Nothing yet.</li>';
    $("#det-count").textContent = "";
    return;
  }
  const counts = r.phrases.map(() => 0);
  for (const d of dets) counts[d.phrase]++;
  renderPhraseChips(r.phrases, counts);
  $("#det-count").textContent = `${dets.length} Found`;
  if (!dets.length) list.innerHTML = '<li class="empty">Nothing above the thresholds. Lower Box Score to see weaker matches.</li>';
  dets.forEach((d, i) => {
    const li = document.createElement("li");
    const [x0, y0, x1, y1] = d.box;
    li.innerHTML = `<span class="sw"></span><span></span><span class="sc"></span>`;
    li.children[0].style.background = r.phrases[d.phrase].color;
    li.children[1].textContent = r.phrases[d.phrase].text;
    li.children[2].textContent = `${d.score.toFixed(2)} · ${Math.round(x1 - x0)}×${Math.round(y1 - y0)}`;
    li.onmouseenter = () => { S.hover = i; draw(); };
    li.onmouseleave = () => { S.hover = -1; draw(); };
    list.append(li);
  });
}

// --------------------------------------------------------------------- webcam
function camSupported() {
  return window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
}

async function openCam() {
  if (!camSupported()) {
    const url = `https://${location.hostname}:${location.port || (S.status && S.status.port) || 8092}${location.pathname}`;
    showBanner("Browsers only allow the webcam on secure pages.", true);
    const a = document.createElement("a");
    a.href = url;
    a.textContent = "Open The Secure Page";
    a.className = "btn subtle";
    $("#banner").append(a);
    return;
  }
  try {
    stopCam();
    const stream = await getCamStream();
    const v = $("#video");
    v.srcObject = stream;
    await v.play();
    S.cam = stream;
    S.imageBlob = null;
    S.imageId = null;
    S.result = null;
    v.classList.toggle("mirror", shouldMirror(stream));
    showViewer("video");
    v.onloadedmetadata = fitOverlay;
    setTimeout(fitOverlay, 300);
    ["#snap-btn", "#live-btn"].forEach((s) => ($(s).hidden = false));
    S.devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    $("#flip-btn").hidden = S.devices.length < 2;
    $("#cam-btn").innerHTML = '<svg><use href="#i-stop"/></svg>Close Webcam';
  } catch (e) {
    const why = { NotAllowedError: "Camera permission was denied.", NotFoundError: "No camera found.",
      NotReadableError: "The camera is in use by another app." }[e.name] || e.message;
    toast(why, true);
  }
}

// Camera and microphone choices (Settings → Devices). An unplugged choice falls back to the default device.
function camWant() {
  const want = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const id = store.get("cam-device", "");
  if (id) want.deviceId = { exact: id };
  else want.facingMode = S.facing;
  return want;
}
function micWant() {
  const want = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
  const id = store.get("mic-device", "");
  if (id) want.deviceId = { exact: id };
  return want;
}
async function withFallback(kind, want) {
  try {
    return await navigator.mediaDevices.getUserMedia({ [kind]: want });
  } catch (e) {
    if (!want.deviceId || !["OverconstrainedError", "NotFoundError"].includes(e.name)) throw e;
    delete want.deviceId;
    if (kind === "video") want.facingMode = S.facing;
    toast(`The chosen ${kind === "video" ? "camera" : "microphone"} isn't connected, so the default one is used.`);
    return navigator.mediaDevices.getUserMedia({ [kind]: want });
  }
}
const getCamStream = () => withFallback("video", camWant());
const getMicStream = () => withFallback("audio", micWant());
function shouldMirror(stream) {
  const track = stream && stream.getVideoTracks()[0];
  const facing = (track && track.getSettings && track.getSettings().facingMode) || "";
  return $("#mirror-cam").checked && facing !== "environment";
}

function stopCam() {
  stopLive();
  if (S.cam) S.cam.getTracks().forEach((t) => t.stop());
  S.cam = null;
  ["#snap-btn", "#live-btn", "#flip-btn"].forEach((s) => ($(s).hidden = true));
  $("#cam-btn").innerHTML = '<svg><use href="#i-webcam"/></svg>Webcam';
}

// A camera frame at the model's input size (what Live sends), or at the camera's full resolution.
async function grabFrame(full = false) {
  const v = $("#video");
  const size = (S.status && S.status.size) || 768;
  const s = full ? 1 : Math.min(1, size / Math.max(v.videoWidth, v.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(v.videoWidth * s);
  c.height = Math.round(v.videoHeight * s);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return await new Promise((r) => c.toBlob(r, "image/jpeg", full ? 0.95 : 0.88));
}

// Show the snapshot at full camera resolution; the board gets a copy at the model's size, as for uploads.
async function snapshot() {
  if (!S.cam) return;
  const photo = await grabFrame(true);
  const url = URL.createObjectURL(photo);
  stopCam();
  const img = $("#photo");
  img.onload = () => { S.sourceW = img.naturalWidth; S.sourceH = img.naturalHeight; fitOverlay(); };
  img.src = url;
  showViewer("photo");
  S.imageBlob = await downscale(photo);
  S.imageId = null;
  detect();
}

// Keep as many frames in flight as the runtime runs at once (two on the board: one on the
// MLA while the CPU samples the other); the newest result wins.
async function liveLoop() {
  const inflight = new Set();
  let seq = 0;
  S.liveShown = -1;
  S.lastShownT = performance.now();
  while (S.live && S.cam) {
    const depth = Math.max(1, Math.min(2, (S.status && S.status.pipeline) || 1));
    if (inflight.size >= depth) {
      await Promise.race(inflight);
      continue;
    }
    const blob = await grabFrame();
    if (!S.live) break;
    const p = detect({ live: true, blob, seq: seq++ }).finally(() => inflight.delete(p));
    inflight.add(p);
  }
  await Promise.allSettled(inflight);
}
function startLive() {
  if (!S.status || S.status.state !== "ready") return toast("Load a model first (Settings → Models).", true);
  if (!promptEl().value.trim()) return toast("Type what to find first.");
  S.live = true;
  S.fps = [];
  $("#live-btn").classList.add("on");
  $("#live-btn span").textContent = "Stop Live";
  $("#mi-fps").hidden = false;
  $("#fps").textContent = "…";
  liveLoop();
}
function stopLive(why) {
  if (!S.live) return;
  S.live = false;
  $("#live-btn").classList.remove("on");
  $("#live-btn span").textContent = "Go Live";
  $("#mi-fps").hidden = true;
  if (why) toast(why, true);
}

// --------------------------------------------------------------------- voice
const V = { rec: null, t0: 0, held: 0, timer: null, key: "", wav: null };
const MAX_REC_S = 15;
const NO_SPEECH_S = 5;          // a click that never hears speech gives up after this long

function voiceLine(html, cls = "") {
  const el = $("#voice-line");
  el.hidden = !html;
  el.className = "voice-line" + (cls ? " " + cls : "");
  el.innerHTML = html || "";
}
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function micDown(e) {
  if (e) e.preventDefault();
  if (V.rec) return stopRec();                 // click-to-talk: the second click stops
  if (V.busy) return;
  const vs = S.status && S.status.voice;
  if (!vs || vs.state !== "ready") {
    if (vs && vs.state === "loading") return toast("Voice is still loading…");
    if (!vs || !vs.can_start) return openSettings("voice");
    const what = (vs.choices || []).length ? "Whisper and a small LLM" : "Whisper";
    if (await confirmBox(`Turn on voice? This loads ${what} on the MLA (a few seconds), next to the detector.`, "Turn On"))
      startVoice();
    return;
  }
  V.holding = true;
  await startRec();
  V.held = performance.now();
}
function micUp() {
  // held for a while = push-to-talk (release stops); a quick click records until a pause or the next click
  V.holding = false;
  if (V.rec && performance.now() - V.held > 400) stopRec();
}

// ---- microphone: opened ahead of time and kept warm, with a pre-roll so the first words count
// Opening a microphone takes a moment; if that happens on the Speak click, the start of the
// sentence is lost (and was measured as "room noise"). So the mic opens when the pointer comes
// near Speak (once permission was given), stays open for a minute after use, and always keeps
// the last 1.5 s: a recording starts with that pre-roll, whose level is the room's noise floor.
const MIC = { stream: null, ctx: null, src: null, node: null, rate: 48000, ring: [], ringS: 0, rec: null,
  rms: 0, closeT: 0, opening: null };
const PREROLL_S = 1.5, WARM_S = 60;

function chunkRms(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s / a.length);
}

function onPcm(buf) {
  const a = new Float32Array(buf);                 // copy: the browser reuses its buffer
  const r = chunkRms(a);
  MIC.rms = r;
  if (MIC.rec) {
    MIC.rec.chunks.push(a);
  } else {
    MIC.ring.push({ a, r });
    MIC.ringS += a.length / MIC.rate;
    while (MIC.ring.length > 1 && MIC.ringS - MIC.ring[0].a.length / MIC.rate >= PREROLL_S) MIC.ringS -= MIC.ring.shift().a.length / MIC.rate;
  }
}

async function micWarm() {
  clearTimeout(MIC.closeT);
  MIC.closeT = setTimeout(micClose, WARM_S * 1000);
  if (MIC.stream) return;
  if (MIC.opening) return MIC.opening;
  MIC.opening = (async () => {
    const stream = await getMicStream();
    const ctx = new AudioContext();
    await ctx.resume().catch(() => {});
    const src = ctx.createMediaStreamSource(stream);
    const node = ctx.createScriptProcessor(2048, 1, 1);
    node.onaudioprocess = (e) => onPcm(e.inputBuffer.getChannelData(0));
    src.connect(node);
    node.connect(ctx.destination);                 // needed for it to run; its output is silence
    Object.assign(MIC, { stream, ctx, src, node, rate: ctx.sampleRate, ring: [], ringS: 0 });
  })().finally(() => { MIC.opening = null; });
  return MIC.opening;
}

function micClose() {
  if (MIC.rec) {                                   // still recording: check again later
    MIC.closeT = setTimeout(micClose, WARM_S * 1000);
    return;
  }
  if (MIC.node) MIC.node.disconnect();
  if (MIC.stream) MIC.stream.getTracks().forEach((t) => t.stop());
  if (MIC.ctx) MIC.ctx.close().catch(() => {});
  Object.assign(MIC, { stream: null, ctx: null, src: null, node: null, ring: [], ringS: 0 });
}

// Warm up on hover/focus only when the mic is already allowed (never trigger a permission prompt on hover).
async function micWarmIfAllowed() {
  const vs = S.status && S.status.voice;
  if (!vs || vs.state !== "ready" || MIC.stream || !navigator.permissions) return;
  try {
    const p = await navigator.permissions.query({ name: "microphone" });
    if (p.state === "granted") micWarm().catch(() => {});
  } catch (e) { /* permissions API without "microphone" */ }
}

function toWav(chunks, rate, out = 16000) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const pcm = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { pcm.set(c, o); o += c.length; }
  const m = Math.floor(n * out / rate), step = rate / out;
  const buf = new ArrayBuffer(44 + m * 2), v = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + m * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, out, true);
  v.setUint32(28, out * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, m * 2, true);
  for (let i = 0; i < m; i++) {                    // linear resampling to 16 kHz
    const x = i * step, j = Math.floor(x), f = x - j;
    const s = (pcm[j] || 0) * (1 - f) + (pcm[j + 1] || 0) * f;
    v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function startRec() {
  if (!navigator.mediaDevices || !window.AudioContext)
    return toast("This browser can't record here. Open the studio over HTTPS.", true);
  try {
    await micWarm();
  } catch (e) {
    return toast("Microphone not available: " + e.message, true);
  }
  // the pre-roll goes first; its level is the room's noise floor
  const pre = MIC.ring.splice(0);
  MIC.ringS = 0;
  const preR = pre.map((c) => c.r).sort((x, y) => x - y);
  MIC.rec = { chunks: pre.map((c) => c.a) };
  V.rec = true;
  V.t0 = performance.now();
  $("#mic-btn").classList.add("rec");
  $("#mic-btn .mic-t").textContent = "Stop";
  voiceLine(`<span class="lvl"><span></span></span><span><span class="what">Listening…</span> <b class="secs">0.0</b> s</span>`);
  // End of speech: call it speech when the level is well above the floor, and stop after a
  // pause once something was said (not while the button or the space bar is held).
  const autostop = $("#voice-autostop").checked, pause = +$("#voice-pause").value;
  const vad = { floor: preR.length >= 4 ? preR[Math.floor(preR.length * 0.2)] : null, voiced: 0, spoke: false, quietSince: null };
  V.noSpeech = false;
  V.timer = setInterval(() => {
    const now = performance.now(), t = (now - V.t0) / 1000;
    const sec = $("#voice-line .secs"), bar = $("#voice-line .lvl span"), what = $("#voice-line .what");
    if (sec) sec.textContent = t.toFixed(1);
    const r = MIC.rms;
    if (bar) bar.style.width = `${Math.round(Math.min(1, r * 8) * 100)}%`;
    if (t >= MAX_REC_S) return stopRec();
    if (!autostop) return;
    if (vad.floor == null) {                       // mic just opened, no pre-roll yet
      if (t < 0.25) return;
      vad.floor = r;
    }
    const on = Math.min(0.05, Math.max(0.012, vad.floor * 3));
    if (r > on) {
      vad.voiced += 80;
      vad.quietSince = null;
      if (vad.voiced >= 160 && !vad.spoke) {
        vad.spoke = true;
        if (what) what.textContent = "Hearing you…";
      }
    } else if (vad.quietSince == null) {
      vad.quietSince = now;
    }
    if (V.holding) return;
    if (vad.spoke && vad.quietSince != null && (now - vad.quietSince) / 1000 >= pause) stopRec();
    else if (!vad.spoke && t >= NO_SPEECH_S) {
      V.noSpeech = true;
      stopRec();
    }
  }, 80);
}

function stopRec() {
  if (!V.rec) return;
  clearInterval(V.timer);
  V.dur = (performance.now() - V.t0) / 1000;
  const rec = MIC.rec;
  MIC.rec = null;
  V.rec = null;
  micWarm().catch(() => {});                       // keep the mic warm for the next request
  $("#mic-btn").classList.remove("rec");
  $("#mic-btn .mic-t").textContent = "Speak";
  V.wav = rec ? toWav(rec.chunks, MIC.rate) : null;
  sendRec();
}

async function sendRec() {
  if (V.noSpeech) return voiceLine("Didn’t hear anything. Click Speak and say what to find.", "err");
  if (V.dur < 0.4) return voiceLine("Too short. Hold Speak while you talk, or click once to start and again to stop.", "err");
  if (!V.wav) return;
  const fd = new FormData();
  fd.append("audio", V.wav, "speech.wav");
  V.busy = true;
  $("#mic-btn").classList.add("wait");
  voiceLine(`<span class="spinner"></span><span>Transcribing…</span>`);
  try {
    const r = await api("/api/voice/prompt", { method: "POST", body: fd });
    const how = r.by === "llm" ? `${esc(S.status.voice.llm_title || "LLM")}` : "rules";
    const ms = `${Math.round(r.timing.asr + r.timing.llm)} ms`;
    if (!r.prompt) {
      voiceLine(`Heard <span class="said">“${esc(r.transcript)}”</span> but nothing to find.`, "err");
      return;
    }
    voiceLine(`<span class="said">“${esc(r.transcript)}”</span> → <span class="to">${esc(r.prompt)}</span>
      <span class="muted small" title="Whisper ${Math.round(r.timing.asr)} ms, ${how} ${Math.round(r.timing.llm)} ms">${ms} · ${how}</span>`);
    promptEl().value = r.prompt;
    store.set("prompt", r.prompt);
    parsePrompt();
    if ($("#voice-autorun").checked && !S.live && (S.imageBlob || S.cam)) detect();
  } catch (e) {
    voiceLine(esc(e.message), "err");
  } finally {
    V.busy = false;
    $("#mic-btn").classList.remove("wait");
  }
}

async function startVoice() {
  const sel = $("#voice-llm");
  try {
    await post("/api/voice/start", { llm: sel ? sel.value : null });
    toast("Loading voice models…");
  } catch (e) {
    toast(e.message, true);
  }
  refreshStatus();
}

// Whisper and the LLMs, downloaded from Hugging Face (simaai/...) into the shared LLiMa folder.
function renderVoiceModels(vs) {
  const list = $("#voice-models");
  const models = vs.models || [];
  $("#voice-models-line").innerHTML = `Precompiled LLiMa models from <a href="https://huggingface.co/simaai" target="_blank" rel="noopener">simaai</a> on Hugging Face,
    kept in <code>${esc(vs.models_dir || "")}</code> next to the other LLiMa models (GenAI Studio uses the same ones).`;
  const busy = models.some((m) => m.job);
  list.innerHTML = "";
  for (const m of models) {
    const el = document.createElement("div");
    el.className = "model";
    const tag = m.present ? '<span class="tag on">On Board</span>' : m.job ? '<span class="tag live">Downloading</span>' : "";
    el.innerHTML = `<div class="name"><span></span><span class="tag">${m.role === "asr" ? "Speech" : "LLM"}</span>${tag}</div>
      <div class="meta"></div><div class="actions"></div>`;
    el.querySelector(".name span").textContent = m.title;
    el.querySelector(".meta").textContent = `${m.size} · ${m.note}`;
    const act = el.querySelector(".actions");
    const btn = (label, icon, action, cls = "", disabled = false) => {
      const b = document.createElement("button");
      b.className = "btn " + cls;
      b.disabled = disabled;
      b.innerHTML = `<svg><use href="#i-${icon}"/></svg>${label}`;
      b.onclick = async () => {
        try { await post(`/api/voice/${action}`, { name: m.name }); } catch (e) { toast(e.message, true); }
        refreshStatus();
      };
      act.append(b);
    };
    if (m.job) btn("Cancel", "close", "cancel");
    else if (!m.present) {
      btn(m.partial ? "Resume" : "Download", "download", "download", "primary", busy);
      if (m.partial) btn("Discard", "trash", "discard");
    }
    if (m.job) {
      const p = m.job, pr = document.createElement("div");
      pr.className = "prog";
      pr.innerHTML = `<div class="bar"><span style="width:${(p.fraction * 100).toFixed(1)}%"></span></div><div class="txt"></div>`;
      pr.querySelector(".txt").textContent = p.stage === "listing" ? "Asking Hugging Face for the file list…"
        : `${fmtBytes(p.done_bytes)} of ${fmtBytes(p.total_bytes)} · ${fmtBytes(p.bytes_per_second)}/s` +
          (p.seconds_left != null ? ` · ${Math.ceil(p.seconds_left / 60)} min left` : "");
      el.append(pr);
    }
    if (m.error) {
      const e = document.createElement("div");
      e.className = "err";
      e.textContent = m.error;
      el.append(e);
    }
    list.append(el);
  }
}

function renderVoice() {
  const vs = S.status && S.status.voice;
  const mic = $("#mic-btn");
  if (!vs) return;
  mic.classList.toggle("off", vs.state !== "ready");
  mic.title = vs.state === "ready" ? "Speak what to find: hold to talk, or click to start and stop"
    : vs.state === "loading" ? "Voice is loading…" : "Voice is off: click to turn it on";
  renderVoiceModels(vs);
  const key = JSON.stringify([vs.state, vs.llm, vs.choices, vs.error, vs.can_start, vs.python]);
  const card = $("#voice-card");
  const tm = card.querySelector(".v-time");
  if (tm && vs.state === "loading" && vs.elapsed != null) tm.textContent = `${Math.round(vs.elapsed)} s`;
  if (key === V.key) return;
  V.key = key;
  const tags = { ready: '<span class="tag on">On</span>', loading: '<span class="tag live">Loading</span>',
    stopping: '<span class="tag live">Stopping</span>', error: '<span class="tag">Failed</span>', off: '<span class="tag">Off</span>' };
  const opts = (vs.choices || []).map((n) => `<option value="${esc(n)}"${n === vs.llm ? " selected" : ""}>${esc(vs.titles[n] || n)}</option>`).join("");
  const busy = vs.state === "loading" || vs.state === "stopping";
  card.innerHTML = `<div class="model ${vs.state === "ready" ? "loaded" : ""}">
    <div class="name">Whisper Small${vs.llm || vs.state === "off" ? " + LLM" : ""} ${tags[vs.state] || ""}</div>
    <div class="meta"></div>
    <div class="actions">
      ${opts ? `<select id="voice-llm" title="LLM that turns requests into phrases"${vs.state !== "off" && vs.state !== "error" ? " disabled" : ""}>${opts}</select>` : ""}
      ${vs.state === "ready" || busy ? `<button class="btn" id="voice-off"${busy ? " disabled" : ""}><svg><use href="#i-stop"/></svg>Turn Off</button>`
        : `<button class="btn primary" id="voice-on"${vs.can_start ? "" : " disabled"}><svg><use href="#i-mic"/></svg>Turn On</button>`}
    </div>
    ${vs.state === "loading" ? '<div class="prog"><div class="pbar indeterminate"><span></span></div><div class="txt">Loading Whisper and the LLM on the MLA… <span class="v-time"></span></div></div>' : ""}
    ${vs.error ? `<div class="err">${esc(vs.error)}</div>` : ""}
  </div>`;
  card.querySelector(".meta").textContent = !vs.python
    ? "Needs pyneat (~/pyneat) to run Whisper and the LLM on the MLA."
    : !vs.can_start ? "Download Whisper Small below to turn voice on."
    : vs.state !== "ready" && !(vs.choices || []).length ? "Speech to text: Whisper Small (a16w8). No LLM downloaded, so simple rules turn requests into phrases."
    : vs.state === "ready" ? `Whisper Small (a16w8)${vs.llm ? " · " + (vs.llm_title || vs.llm) : " · no LLM (rules)"} · loaded in ${vs.load_seconds} s`
      : "Speech to text: Whisper Small (a16w8). Request to phrases: the LLM chosen here.";
  const on = $("#voice-on"), off = $("#voice-off");
  if (on) on.onclick = startVoice;
  if (off) off.onclick = async () => {
    try { await post("/api/voice/stop"); } catch (e) { toast(e.message, true); }
    refreshStatus();
  };
}

// --------------------------------------------------------------------- settings / models
function openSettings(pane = "models") {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.pane === pane));
  $$(".pane").forEach((p) => (p.hidden = p.dataset.pane !== pane));
  const d = $("#settings");
  if (!d.open) d.showModal();
  if (pane === "models") refreshModels();
  if (pane === "devices") renderDevices();
  else stopDevicePreview();
}

// --------------------------------------------------------------------- devices pane
const DEV = { cam: null, mic: null, ctx: null, raf: 0, gen: 0 };

function stopDevicePreview() {
  DEV.gen++;
  if (DEV.cam) DEV.cam.getTracks().forEach((t) => t.stop());
  if (DEV.mic) DEV.mic.getTracks().forEach((t) => t.stop());
  if (DEV.ctx) DEV.ctx.close().catch(() => {});
  cancelAnimationFrame(DEV.raf);
  Object.assign(DEV, { cam: null, mic: null, ctx: null });
  const v = $("#dev-video");
  v.srcObject = null;
  v.hidden = true;
  $("#dev-video-note").textContent = "No preview yet";
  $("#dev-level").style.width = "0";
}

function fillDevices(sel, devs, cur, auto, kind) {
  sel.innerHTML = "";
  sel.append(new Option(auto, ""));
  devs.forEach((d, i) => sel.append(new Option(d.label || `${kind} ${i + 1}`, d.deviceId)));
  if (cur && !devs.some((d) => d.deviceId === cur)) sel.append(new Option(`Saved ${kind} (Not Connected)`, cur));
  sel.value = cur;
}

async function renderDevices() {
  stopDevicePreview();
  const gen = DEV.gen, note = $("#dev-note");
  note.hidden = true;
  if (!camSupported()) {
    note.hidden = false;
    note.textContent = "Browsers only give secure (HTTPS) pages access to cameras and microphones. Open this studio with https:// to choose them.";
    ["#cam-select", "#mic-select"].forEach((s) => { $(s).innerHTML = ""; $(s).disabled = true; });
    $("#dev-allow").hidden = true;
    return;
  }
  const all = await navigator.mediaDevices.enumerateDevices();
  if (gen !== DEV.gen) return;
  // Chrome lists "default" and "communications" aliases next to the real microphones; "" ids mean no permission yet.
  const cams = all.filter((d) => d.kind === "videoinput" && d.deviceId);
  const mics = all.filter((d) => d.kind === "audioinput" && d.deviceId && !["default", "communications"].includes(d.deviceId));
  const named = all.some((d) => d.label);
  $("#dev-allow").hidden = named;
  ["#cam-select", "#mic-select"].forEach((s) => ($(s).disabled = !named));
  fillDevices($("#cam-select"), cams, store.get("cam-device", ""), "Automatic (Back Camera On Phones)", "Camera");
  fillDevices($("#mic-select"), mics, store.get("mic-device", ""), "System Default", "Microphone");
  if (!named) {
    $("#dev-video-note").textContent = "Allow access to see your cameras";
    $("#dev-mic-note").textContent = "Allow access to see your microphones.";
    return;
  }
  startCamPreview(gen);
  startMicMeter(gen);
}

async function startCamPreview(gen) {
  const v = $("#dev-video"), note = $("#dev-video-note");
  let stream = S.cam;                          // the webcam is already open: show it rather than open it twice
  if (!stream) {
    note.textContent = "Starting camera…";
    try {
      stream = await getCamStream();
    } catch (e) {
      if (gen === DEV.gen) note.textContent = e.name === "NotFoundError" ? "No camera found" : `Camera unavailable: ${e.message}`;
      return;
    }
    if (gen !== DEV.gen) return stream.getTracks().forEach((t) => t.stop());
    DEV.cam = stream;
  }
  v.srcObject = stream;
  v.classList.toggle("mirror", shouldMirror(stream));
  v.hidden = false;
  v.play().catch(() => {});
}

async function startMicMeter(gen) {
  const note = $("#dev-mic-note");
  let stream;
  try {
    stream = await getMicStream();
  } catch (e) {
    if (gen === DEV.gen) note.textContent = e.name === "NotFoundError" ? "No microphone found." : `Microphone unavailable: ${e.message}`;
    return;
  }
  if (gen !== DEV.gen) return stream.getTracks().forEach((t) => t.stop());
  DEV.mic = stream;
  note.textContent = "Say something to check the level.";
  try {
    DEV.ctx = new AudioContext();
    const an = DEV.ctx.createAnalyser();
    an.fftSize = 512;
    DEV.ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize), bar = $("#dev-level");
    const tick = () => {
      if (gen !== DEV.gen) return;
      an.getByteTimeDomainData(buf);
      let m = 0;
      for (const x of buf) m = Math.max(m, Math.abs(x - 128));
      bar.style.width = `${Math.min(100, (m / 64) * 100)}%`;
      DEV.raf = requestAnimationFrame(tick);
    };
    tick();
  } catch (e) {
    note.textContent = "Level meter unavailable in this browser.";
  }
}

async function allowDevices() {
  for (const kind of ["video", "audio"]) {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ [kind]: true });
      s.getTracks().forEach((t) => t.stop());
    } catch (e) {
      if (e.name === "NotAllowedError") toast(`${kind === "video" ? "Camera" : "Microphone"} permission was denied. Allow it in the browser's site settings.`, true);
    }
  }
  renderDevices();
}

async function refreshModels() {
  clearTimeout(refreshModels.t);
  if (!$("#settings").open) return;
  let cat;
  try {
    cat = await api("/api/models");
  } catch (e) {
    $("#model-list").textContent = e.message;
    return;
  }
  renderModels(cat);
  await refreshStatus();
  const busy = cat.models.some((m) => m.state === "downloading") ||
    (S.status && (S.status.state === "loading" || S.status.state === "unloading"));
  refreshModels.t = setTimeout(refreshModels, busy ? 900 : 5000);
}

function renderModels(cat) {
  const used = cat.used_bytes || 0, free = cat.free_bytes || 0;
  $("#storage").innerHTML = `<div class="bar"><span style="width:${Math.min(100, (100 * used) / Math.max(1, used + free)).toFixed(1)}%"></span></div>
    <div class="txt"><span>Models use ${fmtBytes(used)} in ${cat.directory}</span><span>${fmtBytes(free)} free</span></div>`;
  const list = $("#model-list");
  list.innerHTML = "";
  if (!cat.models.length) list.innerHTML = '<div class="muted">No models yet. The list from Hugging Face appears here once the board is online.</div>';
  for (const m of cat.models) list.append(modelRow(m));
  const hub = cat.hub;
  $("#hub-line").innerHTML = hub.repo ? `Models come from <a href="${hub.page}" target="_blank" rel="noopener">${hub.repo}</a> on Hugging Face.` +
    (hub.error ? ` <span style="color:var(--err)">${hub.error}</span>` : "") : "";
}

function modelRow(m) {
  const el = document.createElement("div");
  el.className = "model" + (m.loaded ? " loaded" : "");
  const st = S.status || {};
  const busyHere = st.model === m.name && (st.state === "loading" || st.state === "unloading");
  const tags = [];
  if (busyHere) tags.push(`<span class="tag live">${st.state === "loading" ? "Loading" : "Unloading"}</span>`);
  else if (m.loaded) tags.push('<span class="tag live">Loaded</span>');
  if (m.state === "on_board") tags.push('<span class="tag on">On Board</span>');
  if (m.verified === true) tags.push('<span class="tag on">Verified</span>');
  if (m.verified === false) tags.push('<span class="tag" style="color:var(--err)">Damaged</span>');
  el.innerHTML = `<div class="name"></div><div class="meta"></div><div class="actions"></div>`;
  el.querySelector(".name").innerHTML = `<span></span>${tags.join("")}`;
  el.querySelector(".name span").textContent = m.title;
  const meta = [m.details, fmtBytes(m.bytes)];
  if (m.kind === "text") meta.push(m.variant === "tiny" ? "used by the Tiny detector" : "used by every Base detector");
  if (m.latency_ms) meta.push(`~${Math.round(m.latency_ms)} ms per image`);
  if (m.kind === "detector" && m.requires && !m.requires_present) meta.push("needs the text encoder");
  el.querySelector(".meta").textContent = meta.filter(Boolean).join(" · ") + (m.about ? ` — ${m.about}` : "");
  const act = el.querySelector(".actions");
  const btn = (label, icon, fn, cls = "") => {
    const b = document.createElement("button");
    b.className = "btn " + cls;
    b.innerHTML = `<svg><use href="#i-${icon}"/></svg>${label}`;
    b.onclick = fn;
    act.append(b);
    return b;
  };
  if (busyHere) {
    const p = st.progress || {};
    const b = document.createElement("button");
    b.className = "btn";
    b.disabled = true;
    b.innerHTML = `<span class="spinner"></span>${st.state === "loading" ? "Loading…" : "Unloading…"}`;
    act.append(b);
    const pr = document.createElement("div");
    pr.className = "prog";
    pr.innerHTML = `<div class="pbar"><span></span></div><div class="txt"></div>`;
    setBar(pr.querySelector(".pbar"), p.fraction);
    pr.querySelector(".txt").textContent = [p.label, p.detail].filter(Boolean).join(" · ");
    el.append(pr);
  } else if (m.state === "downloading") {
    btn("Cancel", "close", () => modelAction("cancel", m.name));
  } else if (m.state === "available") {
    btn(m.partial_bytes ? "Resume" : "Download", "download", () => modelAction("download", m.name), "primary");
    if (m.partial_bytes) btn("Discard", "trash", () => modelAction("discard", m.name));
  } else if (m.state === "on_board") {
    if (m.kind === "detector") {
      if (m.loaded) btn("Unload", "stop", () => modelAction("unload", m.name));
      else btn("Load", "chip", () => modelAction("load", m.name), "primary");
    }
    btn("Verify", "shield", () => modelAction("verify", m.name));
    if (m.on_hub) btn("Re-download", "refresh", () => modelAction("redownload", m.name));
    btn("Delete", "trash", async () => {
      if (await confirmBox(`Delete ${m.title} from the board? You can download it again later.`, "Delete")) modelAction("delete", m.name);
    });
  }
  if (m.progress) {
    const p = m.progress;
    const pr = document.createElement("div");
    pr.className = "prog";
    const what = p.stage === "download" ? `${fmtBytes(p.done_bytes)} of ${fmtBytes(p.total_bytes)} · ${fmtBytes(p.bytes_per_second)}/s` +
      (p.seconds_left != null ? ` · ${p.seconds_left} s left` : "") : p.stage === "unpack" ? "Unpacking…" : "Checking files…";
    pr.innerHTML = `<div class="bar"><span style="width:${(p.fraction * 100).toFixed(1)}%"></span></div><div class="txt"></div>`;
    pr.querySelector(".txt").textContent = what;
    el.append(pr);
  }
  if (m.error) {
    const e = document.createElement("div");
    e.className = "err";
    e.textContent = m.error;
    el.append(e);
  }
  return el;
}

async function modelAction(action, name) {
  try {
    const r = await post(`/api/models/${action}`, { name });
    if (action === "verify") toast(r.ok ? "Every file matches its checksum." : "Some files are damaged; re-download the model.", !r.ok);
    if (action === "load" || action === "unload") refreshStatus();
    refreshStatus();
  } catch (e) {
    toast(e.message, true);
  }
  refreshModels();
}

// --------------------------------------------------------------------- wiring
function init() {
  // theme
  $("#theme-btn").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    const root = document.documentElement;
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      root.classList.add("theme-anim");
      clearTimeout(init.themeT);
      init.themeT = setTimeout(() => root.classList.remove("theme-anim"), 420);
    }
    root.setAttribute("data-theme", next);
    store.set("theme", next);
    try { localStorage.setItem("dino-theme", next); } catch (e) { /* ignore */ }
  };
  $("#full-btn").onclick = () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
  $("#settings-btn").onclick = () => openSettings("models");
  $("#model-pill").onclick = () => openSettings("models");
  $("#settings-close").onclick = () => $("#settings").close();
  $("#settings").addEventListener("close", stopDevicePreview);
  $("#dev-allow").onclick = allowDevices;
  $("#cam-select").onchange = async (e) => {
    store.set("cam-device", e.target.value);
    stopDevicePreview();
    if (S.cam) {                               // switch the open webcam now, and keep Live going
      const wasLive = S.live;
      await openCam();
      if (wasLive && S.cam) startLive();
    }
    renderDevices();
  };
  $("#mic-select").onchange = (e) => {
    store.set("mic-device", e.target.value);
    micClose();                                     // the warm mic reopens on the new device
    renderDevices();
  };
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener)
    navigator.mediaDevices.addEventListener("devicechange", () => {
      if ($("#settings").open && !$('.pane[data-pane="devices"]').hidden) renderDevices();
    });
  $$(".nav-item").forEach((b) => (b.onclick = () => openSettings(b.dataset.pane)));
  $("#settings").addEventListener("click", (e) => { if (e.target === $("#settings")) $("#settings").close(); });
  $("#reset-mla").onclick = async () => {
    if (!(await confirmBox("Reset the MLA? This unloads the model for everyone using the studio.", "Reset"))) return;
    try {
      const r = await post("/api/reset-mla");
      toast(r.message);
    } catch (e) {
      toast(e.message, true);
    }
    refreshStatus();
    refreshModels();
  };

  // inputs
  const pick = () => $("#file").click();
  $("#upload-btn").onclick = pick;
  $("#pick-btn").onclick = pick;
  $("#file").onchange = (e) => { useFile(e.target.files[0]); e.target.value = ""; };
  $("#cam-btn").onclick = () => (S.cam ? stopCam() : openCam());
  $("#cam-btn-2").onclick = openCam;
  $("#snap-btn").onclick = snapshot;
  $("#live-btn").onclick = () => (S.live ? stopLive() : startLive());
  $("#flip-btn").onclick = () => {
    if (S.devices.length < 2) return;
    const cur = S.cam.getVideoTracks()[0].getSettings().deviceId;
    const i = S.devices.findIndex((d) => d.deviceId === cur);
    store.set("cam-device", S.devices[(i + 1) % S.devices.length].deviceId);
    openCam();
  };
  const stage = $("#stage");
  stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("dragging"); });
  stage.addEventListener("dragleave", () => stage.classList.remove("dragging"));
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    stage.classList.remove("dragging");
    if (e.dataTransfer.files[0]) useFile(e.dataTransfer.files[0]);
  });
  window.addEventListener("paste", (e) => {
    const f = Array.from(e.clipboardData.files || [])[0];
    if (f) useFile(f);
  });

  // prompt + detect
  promptEl().value = store.get("prompt", "");
  promptEl().addEventListener("input", onPromptInput);
  promptEl().addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
      e.preventDefault();
      detect();
    }
  });
  $("#detect-btn").onclick = () => detect();
  const mic = $("#mic-btn");
  mic.addEventListener("pointerenter", micWarmIfAllowed);
  mic.addEventListener("focus", micWarmIfAllowed);
  mic.addEventListener("pointerdown", micDown);
  mic.addEventListener("pointerup", micUp);
  mic.addEventListener("pointerleave", micUp);
  mic.addEventListener("contextmenu", (e) => e.preventDefault());
  const typing = (el) => el && (el.tagName === "TEXTAREA" || el.tagName === "INPUT" || el.tagName === "SELECT");
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !typing(document.activeElement) && !$("#settings").open) micDown(e);
  });
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space" && V.rec && !typing(document.activeElement)) { e.preventDefault(); stopRec(); }
  });
  const as = $("#voice-autostop");
  as.checked = store.get("voice-autostop", true);
  as.addEventListener("change", () => { store.set("voice-autostop", as.checked); $("#voice-pause").disabled = !as.checked; });
  $("#voice-pause").disabled = !as.checked;
  const vp = $("#voice-pause");
  vp.value = store.get("voice-pause", 0.8);
  const vpu = () => { $("#voice-pause-v").textContent = `${(+vp.value).toFixed(1)} s`; store.set("voice-pause", vp.value); };
  vp.addEventListener("input", vpu);
  vpu();
  const ar = $("#voice-autorun");
  ar.checked = store.get("voice-autorun", true);
  ar.addEventListener("change", () => store.set("voice-autorun", ar.checked));
  $("#assist-btn").onclick = async () => {
    try {
      const r = await post("/api/prompt/assist", { text: promptEl().value });
      promptEl().value = r.text;
      parsePrompt();
    } catch (e) {
      toast(e.message, true);
    }
  };

  // thresholds + display (client side: no new run)
  const bind = (id, def, fmt = (v) => v) => {
    const el = $(id), out = $(id + "-v");
    el.value = store.get(id, def);
    const upd = () => { if (out) out.textContent = fmt(el.value); store.set(id, el.value); draw(); };
    el.addEventListener("input", upd);
    upd();
  };
  bind("#box-thr", 0.35, (v) => (+v).toFixed(2));
  bind("#text-thr", 0.25, (v) => (+v).toFixed(2));
  bind("#line-w", 2.5);
  bind("#label-size", 13);
  for (const id of ["#show-scores", "#fill-boxes", "#mirror-cam", "#nms"]) {
    const el = $(id);
    el.checked = store.get(id, el.checked);
    el.addEventListener("change", () => {
      store.set(id, el.checked);
      if (id === "#nms" && S.result && !S.cam) detect();
      else if (id === "#mirror-cam") {
        if (S.cam) $("#video").classList.toggle("mirror", shouldMirror(S.cam));
        const dv = $("#dev-video");
        if (dv.srcObject) dv.classList.toggle("mirror", shouldMirror(dv.srcObject));
        draw();
      } else draw();
    });
  }
  $("#reset-thr").onclick = () => {
    $("#box-thr").value = 0.35;
    $("#text-thr").value = 0.25;
    $("#box-thr").dispatchEvent(new Event("input"));
    $("#text-thr").dispatchEvent(new Event("input"));
  };
  $("#overlay").addEventListener("mousemove", (e) => {
    const r = S.result;
    if (!r) return;
    const cv = $("#overlay"), rect = cv.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * r.width, y = ((e.clientY - rect.top) / rect.height) * r.height;
    const dets = visibleDetections();
    let best = -1, area = Infinity;
    dets.forEach((d, i) => {
      const [x0, y0, x1, y1] = d.box, a = (x1 - x0) * (y1 - y0);
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && a < area) { best = i; area = a; }
    });
    if (best !== S.hover) { S.hover = best; draw(); }
  });
  window.addEventListener("resize", fitOverlay);
  new ResizeObserver(fitOverlay).observe($("#stage"));

  // rail cards fold away from their header; which ones are folded is remembered per browser
  const folded = new Set(store.get("collapsed", []));
  $$(".rail .card[data-card]").forEach((card) => {
    const btn = card.querySelector(".card-toggle"), key = card.dataset.card;
    const apply = (shut) => { card.classList.toggle("collapsed", shut); btn.setAttribute("aria-expanded", String(!shut)); };
    apply(folded.has(key));
    btn.onclick = () => {
      const shut = !card.classList.contains("collapsed");
      apply(shut);
      shut ? folded.add(key) : folded.delete(key);
      store.set("collapsed", [...folded]);
    };
  });

  renderExamples();
  renderSamples();
  $("#hero-model").onclick = () => openSettings("models");
  refreshStatus().then(parsePrompt);
  if (location.pathname === "/models") openSettings("models");
}

document.addEventListener("DOMContentLoaded", init);
