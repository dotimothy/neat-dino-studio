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
  }
  renderStatus();
  clearTimeout(refreshStatus.t);
  const loading = S.status && S.status.state === "loading";
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
  if (st.state === "ready") {
    dot.classList.add("ready");
    name.textContent = matchMedia("(max-width: 600px)").matches ? prettyModel(st.model).split(" · ")[0] : prettyModel(st.model);
    if (st.last_ms) ms.textContent = `${Math.round(st.last_ms)} ms`;
    banner.hidden = true;
  } else if (st.state === "loading") {
    dot.classList.add("loading");
    name.textContent = `Loading ${prettyModel(st.model)}…`;
    showBanner(`Loading ${prettyModel(st.model)} (${st.stage || "starting"}, ${Math.round(st.loading_seconds || 0)} s)…`);
  } else if (st.state === "error") {
    dot.classList.add("error");
    name.textContent = "Load Failed";
    showBanner(st.error || "Loading the model failed.", true, "Open Models");
  } else {
    name.textContent = "No Model";
    showBanner("No model is loaded. Load one in Settings → Models to start detecting.", false, "Open Models");
  }
  $("#about-status").textContent = `Server ${st.version}, backend ${st.backend}` + (st.size ? `, input ${st.size} px` : "") +
    (st.load_seconds ? `, loaded in ${st.load_seconds} s` : "");
}

function showBanner(text, err = false, action = null) {
  const b = $("#banner");
  b.hidden = false;
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
  const m = /dino-(\d+)-(\w+)/.exec(n);
  if (!m) return n;
  return `${m[1] === "768" ? "Accurate" : m[1] === "576" ? "Fast" : m[1]} · ${m[1]} px · ${m[2]}`;
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

function renderTiming(t) {
  const bar = $("#timing");
  bar.hidden = false;
  bar.innerHTML = "";
  const total = t.server_total || t.total || 1;
  for (const [k, label, color] of TIMING_PARTS) {
    if (!t[k]) continue;
    const s = document.createElement("span");
    s.style.flex = `${t[k]} 0 0`;
    s.style.background = color;
    s.title = `${label}: ${t[k].toFixed(1)} ms`;
    if (t[k] / total > 0.09) s.textContent = `${label} ${Math.round(t[k])}`;
    bar.append(s);
  }
  const shown = TIMING_PARTS.reduce((a, [k]) => a + (t[k] || 0), 0);
  const other = total - shown;
  if (other > total * 0.03) {
    const s = document.createElement("span");
    s.style.flex = `${other} 0 0`;
    s.style.background = "#5b6573";
    s.title = `Other (model, transfer, post-processing): ${other.toFixed(1)} ms`;
    if (other / total > 0.09) s.textContent = `Other ${Math.round(other)}`;
    bar.append(s);
  }
  const tot = document.createElement("span");
  tot.className = "total";
  tot.textContent = `${Math.round(total)} ms`;
  tot.title = "Server time for this request";
  bar.append(tot);
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
    const want = { width: { ideal: 1280 }, height: { ideal: 720 } };
    if (S.deviceId) want.deviceId = { exact: S.deviceId };
    else want.facingMode = S.facing;
    const stream = await navigator.mediaDevices.getUserMedia({ video: want, audio: false });
    const v = $("#video");
    v.srcObject = stream;
    await v.play();
    S.cam = stream;
    S.imageBlob = null;
    S.imageId = null;
    S.result = null;
    const track = stream.getVideoTracks()[0];
    const facing = (track.getSettings && track.getSettings().facingMode) || "";
    v.classList.toggle("mirror", $("#mirror-cam").checked && facing !== "environment");
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

function stopCam() {
  stopLive();
  if (S.cam) S.cam.getTracks().forEach((t) => t.stop());
  S.cam = null;
  ["#snap-btn", "#live-btn", "#flip-btn"].forEach((s) => ($(s).hidden = true));
  $("#cam-btn").innerHTML = '<svg><use href="#i-webcam"/></svg>Webcam';
}

async function grabFrame() {
  const v = $("#video");
  const size = (S.status && S.status.size) || 768;
  const s = Math.min(1, size / Math.max(v.videoWidth, v.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(v.videoWidth * s);
  c.height = Math.round(v.videoHeight * s);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return await new Promise((r) => c.toBlob(r, "image/jpeg", 0.88));
}

async function snapshot() {
  if (!S.cam) return;
  const blob = await grabFrame();
  const url = URL.createObjectURL(blob);
  stopCam();
  const img = $("#photo");
  img.onload = () => { S.sourceW = img.naturalWidth; S.sourceH = img.naturalHeight; fitOverlay(); };
  img.src = url;
  showViewer("photo");
  S.imageBlob = blob;
  S.imageId = null;
  detect();
}

async function liveLoop() {
  while (S.live && S.cam) {
    const t0 = performance.now();
    await detect({ live: true, blob: await grabFrame() });
    const dt = performance.now() - t0;
    S.fps.push(dt);
    if (S.fps.length > 10) S.fps.shift();
    const avg = S.fps.reduce((a, b) => a + b, 0) / S.fps.length;
    $("#fps").textContent = `${(1000 / avg).toFixed(1)} FPS · ${Math.round(avg)} ms`;
  }
}
function startLive() {
  if (!S.status || S.status.state !== "ready") return toast("Load a model first (Settings → Models).", true);
  if (!promptEl().value.trim()) return toast("Type what to find first.");
  S.live = true;
  S.fps = [];
  $("#live-btn").classList.add("on");
  $("#live-btn span").textContent = "Stop Live";
  $("#fps").hidden = false;
  liveLoop();
}
function stopLive(why) {
  if (!S.live) return;
  S.live = false;
  $("#live-btn").classList.remove("on");
  $("#live-btn span").textContent = "Go Live";
  $("#fps").hidden = true;
  if (why) toast(why, true);
}

// --------------------------------------------------------------------- settings / models
function openSettings(pane = "models") {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.pane === pane));
  $$(".pane").forEach((p) => (p.hidden = p.dataset.pane !== pane));
  const d = $("#settings");
  if (!d.open) d.showModal();
  if (pane === "models") refreshModels();
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
  const busy = cat.models.some((m) => m.state === "downloading") || (S.status && S.status.state === "loading");
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
  const tags = [];
  if (m.loaded) tags.push('<span class="tag live">Loaded</span>');
  if (m.state === "on_board") tags.push('<span class="tag on">On Board</span>');
  if (m.verified === true) tags.push('<span class="tag on">Verified</span>');
  if (m.verified === false) tags.push('<span class="tag" style="color:var(--err)">Damaged</span>');
  el.innerHTML = `<div class="name"></div><div class="meta"></div><div class="actions"></div>`;
  el.querySelector(".name").innerHTML = `<span></span>${tags.join("")}`;
  el.querySelector(".name span").textContent = m.title;
  const meta = [m.kind === "text" ? "Shared by every detector" : `${m.size} px input`, m.precision, fmtBytes(m.bytes)];
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
  if (m.state === "downloading") {
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
    if (action === "load") toast(`Loading ${prettyModel(name)}…`);
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
    document.documentElement.setAttribute("data-theme", next);
    store.set("theme", next);
    try { localStorage.setItem("dino-theme", next); } catch (e) { /* ignore */ }
  };
  $("#full-btn").onclick = () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
  $("#settings-btn").onclick = () => openSettings("models");
  $("#model-pill").onclick = () => openSettings("models");
  $("#settings-close").onclick = () => $("#settings").close();
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
    S.deviceId = S.devices[(i + 1) % S.devices.length].deviceId;
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
      else draw();
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

  renderExamples();
  refreshStatus().then(parsePrompt);
  if (location.pathname === "/models") openSettings("models");
}

document.addEventListener("DOMContentLoaded", init);
