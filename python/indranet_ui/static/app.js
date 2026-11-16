// IndraNet UI: plain JavaScript and canvas over the JSON API of indranet_ui.server.
"use strict";

const $ = (id) => document.getElementById(id);
const S = {
  info: null, tokens: null, texts: null, run: null, pos: 0,
  tab: "attention", layer: 0, head: -1, attn: null, lens: null, dla: null, patch: null, heads: null,
};

// --- API -------------------------------------------------------------------------------
async function api(name, body, binary = false) {
  const r = await fetch("/api/" + name, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { msg = (await r.json()).error; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  if (binary) {
    const shape = r.headers.get("X-Shape").split(",").map(Number);
    return { data: new Float32Array(await r.arrayBuffer()), shape };
  }
  return r.json();
}

let busyCount = 0;
// Progress and errors go to the header's status (results) and under the prompt (landing page).
function setStatus(text, error = false, home = text) {
  for (const [id, t] of [["status", text], ["home-status", home]]) {
    $(id).className = "status" + (error ? " error" : "");
    $(id).textContent = t;
  }
}

async function busy(label, button, fn) {
  busyCount++;
  if (button) button.disabled = true;
  setStatus(label + "…");
  const t0 = performance.now();
  try {
    const out = await fn();
    setStatus(`${label}: ${((performance.now() - t0) / 1000).toFixed(2)} s`, false, "");
    return out;
  } catch (e) {
    setStatus(e.message, true);
    throw e;
  } finally {
    busyCount--;
    if (button) button.disabled = false;
  }
}

// --- text and colour helpers --------------------------------------------------------------
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// A token as one visible line: spaces as middle dots, newlines and tabs as symbols.
const vis = (s) => String(s).replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/\r/g, "␍");
const chipText = (s) => String(s).replace(/ /g, " ").replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/\r/g, "␍") || "∅";
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : v >= 0.001 ? (100 * v).toFixed(1) + "%" : v.toExponential(1));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

let PAL = {};
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  const get = (n) => {
    const h = cs.getPropertyValue(n).trim().replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  };
  for (const n of ["seq-lo", "seq-hi", "neg", "mid", "pos", "empty", "ink", "muted", "line", "card", "chip", "accent",
                   "hero-a", "hero-b", "net-line", "net-jewel"])
    PAL[n] = get("--" + n);
}
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const rgb = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
const seq = (v) => (v === null || v === undefined || Number.isNaN(v) ? PAL.empty : mix(PAL["seq-lo"], PAL["seq-hi"], clamp(v, 0, 1)));
function div(v, scale) {
  if (v === null || v === undefined || Number.isNaN(v)) return PAL.empty;
  const t = clamp(v / (scale || 1), -1, 1);
  return t < 0 ? mix(PAL.mid, PAL.neg, -t) : mix(PAL.mid, PAL.pos, t);
}
const luminance = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
const textOn = (c) => (luminance(c) > 0.55 ? "#111" : "#fff");

// --- tooltip -------------------------------------------------------------------------------
const tip = $("tooltip");
function showTip(html, e) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = Math.max(4, x) + "px";
  tip.style.top = Math.max(4, y) + "px";
}
const hideTip = () => { tip.hidden = true; };
function topTable(rows) {  // rows: [[text, prob], ...]
  return "<table>" + rows.map(([t, p]) => `<tr><td>${esc(vis(t))}</td><td class="p">${pct(p)}</td></tr>`).join("") + "</table>";
}

// --- heatmap ---------------------------------------------------------------------------------
const MONO = getComputedStyle(document.documentElement).getPropertyValue("--mono");
function fitText(g, s, w) {
  if (g.measureText(s).width <= w) return s;
  while (s.length > 1 && g.measureText(s + "…").width > w) s = s.slice(0, -1);
  return s.length > 1 ? s + "…" : "";
}
// o: rows, cols, cellW, cellH, color(r, c), rowLabels, colLabels, labelW, labelH, text(r, c)?,
//    mark(r, c)? (a dot), hover(r, c) -> html, click(r, c)?, highlightCol?
function heatmap(canvas, o) {
  const dpr = window.devicePixelRatio || 1;
  const labelW = o.labelW ?? 60;
  const rotate = o.cellW < 44;
  let labelH = o.labelH;
  const g0 = canvas.getContext("2d");
  g0.font = `11px ${MONO}`;
  if (labelH === undefined) {
    const longest = Math.max(0, ...(o.colLabels || []).map((s) => g0.measureText(s).width));
    labelH = o.colLabels ? (rotate ? Math.min(90, longest * 0.75 + 12) : 20) : 4;
  }
  const W = labelW + o.cols * o.cellW + (rotate ? 30 : 4), H = labelH + o.rows * o.cellH + 2;
  canvas.width = Math.ceil(W * dpr);
  canvas.height = Math.ceil(H * dpr);
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.font = `11px ${MONO}`;
  g.textBaseline = "middle";
  g.fillStyle = rgb(PAL.muted);
  g.textAlign = "right";
  for (let r = 0; r < o.rows; r++) g.fillText(fitText(g, o.rowLabels[r], labelW - 8), labelW - 6, labelH + r * o.cellH + o.cellH / 2);
  if (o.colLabels) {
    for (let c = 0; c < o.cols; c++) {
      const x = labelW + c * o.cellW + o.cellW / 2;
      g.save();
      g.fillStyle = c === o.highlightCol ? rgb(PAL.ink) : rgb(PAL.muted);
      if (rotate) {
        g.translate(x, labelH - 4);
        g.rotate(-Math.PI / 4);
        g.textAlign = "left";
        g.fillText(fitText(g, o.colLabels[c], 110), 0, 0);
      } else {
        g.textAlign = "center";
        g.fillText(fitText(g, o.colLabels[c], o.cellW - 4), x, labelH / 2);
      }
      g.restore();
    }
  }
  const gap = o.cellW >= 8 && o.cellH >= 8 ? 1 : 0;
  g.textAlign = "center";
  for (let r = 0; r < o.rows; r++) {
    for (let c = 0; c < o.cols; c++) {
      const col = o.color(r, c);
      const x = labelW + c * o.cellW, y = labelH + r * o.cellH;
      g.fillStyle = rgb(col);
      g.fillRect(x, y, o.cellW - gap, o.cellH - gap);
      if (o.text && o.cellW >= 30) {
        const s = o.text(r, c);
        if (s) {
          g.fillStyle = textOn(col);
          g.fillText(fitText(g, s, o.cellW - 5), x + (o.cellW - gap) / 2, y + (o.cellH - gap) / 2 + 0.5);
        }
      }
      if (o.mark && o.mark(r, c)) {
        g.fillStyle = textOn(col);
        g.beginPath();
        g.arc(x + o.cellW - gap - 4, y + 4, 1.8, 0, 2 * Math.PI);
        g.fill();
      }
    }
  }
  if (o.highlightCol !== undefined && o.highlightCol >= 0) {
    g.strokeStyle = rgb(PAL.ink);
    g.lineWidth = 1.5;
    g.strokeRect(labelW + o.highlightCol * o.cellW + 0.5, labelH + 0.5, o.cellW - gap - 1, o.rows * o.cellH - gap - 1);
  }
  const cellAt = (e) => {
    const b = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - b.left - labelW) / o.cellW), r = Math.floor((e.clientY - b.top - labelH) / o.cellH);
    return r >= 0 && r < o.rows && c >= 0 && c < o.cols ? [r, c] : null;
  };
  canvas.onmousemove = (e) => {
    const rc = cellAt(e);
    const html = rc && o.hover(rc[0], rc[1]);
    if (html) showTip(html, e); else hideTip();
    canvas.style.cursor = rc && o.click ? "pointer" : "default";
  };
  canvas.onmouseleave = hideTip;
  canvas.onclick = (e) => {
    const rc = cellAt(e);
    if (rc && o.click) o.click(rc[0], rc[1]);
  };
}

// --- prompt and token strip ------------------------------------------------------------------
async function toTokens(text) {
  if (/^\s*\d+(\s*,\s*\d+)*\s*,?\s*$/.test(text)) {
    const tokens = text.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
    return { tokens, texts: (await api("texts", { tokens })).texts };
  }
  return api("tokenize", { text, special: $("special").checked });
}

async function runPrompt() {
  const text = $("prompt").value;
  if (!text.trim()) return;
  await busy("Running", $("run"), async () => {
    const { tokens, texts } = await toTokens(text);
    const run = await api("run", { tokens });
    Object.assign(S, { tokens, texts, run, pos: tokens.length - 1, attn: null, lens: null, dla: null, patch: null });
    $("dla-pos").value = "";
    $("patch-pos").value = "";
    $("prompt-echo").textContent = text.replace(/\s+/g, " ").trim();
    $("prompt-echo").title = text;
    $("loss").textContent = run.loss === null ? "" : `mean next-token loss ${run.loss.toFixed(3)}`;
    renderStrip();
  });
  showResults(true);
  openTab(S.tab);
}

// Two views: the prompt page, and the results of the last run. The browser's Back returns to the prompt.
function showResults(push) {
  closeMenu();
  $("home").hidden = true;
  document.body.classList.add("results");
  $("top").hidden = false;
  $("results").hidden = false;
  if (push && !(history.state && history.state.view === "results")) history.pushState({ view: "results" }, "", "#results");
  window.scrollTo(0, 0);
}

function showHome() {
  hideTip();
  $("results").hidden = true;
  $("top").hidden = true;
  document.body.classList.remove("results");
  $("home").hidden = false;
  drawNet();
  $("prompt").focus();
}

function editPrompt() {
  if (history.state && history.state.view === "results") history.back();  // popstate shows the prompt
  else showHome();
}

// Example prompts: one click fills the prompt (and the patching fields that go with it) and runs it.
function examples(info) {
  if (info.has_tokenizer) {
    return [
      { name: "Indirect object", prompt: "When Mary and John went to the store, John gave a drink to",
        corrupt: "When Mary and John went to the store, Mary gave a drink to", target: " Mary", other: " John" },
      { name: "Induction", prompt: "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
        corrupt: "The quick brown fox jumps over the lazy cat. The quick brown fox jumps over the lazy", target: " dog", other: " cat" },
      { name: "Fact recall", prompt: "The Eiffel Tower is located in the city of",
        corrupt: "The Colosseum is located in the city of", target: " Paris", other: " Rome" },
    ];
  }
  if (info.has_vocab) return [];
  const seg = [...Array(10).keys()].map((i) => (i * 7919 + 13) % info.vocab_size);
  return [{ name: "Repeated random tokens (induction)", prompt: [...seg, ...seg].join(",") }];
}

function renderExamples(list) {
  $("examples-wrap").hidden = list.length === 0;
  const box = $("examples");
  box.innerHTML = "";
  for (const ex of list) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = ex.name;
    b.title = ex.prompt;
    b.onclick = () => {
      $("prompt").value = ex.prompt;
      growPrompt();
      $("patch-corrupt").value = ex.corrupt || "";
      $("patch-target").value = ex.target || "";
      $("patch-other").value = ex.other || "";
      $("dla-target").value = "";
      $("dla-other").value = "";
      runPrompt();
    };
    box.appendChild(b);
  }
}

// --- the landing page: prompt bar, models menu, picture ------------------------------------------------
function growPrompt() {
  const t = $("prompt");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 200) + "px";
}

const count = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : (n / 1e3).toFixed(1) + "K");

function shortName(info) {
  const p = info.num_parameters;
  return `${info.file} · ${count(p)}${info.fp16 ? " · fp16" : ""}`;
}

// After loading (or switching to) a model: its name everywhere, a fitting placeholder, its examples.
function applyInfo(info) {
  S.info = info;
  $("model-info").innerHTML = describe(info);
  $("model-name").textContent = shortName(info);
  const prompt = $("prompt");
  prompt.placeholder = info.has_tokenizer ? "Type a prompt"
    : info.has_vocab ? "Type a prompt, or token ids like 5,17,3"
      : "Token ids, like 5,17,3 (no vocabulary)";
  renderExamples(examples(info));
}

function closeMenu() {
  $("model-menu").hidden = true;
  $("model-btn").setAttribute("aria-expanded", "false");
}

// The models menu: search, the model families, and "Custom model" on the left; the models of the
// chosen family (or the search's matches, or the custom-model form) in a panel beside it.
const FAMILIES = [[/^llama/, "Llama"], [/^mistral/, "Mistral"], [/^qwen/, "Qwen"], [/^phi/, "Phi"], [/^gemma/, "Gemma"],
  [/^gpt2$/, "GPT-2"], [/^gpt_neox/, "Pythia / GPT-NeoX"], [/^indranet$/, "IndraNet"]];
const familyOf = (m) => (FAMILIES.find(([re]) => re.test(m.model_type || "")) || [null, "Other"])[1];
const M = { list: [], canAdd: false, family: null, query: "", custom: false };

async function toggleMenu() {
  const menu = $("model-menu");
  if (!menu.hidden) return closeMenu();
  menu.hidden = false;
  $("model-btn").setAttribute("aria-expanded", "true");
  menu.innerHTML = `<div class="menu-left"><div class="note">Loading the list…</div></div>`;
  try {
    const r = await api("models");
    M.list = r.models;
    M.canAdd = r.can_add;
  } catch (e) {
    menu.innerHTML = `<div class="menu-left"><div class="note">${esc(e.message)}</div></div>`;
    return;
  }
  const current = M.list.find((m) => m.current);
  Object.assign(M, { family: current ? familyOf(current) : null, query: "", custom: false });
  renderMenu();
  $("model-search").focus();
}

function renderMenu() {
  const menu = $("model-menu");
  const families = [...new Set(M.list.map(familyOf))];
  const current = M.list.find((m) => m.current);
  menu.innerHTML = `<div class="menu-left">
      <label class="search"><input id="model-search" type="search" placeholder="Search models" autocomplete="off" spellcheck="false" aria-label="Search models">
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="m16 16 4.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></label>
      <div class="fams">${families.map((f) => {
        const n = M.list.filter((m) => familyOf(m) === f).length;
        const here = current && familyOf(current) === f;
        return `<button type="button" class="fam" data-family="${esc(f)}"><span>${esc(f)}${here ? ' <i class="dot" title="the loaded model"></i>' : ""}</span><span class="count">${n}</span><span class="chev">›</span></button>`;
      }).join("")}</div>
      <button type="button" class="fam custom" data-custom="1"><span>Custom model…</span><span class="chev">›</span></button>
    </div>
    <div class="menu-right" id="menu-right"></div>`;
  const search = $("model-search");
  search.value = M.query;
  search.oninput = () => { M.query = search.value.trim(); M.custom = false; renderRight(); };
  search.onkeydown = (e) => {
    const items = [...menu.querySelectorAll(".menu-right .item")];
    if (e.key === "Enter" && items.length) { e.preventDefault(); items[0].click(); }
    if (e.key === "ArrowDown" && items.length) { e.preventDefault(); items[0].focus(); }
  };
  const fine = window.matchMedia("(hover: hover)").matches;
  menu.querySelectorAll(".fam").forEach((b) => {
    const open = () => {
      const custom = !!b.dataset.custom;
      if (M.custom === custom && (custom || (M.family === b.dataset.family && !M.query))) return;
      Object.assign(M, { custom, family: custom ? M.family : b.dataset.family, query: "" });
      search.value = "";
      renderRight();
    };
    b.onclick = () => { open(); const first = menu.querySelector(".menu-right .item, .menu-right input"); if (first) first.focus(); };
    if (fine) b.onmouseenter = open;
  });
  renderRight();
}

function renderRight() {
  const menu = $("model-menu"), right = $("menu-right");
  const gb = (b) => (b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : (b / 1e6).toFixed(1) + " MB");
  const q = M.query.toLowerCase();
  const matches = (m) => [m.name, m.source, m.model_type, familyOf(m), m.folder].some((v) => v && v.toLowerCase().includes(q));
  // the row the panel opens from: highlighted, and the panel lines up with it (when side by side)
  let row = null;
  menu.querySelectorAll(".fam").forEach((b) => {
    const on = M.custom ? !!b.dataset.custom : !q && b.dataset.family === M.family;
    b.classList.toggle("active", on);
    b.classList.toggle("dim", !!q && !b.dataset.custom && !M.list.some((m) => familyOf(m) === b.dataset.family && matches(m)));
    if (on) row = b;
  });
  right.innerHTML = "";
  if (M.custom) {
    right.innerHTML = `<div class="panel-title">Open your own model</div>
      <p class="hint">A <code>.safetensors</code> file from <code>indranet.export</code> or <code>indranet_hf</code>, or a folder of them, on this computer.</p>
      <form class="add-form"><input id="model-path" type="text" placeholder="/path/to/model.safetensors" spellcheck="false" autocomplete="off" ${M.canAdd ? "" : "disabled"}>
        <button class="primary" type="submit" ${M.canAdd ? "" : "disabled"}>Open</button></form>
      <div id="add-error" class="status error"></div>
      <p class="hint">${M.canAdd ? "A Hugging Face checkpoint? Convert it first:<br><code>python -m indranet_hf Qwen/Qwen2.5-0.5B qwen.safetensors</code>"
        : "Adding models is off because this page is served to other computers; list them on the command line instead."}</p>`;
    right.querySelector("form").onsubmit = async (e) => {
      e.preventDefault();
      const path = $("model-path").value.trim();
      if (!path) return;
      try {
        const r = await api("add", { path });
        M.list = r.models;
        if (r.added.length === 1) return selectModel(M.list[r.added[0]]);
        const first = r.added.length ? M.list[r.added[0]] : M.list.find((m) => m.folder === path || m.name === path.split("/").pop());
        Object.assign(M, { custom: false, family: first ? familyOf(first) : M.family });
        renderMenu();
      } catch (err) {
        $("add-error").textContent = err.message;
      }
    };
  } else {
    const items = q ? M.list.filter(matches) : M.list.filter((m) => familyOf(m) === M.family);
    right.insertAdjacentHTML("beforeend", `<div class="panel-title">${q ? `Matching “${esc(M.query)}”` : esc(M.family || "")}</div>`);
    if (!items.length) right.insertAdjacentHTML("beforeend", `<div class="note">No model matches. Open one with Custom model.</div>`);
    for (const m of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "item";
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", m.current ? "true" : "false");
      b.setAttribute("aria-label", m.name);
      const parts = [m.source, m.n_layers && `${m.n_layers} layers`, m.d_model && `d ${m.d_model}`, m.size && gb(m.size)].filter(Boolean);
      b.innerHTML = `<b title="${esc(m.folder)}">${esc(m.name)}</b><span class="sub">${esc(parts.join(" · "))}</span><span class="mark">${m.current ? "✓" : ""}</span>`;
      b.onclick = () => selectModel(m);
      b.onkeydown = (e) => {
        const all = [...right.querySelectorAll(".item")], k = all.indexOf(b);
        if (e.key === "ArrowDown" && all[k + 1]) { e.preventDefault(); all[k + 1].focus(); }
        if (e.key === "ArrowUp") { e.preventDefault(); (all[k - 1] || $("model-search")).focus(); }
      };
      right.appendChild(b);
    }
  }
  const side = right.offsetLeft > 0;  // side by side; on narrow screens the panel wraps below
  right.style.marginTop = side && row ? Math.max(0, row.offsetTop - 6) + "px" : "0";
}

async function selectModel(m) {
  closeMenu();
  if (!m || m.current) return;
  const info = await busy(`Loading ${m.name}`, $("run"), () => api("select", { index: m.index }));
  Object.assign(S, { tokens: null, texts: null, run: null, attn: null, lens: null, dla: null, patch: null, heads: null, layer: 0, head: -1 });
  for (const id of ["patch-corrupt", "patch-target", "patch-other", "dla-target", "dla-other"]) $(id).value = "";
  applyInfo(info);
  setStatus("", false, "");
  $("prompt").focus();
}

// The landing page's picture: Indra's net, a lattice of jewels, each joined to its neighbours. It is
// quieter towards the middle, behind the card, and the same on every visit (a fixed seed).
function drawNet() {
  const c = $("net");
  const w = c.clientWidth, h = c.clientHeight, dpr = window.devicePixelRatio || 1;
  if (!w || !h) return;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bg = g.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, rgb(PAL["hero-a"]));
  bg.addColorStop(1, rgb(PAL["hero-b"]));
  g.fillStyle = bg;
  g.fillRect(0, 0, w, h);
  let seed = 20251117;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const s = clamp(Math.min(w, h) / 6.5, 70, 130), dy = s * 0.866;
  const rows = [];
  for (let r = 0, y = -dy; y < h + dy; r++, y += dy) {
    const row = [];
    for (let x = (r % 2 ? s / 2 : 0) - s; x < w + s; x += s) row.push([x + (rnd() - 0.5) * s * 0.24, y + (rnd() - 0.5) * s * 0.24, rnd()]);
    rows.push(row);
  }
  const cx = w / 2, cy = h * 0.5, far = Math.hypot(cx, cy);
  const fade = (x, y) => 0.18 + 0.82 * clamp(Math.hypot(x - cx, (y - cy) * 1.3) / far, 0, 1);
  const line = PAL["net-line"], jewel = PAL["net-jewel"];
  const rgba = (col, a) => `rgba(${col[0]},${col[1]},${col[2]},${a.toFixed(3)})`;
  g.lineWidth = 1;
  const join = (a, b) => {
    if (!a || !b) return;
    g.strokeStyle = rgba(line, 0.22 * fade((a[0] + b[0]) / 2, (a[1] + b[1]) / 2));
    g.beginPath();
    g.moveTo(a[0], a[1]);
    g.lineTo(b[0], b[1]);
    g.stroke();
  };
  rows.forEach((row, r) => row.forEach((p, i) => {
    join(p, row[i + 1]);
    const next = rows[r + 1];
    if (!next) return;
    if (r % 2) { join(p, next[i]); join(p, next[i + 1]); } else { join(p, next[i - 1]); join(p, next[i]); }
  }));
  for (const row of rows) {
    for (const [x, y, v] of row) {
      const f = fade(x, y);
      if (v > 0.9) {  // a lit jewel, with a glow
        const glow = g.createRadialGradient(x, y, 0, x, y, 18);
        glow.addColorStop(0, rgba(jewel, 0.45 * f));
        glow.addColorStop(1, rgba(jewel, 0));
        g.fillStyle = glow;
        g.beginPath(); g.arc(x, y, 18, 0, 2 * Math.PI); g.fill();
        g.fillStyle = rgba(jewel, 0.95 * f + 0.05);
        g.beginPath(); g.arc(x, y, 3.4, 0, 2 * Math.PI); g.fill();
        g.fillStyle = `rgba(255,255,255,${(0.8 * f).toFixed(3)})`;
        g.beginPath(); g.arc(x - 1.1, y - 1.1, 1.1, 0, 2 * Math.PI); g.fill();
      } else {
        g.fillStyle = rgba(jewel, 0.5 * f + 0.06);
        g.beginPath(); g.arc(x, y, 1.3 + v * 1.6, 0, 2 * Math.PI); g.fill();
      }
    }
  }
}

function renderStrip() {
  const strip = $("strip");
  strip.innerHTML = "";
  S.texts.forEach((t, i) => {
    const chip = document.createElement("span");
    chip.className = "chip" + (i === S.pos ? " selected" : "");
    chip.textContent = chipText(t);
    const p = i > 0 ? S.run.next_prob[i - 1] : null;
    if (p !== null) {
      const bg = mix(PAL.chip, PAL.neg, clamp(-Math.log(Math.max(p, 1e-30)) / 10, 0, 1));  // surprisal, 10 nats = full
      chip.style.background = rgb(bg);
      chip.style.color = textOn(bg);
    }
    const bar = document.createElement("span");
    bar.className = "bar";
    chip.appendChild(bar);
    chip.onmousemove = (e) => showTip(chipTip(i), e);
    chip.onmouseleave = hideTip;
    chip.onclick = () => selectPos(i);
    strip.appendChild(chip);
  });
  updateStripBars();
}

function chipTip(i) {
  const id = S.tokens[i], prev = i > 0 ? S.run.next_prob[i - 1] : null;
  let html = `<div class="mono"><b>${esc(vis(S.texts[i]))}</b> <span class="t">pos ${i} · id ${id}</span></div>`;
  if (prev !== null) html += `<div>p(this token) ${pct(prev)} · rank ${S.run.next_rank[i - 1]}</div>`;
  html += `<div class="t" style="margin-top:4px">predicted next:</div>` + topTable(S.run.top[i].map(([, t, p]) => [t, p]));
  return html;
}

function selectPos(i) {
  S.pos = i;
  document.querySelectorAll("#strip .chip").forEach((c, j) => c.classList.toggle("selected", j === i));
  updateStripBars();
  if (S.tab === "attention" && S.head >= 0) renderAttnDetail();
  if (S.tab === "lens") renderLens();
}

// Attention from the selected token, for the open head, as bars under the chips.
function updateStripBars() {
  const bars = document.querySelectorAll("#strip .chip .bar");
  const a = S.attn, show = S.tab === "attention" && a && S.head >= 0 && a.layer === S.layer;
  $("strip-hint").textContent = show
    ? `bars: where token ${S.pos} looks in L${S.layer}H${S.head} (darker = more attention); click another token to move the query`
    : "colour: how surprising each token was (−log p, full red at 10 nats); click one to select its position";
  bars.forEach((b, j) => {
    if (!show) { b.style.opacity = 0; return; }
    const T = a.T, row = (S.head * T + S.pos) * T;
    let max = 0;
    for (let k = 0; k <= S.pos; k++) max = Math.max(max, a.data[row + k]);
    b.style.opacity = j <= S.pos && max > 0 ? Math.max(0.04, a.data[row + j] / max) : 0;
  });
}

// --- tabs ------------------------------------------------------------------------------------
function openTab(tab) {
  S.tab = tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  for (const t of ["attention", "lens", "dla", "patch", "heads"]) $("panel-" + t).hidden = t !== tab || !S.tokens;
  hideTip();
  updateStripBars();
  if (!S.tokens) return;
  ({ attention: showAttention, lens: showLens, dla: showDla, patch: showPatch, heads: showHeads })[tab]();
}

// --- attention -------------------------------------------------------------------------------
async function showAttention() {
  const slider = $("attn-layer");
  slider.max = S.info.n_layers - 1;
  slider.value = S.layer;
  $("attn-layer-label").textContent = S.layer;
  if (!S.attn || S.attn.layer !== S.layer) {
    const layer = S.layer;
    const r = await busy(`Attention L${layer}`, null, () => api("attention", { tokens: S.tokens, layer }, true));
    if (layer !== S.layer) return;  // the slider moved on
    S.attn = { layer, H: r.shape[0], T: r.shape[1], data: r.data };
    if (S.head >= S.attn.H) S.head = -1;
  }
  renderHeadGrid();
  renderAttnDetail();
  updateStripBars();
}

function patternImage(canvas, h) {
  const { T, data } = S.attn;
  canvas.width = T;
  canvas.height = T;
  const g = canvas.getContext("2d");
  const img = g.createImageData(T, T);
  for (let q = 0; q < T; q++) {
    for (let k = 0; k < T; k++) {
      const c = k > q ? PAL.empty : seq(data[(h * T + q) * T + k]);
      const o = (q * T + k) * 4;
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
}

function renderHeadGrid() {
  const grid = $("attn-grid");
  grid.innerHTML = "";
  for (let h = 0; h < S.attn.H; h++) {
    const box = document.createElement("div");
    box.className = "head" + (h === S.head ? " open" : "");
    const c = document.createElement("canvas");
    patternImage(c, h);
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = `L${S.layer}H${h}`;
    box.append(c, name);
    box.onclick = () => { S.head = S.head === h ? -1 : h; renderHeadGrid(); renderAttnDetail(); updateStripBars(); };
    grid.appendChild(box);
  }
}

function renderAttnDetail() {
  const show = S.head >= 0 && S.attn;
  $("attn-detail").hidden = !show;
  if (!show) return;
  const { T, data } = S.attn, h = S.head;
  $("attn-detail-title").textContent = `L${S.layer}H${h}`;
  const cell = clamp(Math.floor(640 / T), 4, 30);
  const labels = S.texts.map((t, i) => `${i} ${vis(t)}`);
  heatmap($("attn-big"), {
    rows: T, cols: T, cellW: cell, cellH: cell, labelW: 120, rowLabels: labels, colLabels: labels,
    highlightCol: -1,
    color: (q, k) => (k > q ? PAL.empty : seq(data[(h * T + q) * T + k])),
    text: cell >= 30 ? (q, k) => {
      if (k > q) return "";
      const v = data[(h * T + q) * T + k];
      return v >= 0.995 ? "1" : fmt(v, 2).replace(/^0/, "");
    } : null,
    hover: (q, k) => k > q ? null :
      `<div class="mono">${esc(vis(S.texts[q]))} <span class="t">(${q})</span> → ${esc(vis(S.texts[k]))} <span class="t">(${k})</span></div>` +
      `<div>attention ${fmt(data[(h * T + q) * T + k], 4)}</div>`,
    click: (q) => selectPos(q),
  });
}

// --- logit lens -------------------------------------------------------------------------------
async function showLens() {
  if (!S.lens) S.lens = await busy("Logit lens", null, () => api("lens", { tokens: S.tokens, k: 5 }));
  renderLens();
}

function renderLens() {
  const L = S.lens;
  if (!L) return;
  const T = S.tokens.length, metric = $("lens-metric").value;
  const avail = $("panel-lens").clientWidth - 120;
  const cellW = clamp(Math.floor(avail / T), 30, 76);
  heatmap($("lens-canvas"), {
    rows: L.stages.length, cols: T, cellW, cellH: 20, labelW: 70,
    rowLabels: L.stages, colLabels: S.texts.map((t, i) => (cellW >= 44 ? vis(t) : `${i} ${vis(t)}`)),
    highlightCol: S.pos,
    color: (s, t) => seq(metric === "top" ? L.top_probs[s][t][0] : L.next_prob[s][t]),
    text: (s, t) => vis(L.top_texts[s][t][0]),
    hover: (s, t) => {
      let html = `<div><b>${esc(L.stages[s])}</b> <span class="t">at</span> <span class="mono">${esc(vis(S.texts[t]))}</span> <span class="t">(pos ${t})</span></div>`;
      html += topTable(L.top_texts[s][t].map((w, j) => [w, L.top_probs[s][t][j]]));
      if (t + 1 < T) html += `<div style="margin-top:3px">actual next <span class="mono">${esc(vis(S.texts[t + 1]))}</span>: ${pct(L.next_prob[s][t])}, rank ${L.next_rank[s][t]}</div>`;
      return html;
    },
    click: (s, t) => selectPos(t),
  });
}

// --- direct logit attribution --------------------------------------------------------------------
function showDla() {
  if ($("dla-pos").value === "") $("dla-pos").value = S.pos;
  if (S.dla) renderDla();
}

async function runDla() {
  const pos = $("dla-pos").value === "" ? S.pos : Number($("dla-pos").value);
  const body = { tokens: S.tokens, pos, target: $("dla-target").value, other: $("dla-other").value };
  S.dla = await busy("Attribution", $("dla-run"), () => api("dla", body));
  renderDla();
}

function renderDla() {
  const d = S.dla, info = S.info, L = info.n_layers, H = Math.max(...info.layer_heads);
  const vals = Object.fromEntries(d.labels.map((l, i) => [l, d.values[i]]));
  const what = d.other !== -1
    ? `logit(<b>${esc(vis(d.target_text))}</b>) − logit(<b>${esc(vis(d.other_text))}</b>)`
    : `logit(<b>${esc(vis(d.target_text))}</b>)`;
  $("dla-summary").className = "summary";
  $("dla-summary").innerHTML = `${what} at position ${d.pos} (<span class="mono">${esc(vis(S.texts[d.pos]))}</span>) = ` +
    `<b>${fmt(d.actual)}</b>; the parts sum to ${fmt(d.sum)}. Blue pushes towards the target, red away.`;
  const cell = (r, c) => (c < H ? vals[`L${r}H${c}`] : vals[`L${r}.mlp`]);
  let scale = 0;
  for (let r = 0; r < L; r++) for (let c = 0; c <= H; c++) { const v = cell(r, c); if (v !== undefined && v !== null) scale = Math.max(scale, Math.abs(v)); }
  // The map fills the left part of the panel; the largest parts are listed in the right.
  const left = $("dla-canvas").closest(".split").firstElementChild;
  const cw = clamp(Math.floor((left.clientWidth - 40 - 34) / (H + 1)), 8, 44), every = Math.ceil(16 / cw);
  heatmap($("dla-canvas"), {
    rows: L, cols: H + 1, cellW: cw, cellH: clamp(cw, 14, 24), labelW: 40,
    rowLabels: [...Array(L).keys()].map((i) => "L" + i),
    colLabels: [...[...Array(H).keys()].map((h) => (h % every ? "" : "H" + h)), "MLP"],
    color: (r, c) => { const v = cell(r, c); return v === undefined ? PAL.empty : div(v, scale); },
    text: (r, c) => { const v = cell(r, c); return v === undefined ? "" : fmt(v, 1); },
    hover: (r, c) => {
      const v = cell(r, c);
      if (v === undefined) return null;
      return `<div class="mono"><b>${c < H ? `L${r}H${c}` : `L${r} MLP`}</b> ${fmt(v, 4)}</div>` +
        (c < H ? `<div class="t">click to open its attention</div>` : "");
    },
    click: (r, c) => { if (c < H && cell(r, c) !== undefined) openHead(r, c); },
  });
  const order = d.labels.map((l, i) => [l, d.values[i]]).filter(([, v]) => v !== null)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 18);
  const max = Math.max(1e-9, ...order.map(([, v]) => Math.abs(v)));
  $("dla-bars").innerHTML = `<div class="hint" style="margin-bottom:4px">largest parts</div>` + order.map(([l, v]) => {
    const w = (50 * Math.abs(v)) / max, col = rgb(v < 0 ? PAL.neg : PAL.pos);
    return `<div class="bar-row"><span class="bar-label">${esc(l)}</span><span class="bar-track"><span class="bar-fill" style="left:${v < 0 ? 50 - w : 50}%;width:${w}%;background:${col}"></span></span><span class="bar-value">${fmt(v, 2)}</span></div>`;
  }).join("");
}

function openHead(layer, head) {
  S.layer = layer;
  S.head = head;
  openTab("attention");
}

// --- patching ---------------------------------------------------------------------------------
function showPatch() {
  if ($("patch-pos").value === "") $("patch-pos").value = S.pos;
  if (S.patch) renderPatch();
}

async function runPatch() {
  const text = $("patch-corrupt").value;
  if (!text.trim()) {
    $("patch-summary").className = "summary error";
    $("patch-summary").textContent = "Give a corrupted prompt with the same number of tokens as the prompt.";
    return;
  }
  await busy("Patching", $("patch-run"), async () => {
    const { tokens: corrupt, texts } = await toTokens(text);
    if (corrupt.length !== S.tokens.length) {
      throw new Error(`the corrupted prompt has ${corrupt.length} tokens, the prompt ${S.tokens.length}: ` +
        texts.map(vis).join(" | "));
    }
    const body = {
      clean: S.tokens, corrupt, pos: $("patch-pos").value === "" ? S.pos : Number($("patch-pos").value),
      target: $("patch-target").value, other: $("patch-other").value, site: $("patch-site").value,
      method: $("patch-method").value, steps: Number($("patch-steps").value) || 1, verify: Number($("patch-verify").value) || 10,
    };
    S.patch = await api("patch", body);
    S.patch.corruptTexts = texts;
    S.patch.site = body.site;
  });
  renderPatch();
}

function renderPatch() {
  const p = S.patch, rows = p.row_labels.length, cols = p.col_labels.length;
  const grid = p.normalized ? p.values : p.effect;
  let scale = 1;
  if (!p.normalized) { scale = 0; grid.forEach((r) => r.forEach((v) => { if (v !== null) scale = Math.max(scale, Math.abs(v)); })); }
  const metric = p.other !== -1 ? `logit(${vis(p.target_text)}) − logit(${vis(p.other_text)})` : `log p(${vis(p.target_text)})`;
  $("patch-summary").className = "summary";
  $("patch-summary").innerHTML = `<span class="mono">${esc(metric)}</span>: clean <b>${fmt(p.clean_metric)}</b>, corrupted <b>${fmt(p.corrupt_metric)}</b>. ` +
    (p.normalized ? "Each cell: the fraction of the clean metric restored by patching that site in from the clean run (0 = corrupted, 1 = clean)."
      : "The corruption does not change this metric, so cells show raw effects (patched − corrupted).") +
    (p.exact ? " Dotted cells were measured exactly; the rest are attribution estimates." : "");
  const isHeads = p.site === "heads";
  const colLabels = isHeads ? p.col_labels : p.col_labels.map((c, i) => {
    const a = S.texts[i], b = p.corruptTexts[i];
    return a === b ? vis(a) : `${vis(b)}→${vis(a)}`;
  });
  const cw = isHeads ? clamp(Math.floor(($("panel-patch").clientWidth - 100) / cols), 22, 48)
    : clamp(Math.floor(($("panel-patch").clientWidth - 100) / cols), 30, 72);
  heatmap($("patch-canvas"), {
    rows, cols, cellW: cw, cellH: 22, labelW: 44, rowLabels: p.row_labels, colLabels,
    color: (r, c) => div(grid[r][c], scale),
    text: (r, c) => (grid[r][c] === null ? "" : fmt(grid[r][c], 2)),
    mark: p.exact ? (r, c) => p.exact[r][c] === 1 : null,
    hover: (r, c) => {
      const where = isHeads ? `${p.row_labels[r]}${p.col_labels[c]}` : `${p.row_labels[r]} ${p.site} at ${c} (${vis(S.texts[c])})`;
      let html = `<div class="mono"><b>${esc(where)}</b></div>`;
      if (p.normalized) html += `<div>restored ${fmt(p.values[r][c])}</div>`;
      html += `<div>effect ${fmt(p.effect[r][c], 4)}</div>`;
      if (p.exact) html += `<div class="t">${p.exact[r][c] ? "measured exactly" : "attribution estimate"}</div>`;
      if (isHeads) html += `<div class="t">click to open its attention</div>`;
      return html;
    },
    click: isHeads ? (r, c) => openHead(r, c) : (r, c) => selectPos(c),
  });
}

// --- head scores ------------------------------------------------------------------------------
function showHeads() {
  if (S.heads) renderHeads();
}

async function runHeads() {
  const body = { segment_len: Number($("heads-seg").value) || 24, n_samples: Number($("heads-n").value) || 8 };
  S.heads = await busy("Head scores", $("heads-run"), () => api("heads", body));
  renderHeads();
}

function renderHeads() {
  const out = $("heads-out"), d = S.heads, L = S.info.n_layers, H = Math.max(...S.info.layer_heads);
  out.innerHTML = "";
  const names = { induction: "Induction", previous_token: "Previous token", duplicate_token: "Duplicate token", first_token: "First token (sink)", self: "Self" };
  const cell = clamp(Math.floor(260 / H), 6, 22), every = Math.ceil(13 / cell);  // label every row that fits
  for (const [key, title] of Object.entries(names)) {
    const box = document.createElement("div");
    box.innerHTML = `<h4>${title}</h4>`;
    const c = document.createElement("canvas");
    box.appendChild(c);
    out.appendChild(box);
    heatmap(c, {
      rows: L, cols: H, cellW: cell, cellH: cell, labelW: 34, labelH: 4,
      rowLabels: [...Array(L).keys()].map((i) => (i % every ? "" : "L" + i)),
      color: (r, h) => seq(d[key][r][h]),
      hover: (r, h) => (d[key][r][h] === null ? null : `<div class="mono"><b>L${r}H${h}</b> ${fmt(d[key][r][h])}</div><div class="t">click to open its attention</div>`),
      click: (r, h) => { if (d[key][r][h] !== null) openHead(r, h); },
    });
  }
  const hop = document.createElement("div");
  hop.innerHTML = `<h4>Two-hop induction (writer → reader)</h4><ol>` +
    d.top_two_hop.slice(0, 10).map(([w, r, v]) => `<li>${esc(w)} → ${esc(r)} ${fmt(v)}</li>`).join("") + "</ol>";
  out.appendChild(hop);
}

// --- start ---------------------------------------------------------------------------------------
function describe(info) {
  const gb = (info.weight_bytes / 1e9).toFixed(2);
  const params = count(info.num_parameters);
  const heads = new Set(info.layer_heads).size === 1 ? info.layer_heads[0] : "mixed";
  return `<b>${esc(info.file)}</b> · ${info.model_type ? esc(info.model_type) + " · " : ""}${params} parameters · ` +
    `${info.n_layers} layers × ${heads} heads · d_model ${info.d_model} · vocab ${info.vocab_size} · ` +
    `${info.fp16 ? "fp16" : "fp32"} ${gb} GB · device ${info.device}`;
}

async function start() {
  readPalette();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readPalette();
    drawNet();
    if (S.tokens) { renderStrip(); openTab(S.tab); }
  });
  drawNet();
  applyInfo(await api("info"));
  const prompt = $("prompt");
  history.replaceState({ view: "home" }, "", location.pathname);
  window.addEventListener("popstate", (e) => {
    if (e.state && e.state.view === "results" && S.tokens) showResults(false);
    else showHome();
  });
  $("edit").onclick = editPrompt;
  $("brand").onclick = editPrompt;
  $("run").onclick = runPrompt;
  prompt.addEventListener("input", growPrompt);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); runPrompt(); }
  });
  $("model-btn").onclick = toggleMenu;
  $("model-name").onclick = toggleMenu;
  document.addEventListener("click", (e) => {
    // the event's path, not e.target.closest: a click may re-render the element it landed on
    const path = e.composedPath();
    if (!path.some((n) => n.classList && n.classList.contains("ask")) && !path.includes($("model-name"))) closeMenu();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("model-menu").hidden) { closeMenu(); $("model-btn").focus(); } });
  document.querySelectorAll("#tabs button").forEach((b) => { b.onclick = () => openTab(b.dataset.tab); });
  $("attn-layer").oninput = (e) => { S.layer = Number(e.target.value); $("attn-layer-label").textContent = S.layer; showAttention(); };
  $("attn-close").onclick = () => { S.head = -1; renderHeadGrid(); renderAttnDetail(); updateStripBars(); };
  $("lens-metric").onchange = renderLens;
  $("dla-run").onclick = runDla;
  $("patch-run").onclick = runPatch;
  $("heads-run").onclick = runHeads;
  for (const id of ["dla-target", "dla-other", "dla-pos"]) $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") runDla(); });
  let timer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      drawNet();
      if (S.tab === "lens") renderLens();
      if (S.tab === "dla" && S.dla) renderDla();
      if (S.tab === "patch" && S.patch) renderPatch();
    }, 150);
  });
  prompt.focus();
}

start().catch((e) => {
  setStatus("could not reach the server: " + e.message, true);
});
