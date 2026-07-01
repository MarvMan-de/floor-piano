/* Floor Piano — Tile Configuration WebUI
 * Vanilla JS, no build step required.
 *
 * State machine modes:
 *   LIVE   — camera/video playing, SVG non-interactive
 *   FREEZE — playback paused / static frame, SVG interactive
 *   DRAW   — placing polygon vertices; double-click closes polygon
 *   SELECT — tile selected; drag vertices or interior to reshape/move
 *
 * Source types:
 *   camera — live camera via MJPEG, feed always /video_feed
 *   video  — video file via MJPEG; freeze = pause, live = play
 */

"use strict";

// ── Note names ────────────────────────────────────────────────────────────
const NOTE_NAMES = [
  "C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4",
  "C5","C#5","D5","D#5","E5","F5","F#5","G5","G#5","A5","A#5","B5",
];

// ── Colour palette ────────────────────────────────────────────────────────
const PALETTE = [
  "#4A90D9","#7ED321","#F5A623","#D0021B","#9013FE",
  "#50E3C2","#F8E71C","#BD10E0","#4A4A4A","#B8E986",
  "#417505","#9B9B9B","#D94A4A","#4AD97A","#D9C44A",
  "#4AD9D9","#D94AD9","#A04AD9","#4A7AD9","#D9804A",
  "#8AD94A","#D94A8A","#4AD9A0","#C44AD9",
];

// ── Shared state ──────────────────────────────────────────────────────────
const state = {
  // Canvas mode
  mode: "LIVE",        // "LIVE" | "FREEZE" | "DRAW" | "SELECT" | "CORNERS"
  tiles: [],
  selectedId: null,
  drawVertices: [],
  rotation: 0,          // view rotation in degrees: 0 | 90 | 180 | 270
  feedFW: null,         // manual on-screen footprint width  (px); null = auto-fit
  feedFH: null,         // manual on-screen footprint height (px); null = auto-fit
  dirty: false,
  undoStack: [],
  suggestions: [],
  dragState: null,
  cornerPoints: [],     // 4 mat corners (frame coords) for the corner tool
  frameSize: { w: 640, h: 480 },
  _nextId: 1,

  // Video / source
  video: {
    sourceType: "camera",  // "camera" | "video"
    path: null,
    playing: false,
    speed: 1,
    frame: 0,
    totalFrames: 0,
    fps: 30,
  },

  // Polling interval handles
  _statusInterval: null,
  _triggerInterval: null,
  // Scrubber being dragged (suppress status updates)
  _scrubbing: false,

  // Play/test mode: server does depth finger-detection, browser plays the notes
  playMode: false,
  _playInterval: null,
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const feedEl          = document.getElementById("feed");
const feedWrap        = document.getElementById("feed-wrap");
const feedContainer   = document.getElementById("feed-container");
const resizeHandles   = document.querySelectorAll(".resize-handle");
const handleE         = document.querySelector(".resize-handle.e");
const handleS         = document.querySelector(".resize-handle.s");
const handleSE        = document.querySelector(".resize-handle.se");
const overlayEl       = document.getElementById("overlay");
const tilesLayer      = document.getElementById("tiles-layer");
const suggestLayer    = document.getElementById("suggestions-layer");
const handlesLayer    = document.getElementById("handles-layer");
const cornersLayer    = document.getElementById("corners-layer");
const drawHandlesLayer = document.getElementById("draw-handles-layer");
const drawPreview     = document.getElementById("draw-preview");
const cursorLine      = document.getElementById("draw-cursor-line");
const tileList      = document.getElementById("tile-list");
const propsForm     = document.getElementById("props-form");
const statusMsg     = document.getElementById("status-msg");
const modeLabel     = document.getElementById("mode-label");
const btnFreeze     = document.getElementById("btn-freeze");
const btnSave       = document.getElementById("btn-save");
const btnDiscard    = document.getElementById("btn-discard");
const btnRotate     = document.getElementById("btn-rotate");
const btnAddDraw    = document.getElementById("btn-add-draw");
const btnAutodetect = document.getElementById("btn-autodetect");
const btnPlaceCorners = document.getElementById("btn-place-corners");
const btnPlay        = document.getElementById("btn-play");
const btnCaptureSurface = document.getElementById("btn-capture-surface");
const suggActions   = document.getElementById("suggestion-actions");
const btnAcceptAll  = document.getElementById("btn-accept-all");
const btnRejectAll  = document.getElementById("btn-reject-all");

// Source
const videoFileRow    = document.getElementById("video-file-row");
const videoFileSelect = document.getElementById("video-file-select");

// Video controls
const videoControls  = document.getElementById("video-controls");
const btnPlayPause   = document.getElementById("btn-play-pause");
const timeCurrent    = document.getElementById("time-current");
const timeTotal      = document.getElementById("time-total");
const timeline       = document.getElementById("timeline");
const speedSelect    = document.getElementById("speed-select");

// ── Init ──────────────────────────────────────────────────────────────────
async function init() {
  await fetchConfig();
  await initSources();
  setMode("LIVE");
  bindEvents();
  layoutFeed();
  // Re-fit the feed whenever its available area changes (window resize,
  // video controls appearing/disappearing, etc.)
  if (window.ResizeObserver) {
    new ResizeObserver(() => layoutFeed()).observe(feedWrap);
  } else {
    window.addEventListener("resize", layoutFeed);
  }
}

async function fetchConfig() {
  const res = await fetch("/api/config");
  const doc = await res.json();
  const size = { w: doc.frame_width || 640, h: doc.frame_height || 480 };
  state.frameSize = size;
  overlayEl.setAttribute("viewBox", `0 0 ${size.w} ${size.h}`);
  state.tiles = (doc.tiles || []).map(t => ({ ...t, id: t.id ?? state._nextId++ }));
  state._nextId = Math.max(state._nextId, ...state.tiles.map(t => t.id + 1), 1);
  state.dirty = false;
  renderAll();
  layoutFeed();
}

// ── Responsive feed sizing + rotation ─────────────────────────────────────
// Sizes #feed-container to fit the available area while preserving the frame
// aspect ratio, accounting for the current view rotation. Both the <img> and
// the SVG overlay live inside the container, so they rotate together and the
// polygon coordinates stay in the original (unrotated) frame space.
// Minimum on-screen footprint (px) when the user shrinks the feed.
const MIN_FEED_PX = 120;

function layoutFeed() {
  if (!feedWrap) return;
  const availW = feedWrap.clientWidth;
  const availH = feedWrap.clientHeight;
  if (availW <= 0 || availH <= 0) return;

  const { w, h } = state.frameSize;
  const rot = ((state.rotation % 360) + 360) % 360;
  const rotated = rot === 90 || rot === 270;

  // On-screen footprint (bounding box after rotation). Width and height are
  // controlled independently; null means "auto-fit at the native aspect".
  let fW, fH;
  if (state.feedFW == null || state.feedFH == null) {
    const boxW = rotated ? h : w;
    const boxH = rotated ? w : h;
    const s = Math.min(availW / boxW, availH / boxH);
    fW = boxW * s;
    fH = boxH * s;
  } else {
    fW = state.feedFW;
    fH = state.feedFH;
  }

  fW = Math.max(MIN_FEED_PX, Math.min(fW, availW));
  fH = Math.max(MIN_FEED_PX, Math.min(fH, availH));

  // Translate the footprint back into the container's own (pre-rotation) box.
  feedContainer.style.width  = `${Math.round(rotated ? fH : fW)}px`;
  feedContainer.style.height = `${Math.round(rotated ? fW : fH)}px`;
  feedContainer.style.transform = rot ? `rotate(${rot}deg)` : "";

  // Place the resize grips on the feed's on-screen edges (the grips live in
  // #feed-wrap, not the rotating container, so they stay screen-aligned).
  const cx = availW / 2;
  const cy = availH / 2;
  const rightX  = cx + fW / 2;
  const bottomY = cy + fH / 2;
  if (handleE)  { handleE.style.left  = `${rightX}px`;  handleE.style.top  = `${cy}px`; }
  if (handleS)  { handleS.style.left  = `${cx}px`;      handleS.style.top  = `${bottomY}px`; }
  if (handleSE) { handleSE.style.left = `${rightX}px`;  handleSE.style.top = `${bottomY}px`; }
}

function rotateFeed() {
  // Keep the current footprint but swap its dimensions so the picture doesn't
  // suddenly jump when a manual size is in effect.
  if (state.feedFW != null && state.feedFH != null) {
    [state.feedFW, state.feedFH] = [state.feedFH, state.feedFW];
  }
  state.rotation = (state.rotation + 90) % 360;
  layoutFeed();
}

// ── Drag-to-resize the feed ────────────────────────────────────────────────
// axis: "x" = width only, "y" = height only, "xy" = both (free).
function onResizeStart(e) {
  if (e.button !== undefined && e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();

  const axis = e.currentTarget.dataset.axis || "xy";

  // The container is centred in #feed-wrap, so its centre is fixed. Seed the
  // manual footprint from what's currently rendered so edge-drags have a base.
  const cur = feedContainer.getBoundingClientRect();
  if (state.feedFW == null) state.feedFW = cur.width;
  if (state.feedFH == null) state.feedFH = cur.height;

  const move = (ev) => {
    const rect = feedWrap.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    if (axis !== "y") state.feedFW = 2 * Math.abs(ev.clientX - cx);
    if (axis !== "x") state.feedFH = 2 * Math.abs(ev.clientY - cy);
    layoutFeed();
  };
  const up = () => {
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    document.body.style.userSelect = "";
  };
  document.body.style.userSelect = "none";
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up);
}

function resetFeedSize() {
  // Back to original format (native aspect, auto-fit).
  state.feedFW = null;
  state.feedFH = null;
  layoutFeed();
}

// ── Source initialisation ─────────────────────────────────────────────────
async function initSources() {
  try {
    const res = await fetch("/api/sources");
    const data = await res.json();
    const videos = data.videos || [];
    videoFileSelect.innerHTML = videos.length
      ? videos.map(v => `<option value="${esc(v)}">${esc(basename(v))}</option>`).join("")
      : `<option value="">— no videos found —</option>`;
  } catch (_) {
    videoFileSelect.innerHTML = `<option value="">— error loading sources —</option>`;
  }
}

function basename(path) {
  return path.replace(/^.*[\\/]/, "");
}

// ── Source switching ──────────────────────────────────────────────────────
async function switchSource(type, path) {
  const body = type === "video" ? { type, path } : { type };
  try {
    const res = await fetch("/api/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Source switch failed");
    state.video.sourceType = type;
    state.video.path = path || null;
    // Go back to LIVE to restart the stream
    setMode("LIVE");
    startVideoPolling(type === "video");
    setStatus(type === "video" ? `Video: ${basename(path)}` : "Live camera");
    setTimeout(() => setStatus(""), 3000);
  } catch (err) {
    setStatus("Source error: " + err.message, "err");
  }
}

// ── Playback control ──────────────────────────────────────────────────────
async function apiPlayback(action, speed) {
  const body = { action };
  if (speed !== undefined) body.speed = speed;
  await fetch("/api/playback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function apiSeek(frame) {
  await fetch("/api/seek", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame }),
  });
}

// ── Video status polling ──────────────────────────────────────────────────
function startVideoPolling(enabled) {
  clearInterval(state._statusInterval);
  clearInterval(state._triggerInterval);
  state._statusInterval = null;
  state._triggerInterval = null;

  if (!enabled) {
    videoControls.style.display = "none";
    return;
  }

  videoControls.style.display = "flex";

  state._statusInterval = setInterval(async () => {
    if (state._scrubbing) return;
    try {
      const res = await fetch("/api/video_status");
      const s = await res.json();
      state.video.playing = s.playing;
      state.video.frame = s.frame;
      state.video.totalFrames = s.total_frames;
      state.video.fps = s.fps || 30;
      updateVideoUI();
    } catch (_) {}
  }, 500);

  state._triggerInterval = setInterval(pollTriggers, 200);
}

function updateVideoUI() {
  const { frame, totalFrames, playing, speed } = state.video;

  btnPlayPause.textContent = playing ? "⏸ Pause" : "▶ Play";
  timeCurrent.textContent = formatTime(frame, state.video.fps);
  timeTotal.textContent   = formatTime(totalFrames, state.video.fps);

  // Update scrubber without causing feedback
  timeline.max = String(Math.max(totalFrames, 1));
  if (!state._scrubbing) timeline.value = String(frame);

  // Keep speed selector in sync if changed server-side
  if (speedSelect.value !== String(speed)) speedSelect.value = String(speed);

  // Sync mode badge when video plays/pauses externally
  if (playing && state.mode !== "LIVE") {
    // Don't force mode change here — user may be editing polygons
  }
}

function formatTime(frame, fps) {
  const secs = Math.floor(frame / (fps || 30));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ── Trigger polling & tile flash ──────────────────────────────────────────
async function pollTriggers() {
  try {
    const res = await fetch("/api/triggers");
    const data = await res.json();
    for (const ev of data.triggers || []) {
      flashTile(ev.tile_id);
      const tile = state.tiles.find(t => t.id === ev.tile_id);
      if (tile) playNote(tile.note);   // browser plays the pressed note
    }
  } catch (_) {}
}

function flashTile(tileId) {
  const polyEl = tilesLayer.querySelector(`[data-tile-id="${tileId}"]`);
  if (polyEl) {
    polyEl.classList.add("triggered");
    setTimeout(() => polyEl.classList.remove("triggered"), 200);
  }

  // Also flash the sidebar row
  const li = tileList.querySelector(`[data-tile-id="${tileId}"]`);
  if (li) {
    li.classList.add("triggered");
    setTimeout(() => li.classList.remove("triggered"), 200);
  }
}

// ── Play / test mode (server does depth finger-detection, browser plays) ────
async function captureSurface() {
  setStatus("Erfasse Oberfläche… (Finger weg vom Bild!)");
  try {
    const res = await fetch("/api/capture_surface", { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || "capture failed");
    const data = await res.json();
    setStatus(`Oberfläche erfasst (${Math.round((data.valid_frac || 0) * 100)}% gültig) `
              + "— jetzt Tasten drücken", "ok");
  } catch (err) {
    setStatus("Oberfläche erfassen fehlgeschlagen: " + err.message, "err");
  }
}

async function togglePlayMode() {
  const on = !state.playMode;
  try {
    const res = await fetch("/api/play", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: on }),
    });
    const data = await res.json().catch(() => ({}));
    state.playMode = on;
    if (btnPlay) btnPlay.textContent = on ? "⏹ Stop" : "▶ Play";
    clearInterval(state._playInterval);
    state._playInterval = null;
    if (on) {
      setMode("LIVE");   // live feed with the tiles overlaid; presses flash + sound
      state._playInterval = setInterval(pollTriggers, 120);
      setStatus(data.surface_ready
        ? "Play: drücke die Tasten im Bild"
        : "Play aktiv — bitte zuerst 'Oberfläche erfassen'",
        data.surface_ready ? "" : "err");
    } else {
      setStatus("Play beendet");
      setTimeout(() => setStatus(""), 1500);
    }
  } catch (err) {
    setStatus("Play-Umschalten fehlgeschlagen: " + err.message, "err");
  }
}

// ── Mode transitions ──────────────────────────────────────────────────────
function setMode(newMode) {
  const prevMode = state.mode;
  state.mode = newMode;
  modeLabel.textContent = newMode;
  modeLabel.className = `mode-badge ${newMode}`;

  const isVideo = state.video.sourceType === "video";
  const interactive = newMode !== "LIVE";
  overlayEl.classList.toggle("interactive", interactive);

  if (newMode === "LIVE") {
    // Always reconnect the MJPEG stream when going live
    feedEl.src = "/video_feed";
    btnFreeze.textContent = "Freeze";
    clearDrawPreview();
    state.drawVertices = [];
    state.selectedId = null;
    // Resume video if in video mode
    if (isVideo && prevMode !== "LIVE") {
      apiPlayback("play").catch(() => {});
      state.video.playing = true;
      updateVideoUI();
    }
    renderAll();
  } else if (newMode === "FREEZE") {
    btnFreeze.textContent = "Go Live";
    if (isVideo) {
      // Video mode: pause the stream; MJPEG keeps showing last frame
      if (prevMode === "LIVE") {
        apiPlayback("pause").catch(() => {});
        state.video.playing = false;
        updateVideoUI();
      }
      // feedEl.src stays as /video_feed — server keeps streaming the paused frame
    } else {
      // Camera mode: snap a static JPEG
      feedEl.src = `/api/frame?t=${Date.now()}`;
    }
  } else {
    // DRAW / SELECT inherit freeze state
    btnFreeze.textContent = "Go Live";
  }

  if (newMode === "DRAW") {
    overlayEl.style.cursor = "crosshair";
    setStatus("Click to add points, click a dot to close");
  } else if (newMode === "CORNERS") {
    overlayEl.style.cursor = "crosshair";
    setStatus(state.cornerPoints.length < 4
      ? `Ecke ${state.cornerPoints.length + 1}/4 der Matte klicken (Reihenfolge egal)`
      : "Ecken ziehen zum Feinjustieren, dann Save");
  } else {
    overlayEl.style.cursor = "";
  }

  if (newMode !== "DRAW") {
    clearDrawPreview();
    state.drawVertices = [];
  }

  renderAll();
}

// ── Rendering ─────────────────────────────────────────────────────────────
function renderAll() {
  renderTiles();
  renderHandles();
  renderCorners();
  renderSuggestions();
  renderSidebar();
  renderPropsPanel();
}

function renderCorners() {
  cornersLayer.innerHTML = "";
  if (state.mode !== "CORNERS") return;
  const pts = state.cornerPoints;
  if (pts.length >= 2) {
    const outline = svgEl(pts.length >= 4 ? "polygon" : "polyline", {
      points: pts.map(p => p.join(",")).join(" "),
      fill: "none", stroke: "#00e5ff", "stroke-width": "2",
      "stroke-dasharray": "6 4", "pointer-events": "none",
    });
    cornersLayer.appendChild(outline);
  }
  pts.forEach(([x, y], idx) => {
    const c = svgEl("circle", {
      cx: x, cy: y, r: "9", fill: "#00e5ff", stroke: "#00303a", "stroke-width": "2",
    });
    c.style.cursor = "grab";
    c.addEventListener("mousedown", e => onCornerMouseDown(e, idx));
    cornersLayer.appendChild(c);
    const t = svgEl("text", { x: x + 12, y: y - 10, class: "tile-text",
                              style: "fill:#00e5ff;font-size:13px" });
    t.textContent = String(idx + 1);
    cornersLayer.appendChild(t);
  });
}

function renderTiles() {
  tilesLayer.innerHTML = "";
  for (const tile of state.tiles) {
    if (tile.polygon.length < 3) continue;
    const pts = tile.polygon.map(p => p.join(",")).join(" ");
    const poly = svgEl("polygon", {
      points: pts,
      fill: tile.color || "#4A90D9",
      stroke: tile.color || "#4A90D9",
      "stroke-width": "2",
      "fill-opacity": tile.enabled ? "0.15" : "0.05",
      "stroke-opacity": tile.enabled ? "1" : "0.4",
      class: "tile-poly" + (tile.id === state.selectedId ? " selected" : ""),
      "data-tile-id": tile.id,
    });
    poly.addEventListener("mousedown", e => onPolyMouseDown(e, tile.id));
    poly.addEventListener("click", e => { e.stopPropagation(); onPolyClick(e, tile.id); });
    tilesLayer.appendChild(poly);

    const [cx, cy] = centroid(tile.polygon);
    const txt = svgEl("text", { x: cx, y: cy, class: "tile-text" });
    txt.textContent = tile.label;
    tilesLayer.appendChild(txt);
  }
}

function renderHandles() {
  handlesLayer.innerHTML = "";
  if (state.mode !== "SELECT" || state.selectedId === null) return;
  const tile = state.tiles.find(t => t.id === state.selectedId);
  if (!tile) return;
  tile.polygon.forEach(([x, y], idx) => {
    const c = svgEl("circle", { cx: x, cy: y, r: "6", class: "vertex-handle", "data-idx": idx });
    c.addEventListener("mousedown", e => onVertexMouseDown(e, tile.id, idx));
    handlesLayer.appendChild(c);
  });
}

function renderSuggestions() {
  suggestLayer.innerHTML = "";
  suggActions.style.display = state.suggestions.length ? "flex" : "none";
  state.suggestions.forEach((s, idx) => {
    const pts = s.polygon.map(p => p.join(",")).join(" ");
    const poly = svgEl("polygon", {
      points: pts,
      class: "suggestion-poly",
      "data-sugg-idx": idx,
    });
    poly.addEventListener("click", e => { e.stopPropagation(); acceptSuggestion(idx); });
    suggestLayer.appendChild(poly);
    const [cx, cy] = centroid(s.polygon);
    const txt = svgEl("text", { x: cx, y: cy, class: "tile-text", style: "fill:#ffdc00;font-size:11px" });
    txt.textContent = s.label;
    suggestLayer.appendChild(txt);
  });
}

function renderSidebar() {
  tileList.innerHTML = "";
  for (const tile of state.tiles) {
    const li = document.createElement("li");
    li.dataset.tileId = tile.id;
    if (tile.id === state.selectedId) li.classList.add("selected");
    if (!tile.enabled) li.classList.add("disabled");

    const swatch = document.createElement("div");
    swatch.className = "tile-swatch";
    swatch.style.background = tile.color || "#4A90D9";

    const lbl = document.createElement("span");
    lbl.className = "tile-label";
    lbl.textContent = tile.label;

    const note = document.createElement("span");
    note.className = "tile-note";
    note.textContent = tile.note;

    const tog = document.createElement("span");
    tog.className = "tile-toggle " + (tile.enabled ? "on" : "off");
    tog.textContent = tile.enabled ? "●" : "○";
    tog.title = tile.enabled ? "Disable" : "Enable";
    tog.addEventListener("click", e => {
      e.stopPropagation();
      pushUndo(); tile.enabled = !tile.enabled; state.dirty = true; renderAll();
    });

    const del = document.createElement("button");
    del.className = "tile-delete";
    del.textContent = "✕";
    del.title = "Delete tile";
    del.addEventListener("click", e => { e.stopPropagation(); removeTile(tile.id); });

    li.appendChild(swatch);
    li.appendChild(lbl);
    li.appendChild(note);
    li.appendChild(tog);
    li.appendChild(del);
    li.addEventListener("click", () => {
      playNote(tile.note);
      if (state.mode === "LIVE") return;
      selectTile(tile.id);
    });
    tileList.appendChild(li);
  }
}

function renderPropsPanel() {
  if (state.selectedId === null) {
    propsForm.innerHTML = '<p class="hint">Select a tile to edit its properties.</p>';
    return;
  }
  const tile = state.tiles.find(t => t.id === state.selectedId);
  if (!tile) { propsForm.innerHTML = ""; return; }

  propsForm.innerHTML = `
    <div class="prop-row">
      <label>Label</label>
      <input id="prop-label" type="text" value="${esc(tile.label)}" />
    </div>
    <div class="prop-row">
      <label>Note</label>
      <select id="prop-note">
        ${NOTE_NAMES.map(n => `<option value="${n}"${n === tile.note ? " selected" : ""}>${n}</option>`).join("")}
      </select>
    </div>
    <div class="prop-row">
      <label>Color</label>
      <input id="prop-color" type="color" value="${tile.color || "#4A90D9"}" />
    </div>
    <div class="prop-row">
      <label>Enabled</label>
      <input id="prop-enabled" type="checkbox" ${tile.enabled ? "checked" : ""} style="width:auto"/>
    </div>
    <div class="prop-row">
      <label>Polygon points</label>
      <textarea id="prop-polygon" rows="5" style="background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:4px;padding:5px 8px;font-size:11px;font-family:monospace;width:100%">${JSON.stringify(tile.polygon)}</textarea>
    </div>
  `;

  document.getElementById("prop-label").addEventListener("input", e => {
    pushUndo(); tile.label = e.target.value; state.dirty = true; renderTiles(); renderSidebar();
  });
  document.getElementById("prop-note").addEventListener("change", e => {
    pushUndo(); tile.note = e.target.value; state.dirty = true; renderSidebar();
  });
  document.getElementById("prop-color").addEventListener("input", e => {
    tile.color = e.target.value; state.dirty = true; renderTiles(); renderSidebar();
  });
  document.getElementById("prop-enabled").addEventListener("change", e => {
    pushUndo(); tile.enabled = e.target.checked; state.dirty = true; renderAll();
  });
  document.getElementById("prop-polygon").addEventListener("change", e => {
    try {
      const parsed = JSON.parse(e.target.value);
      if (!Array.isArray(parsed) || parsed.length < 3) throw new Error();
      pushUndo();
      tile.polygon = parsed;
      state.dirty = true;
      renderAll();
    } catch {
      e.target.style.borderColor = "var(--danger)";
      setTimeout(() => { e.target.style.borderColor = ""; }, 1500);
    }
  });
}

// ── Tile CRUD ─────────────────────────────────────────────────────────────
function addTile(polygon) {
  pushUndo();
  const usedNotes = new Set(state.tiles.map(t => t.note));
  const note = NOTE_NAMES.find(n => !usedNotes.has(n)) || NOTE_NAMES[0];
  const color = PALETTE[state.tiles.length % PALETTE.length];
  const id = state._nextId++;
  state.tiles.push({ id, label: note, note, polygon, color, enabled: true });
  state.dirty = true;
  selectTile(id);
  setMode("SELECT");
}

function removeTile(id) {
  pushUndo();
  state.tiles = state.tiles.filter(t => t.id !== id);
  if (state.selectedId === id) state.selectedId = null;
  state.dirty = true;
  renderAll();
}

function selectTile(id) {
  state.selectedId = id;
  if (state.mode === "FREEZE") setMode("SELECT");
  renderAll();
}

// ── Canvas events ─────────────────────────────────────────────────────────
function onOverlayClick(e) {
  if (state.mode === "CORNERS") {
    if (state.cornerPoints.length < 4) {
      state.cornerPoints.push(svgPoint(e));
      renderCorners();
      if (state.cornerPoints.length === 4) generateTilesFromCorners();
      else setStatus(`Ecke ${state.cornerPoints.length + 1}/4 der Matte klicken`);
    }
    return;
  }
  if (state.mode === "DRAW") {
    const pt = svgPoint(e);
    state.drawVertices.push(pt);
    updateDrawPreview();
    return;
  }
  if (state.mode === "SELECT" || state.mode === "FREEZE") {
    state.selectedId = null;
    setMode("FREEZE");
  }
}

function onOverlayDblClick(e) {
  if (state.mode !== "DRAW") return;
  e.preventDefault();
  if (state.drawVertices.length > 0) state.drawVertices.pop();
  if (state.drawVertices.length < 3) {
    setStatus("Need at least 3 points", "err");
    state.drawVertices = [];
    clearDrawPreview();
    return;
  }
  addTile([...state.drawVertices]);
  state.drawVertices = [];
  clearDrawPreview();
}

function onOverlayMouseMove(e) {
  if (state.mode === "DRAW" && state.drawVertices.length > 0) {
    const pt = svgPoint(e);
    const last = state.drawVertices[state.drawVertices.length - 1];
    cursorLine.setAttribute("x1", last[0]);
    cursorLine.setAttribute("y1", last[1]);
    cursorLine.setAttribute("x2", pt[0]);
    cursorLine.setAttribute("y2", pt[1]);
    cursorLine.style.display = "";
  }
  if (state.dragState) onDragMove(e);
}

function onOverlayMouseUp() {
  if (state.dragState) {
    const wasCorner = state.dragState.kind === "corner";
    state.dragState = null;
    renderHandles();
    if (wasCorner && state.cornerPoints.length === 4) generateTilesFromCorners();
  }
}

function onCornerMouseDown(e, idx) {
  if (e.button !== 0) return;
  e.stopPropagation();
  state.dragState = {
    kind: "corner",
    cornerIdx: idx,
    startPt: svgPoint(e),
    orig: [...state.cornerPoints[idx]],
  };
}

function onPolyClick(e, tileId) {
  if (state.mode === "DRAW") return;
  const tile = state.tiles.find(t => t.id === tileId);
  if (tile) playNote(tile.note);              // painted keys act as buttons in any mode
  if (state.mode === "LIVE" || state.mode === "CORNERS") return;  // just play, don't edit
  selectTile(tileId);
}

function onPolyMouseDown(e, tileId) {
  if (state.mode === "LIVE" || state.mode === "DRAW") return;
  if (e.button !== 0) return;
  e.stopPropagation();
  selectTile(tileId);
  const tile = state.tiles.find(t => t.id === tileId);
  if (!tile) return;
  pushUndo();
  state.dragState = {
    kind: "tile",
    tileId,
    startPt: svgPoint(e),
    origPoly: tile.polygon.map(p => [...p]),
  };
}

function onVertexMouseDown(e, tileId, vertexIdx) {
  if (state.mode === "LIVE") return;
  if (e.button !== 0) return;
  e.stopPropagation();
  const tile = state.tiles.find(t => t.id === tileId);
  if (!tile) return;
  pushUndo();
  state.dragState = {
    kind: "vertex",
    tileId,
    vertexIdx,
    startPt: svgPoint(e),
    origPoly: tile.polygon.map(p => [...p]),
  };
}

function onDragMove(e) {
  const ds = state.dragState;
  if (!ds) return;
  const pt = svgPoint(e);
  const dx = pt[0] - ds.startPt[0];
  const dy = pt[1] - ds.startPt[1];
  if (ds.kind === "corner") {
    state.cornerPoints[ds.cornerIdx] = [ds.orig[0] + dx, ds.orig[1] + dy];
    renderCorners();
    if (state.cornerPoints.length === 4) scheduleGenerate();  // live-ish snap
    return;
  }
  const tile = state.tiles.find(t => t.id === ds.tileId);
  if (!tile) return;
  if (ds.kind === "vertex") {
    tile.polygon = ds.origPoly.map((p, i) =>
      i === ds.vertexIdx ? [p[0] + dx, p[1] + dy] : [...p]
    );
  } else {
    tile.polygon = ds.origPoly.map(p => [p[0] + dx, p[1] + dy]);
  }
  state.dirty = true;
  renderTiles();
  renderHandles();
  updatePropsPolygonField(tile);
}

// ── Draw preview ──────────────────────────────────────────────────────────
function updateDrawPreview() {
  drawHandlesLayer.innerHTML = "";
  if (state.drawVertices.length === 0) { clearDrawPreview(); return; }

  drawPreview.setAttribute("points", state.drawVertices.map(p => p.join(",")).join(" "));
  drawPreview.style.display = "";

  // Draw a clickable dot on every placed vertex. Clicking one closes the
  // polygon at that point and finishes the tile.
  const closable = state.drawVertices.length >= 3;
  state.drawVertices.forEach(([x, y], idx) => {
    // Explicit r attribute — a CSS-only `r` (without px units) is ignored by
    // browsers and would render an invisible zero-radius circle.
    const c = svgEl("circle", {
      cx: x, cy: y,
      r: idx === 0 ? "8" : "6",
      class: "draw-vertex" + (idx === 0 ? " first" : "") + (closable ? " closable" : ""),
      "data-idx": idx,
    });
    c.addEventListener("mousedown", e => e.stopPropagation());
    c.addEventListener("click", e => { e.stopPropagation(); closeDrawPolygon(); });
    drawHandlesLayer.appendChild(c);
  });
}

function closeDrawPolygon() {
  if (state.drawVertices.length < 3) {
    setStatus("Need at least 3 points to close", "err");
    return;
  }
  addTile([...state.drawVertices]);
  state.drawVertices = [];
  clearDrawPreview();
}

function clearDrawPreview() {
  drawPreview.setAttribute("points", "");
  drawPreview.style.display = "none";
  cursorLine.style.display = "none";
  drawHandlesLayer.innerHTML = "";
}

// ── Autodetect ────────────────────────────────────────────────────────────
async function runAutodetect() {
  // Freeze first so we detect on a still frame
  if (state.mode === "LIVE") setMode("FREEZE");
  setStatus("Running autodetect…");
  try {
    const res = await fetch("/api/autodetect", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.suggestions = data.suggestions || [];
    setStatus(`${state.suggestions.length} suggestion(s) found`);
    renderSuggestions();
  } catch (err) {
    setStatus("Autodetect failed: " + err.message, "err");
  }
}

function acceptSuggestion(idx) {
  const s = state.suggestions[idx];
  if (!s) return;
  state.suggestions.splice(idx, 1);
  addTile(s.polygon);
  renderSuggestions();
}

function acceptAllSuggestions() {
  pushUndo();
  for (const s of state.suggestions) {
    const usedNotes = new Set(state.tiles.map(t => t.note));
    const note = NOTE_NAMES.find(n => !usedNotes.has(n)) || NOTE_NAMES[0];
    const color = PALETTE[state.tiles.length % PALETTE.length];
    const id = state._nextId++;
    state.tiles.push({ id, label: s.label || note, note: s.note || note, polygon: s.polygon, color, enabled: true });
  }
  state.suggestions = [];
  state.dirty = true;
  renderAll();
}

function clearSuggestions() {
  state.suggestions = [];
  renderSuggestions();
}

// ── Mat corner tool (reliable: place 4 corners -> project the 24 keys) ───────
function enterCornersMode() {
  if (state.mode === "LIVE") setMode("FREEZE");
  state.cornerPoints = [];
  state.suggestions = [];
  setMode("CORNERS");
}

async function runAutoCorners() {
  if (state.mode === "LIVE") setMode("FREEZE");
  setStatus("Suche Mattenecken…");
  try {
    const res = await fetch("/api/detect_corners", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (!data.corners) {
      state.cornerPoints = [];
      setMode("CORNERS");
      setStatus("Keine Matte erkannt — bitte die 4 Ecken manuell setzen", "err");
      return;
    }
    state.cornerPoints = data.corners.map(p => [Math.round(p[0]), Math.round(p[1])]);
    setMode("CORNERS");
    generateTilesFromCorners();
    setStatus("Ecken vorgeschlagen — auf die echten Mattenecken ziehen, dann Save");
  } catch (err) {
    setStatus("Auto-Ecken fehlgeschlagen: " + err.message, "err");
  }
}

// Sort 4 points into TL,TR,BR,BL so placement order doesn't matter.
function orderCorners(pts) {
  const sum = p => p[0] + p[1];
  const diff = p => p[0] - p[1];
  return [
    pts.reduce((a, b) => (sum(b) < sum(a) ? b : a)),   // TL: smallest x+y
    pts.reduce((a, b) => (diff(b) > diff(a) ? b : a)),  // TR: largest x-y
    pts.reduce((a, b) => (sum(b) > sum(a) ? b : a)),    // BR: largest x+y
    pts.reduce((a, b) => (diff(b) < diff(a) ? b : a)),  // BL: smallest x-y
  ];
}

let _genTimer = null;
function scheduleGenerate() {
  clearTimeout(_genTimer);
  _genTimer = setTimeout(generateTilesFromCorners, 120);  // debounce during drag
}

async function generateTilesFromCorners() {
  if (state.cornerPoints.length !== 4) return;
  try {
    const res = await fetch("/api/generate_tiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corners: orderCorners(state.cornerPoints) }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "generate failed");
    const data = await res.json();
    pushUndo();
    state.tiles = (data.tiles || []).map((t, i) => ({
      id: state._nextId++,
      label: t.label, note: t.note, polygon: t.polygon,
      color: PALETTE[i % PALETTE.length], enabled: true,
    }));
    state.dirty = true;
    state.suggestions = [];
    renderAll();
    setStatus(`${state.tiles.length} Tasten projiziert — Ecken justieren, dann Save`);
  } catch (err) {
    setStatus("Tasten-Erzeugung fehlgeschlagen: " + err.message, "err");
  }
}

// ── Save / discard ────────────────────────────────────────────────────────
async function saveConfig() {
  const doc = {
    tiles: state.tiles.map(({ id, label, note, polygon, color, enabled }) =>
      ({ id, label, note, polygon, color, enabled })
    ),
    camera_index: state.frameSize.camera_index || 0,
    frame_width: state.frameSize.w,
    frame_height: state.frameSize.h,
  };
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(doc),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Save failed");
    }
    state.dirty = false;
    setStatus("Saved ✓", "ok");
    setTimeout(() => setStatus(""), 3000);
  } catch (err) {
    setStatus(err.message, "err");
  }
}

async function discardChanges() {
  if (state.dirty && !confirm("Discard unsaved changes?")) return;
  await fetchConfig();
  state.suggestions = [];
  setMode("LIVE");
  setStatus("Changes discarded");
  setTimeout(() => setStatus(""), 2000);
}

// ── Undo ──────────────────────────────────────────────────────────────────
function pushUndo() {
  state.undoStack.push(JSON.parse(JSON.stringify(state.tiles)));
  if (state.undoStack.length > 50) state.undoStack.shift();
}

function undo() {
  if (!state.undoStack.length) return;
  state.tiles = state.undoStack.pop();
  state.dirty = true;
  renderAll();
}

// ── Status ────────────────────────────────────────────────────────────────
function setStatus(msg, cls = "") {
  statusMsg.textContent = msg;
  statusMsg.className = cls;
}

// ── Helpers ───────────────────────────────────────────────────────────────
function svgPoint(e) {
  // Use the SVG screen CTM so the mapping stays correct under any CSS
  // transform on the container (including the view rotation).
  const ctm = overlayEl.getScreenCTM();
  if (ctm) {
    const p = overlayEl.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const m = p.matrixTransform(ctm.inverse());
    return [Math.round(m.x), Math.round(m.y)];
  }
  const rect = overlayEl.getBoundingClientRect();
  const { w, h } = state.frameSize;
  return [
    Math.round((e.clientX - rect.left) * (w / rect.width)),
    Math.round((e.clientY - rect.top)  * (h / rect.height)),
  ];
}

function centroid(polygon) {
  const xs = polygon.map(p => p[0]);
  const ys = polygon.map(p => p[1]);
  return [
    Math.round(xs.reduce((a, b) => a + b, 0) / xs.length),
    Math.round(ys.reduce((a, b) => a + b, 0) / ys.length),
  ];
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

// Play a note's sample in the browser (test the layout by clicking tiles).
// Samples are served from /sounds; filename convention: C#4 -> Cs4.wav.
const _audioCache = {};
function playNote(note) {
  if (!note) return;
  try {
    let a = _audioCache[note];
    if (!a) {
      a = new Audio("/sounds/" + note.replace("#", "s") + ".wav");
      _audioCache[note] = a;
    }
    a.currentTime = 0;
    a.play().catch(() => {});  // ignore autoplay/decode errors (e.g. missing sample)
  } catch (_) {}
}

function updatePropsPolygonField(tile) {
  const f = document.getElementById("prop-polygon");
  if (f && tile.id === state.selectedId) f.value = JSON.stringify(tile.polygon);
}

// ── Bind events ───────────────────────────────────────────────────────────
function bindEvents() {
  // Toolbar
  btnFreeze.addEventListener("click", () => {
    if (state.mode === "LIVE") setMode("FREEZE");
    else setMode("LIVE");
  });
  btnSave.addEventListener("click", saveConfig);
  btnDiscard.addEventListener("click", discardChanges);
  btnAddDraw.addEventListener("click", () => {
    if (state.mode === "LIVE") setMode("FREEZE");
    setMode("DRAW");
  });
  btnRotate.addEventListener("click", rotateFeed);
  resizeHandles.forEach(h => {
    h.addEventListener("mousedown", onResizeStart);
    h.addEventListener("dblclick", resetFeedSize);
  });
  if (btnPlaceCorners) btnPlaceCorners.addEventListener("click", enterCornersMode);
  if (btnPlay) btnPlay.addEventListener("click", togglePlayMode);
  if (btnCaptureSurface) btnCaptureSurface.addEventListener("click", captureSurface);
  btnAutodetect.addEventListener("click", runAutoCorners);
  btnAcceptAll.addEventListener("click", acceptAllSuggestions);
  btnRejectAll.addEventListener("click", clearSuggestions);

  // Source radios
  document.querySelectorAll("input[name='source-type']").forEach(radio => {
    radio.addEventListener("change", async e => {
      const type = e.target.value;
      videoFileRow.style.display = type === "video" ? "block" : "none";
      if (type === "camera") {
        await switchSource("camera");
      } else {
        const path = videoFileSelect.value;
        if (path) await switchSource("video", path);
      }
    });
  });

  videoFileSelect.addEventListener("change", async () => {
    const path = videoFileSelect.value;
    if (path) await switchSource("video", path);
  });

  // Video controls
  btnPlayPause.addEventListener("click", async () => {
    const action = state.video.playing ? "pause" : "play";
    await apiPlayback(action, parseFloat(speedSelect.value));
    state.video.playing = !state.video.playing;
    if (state.video.playing) {
      setMode("LIVE");
    }
    updateVideoUI();
  });

  timeline.addEventListener("mousedown", () => { state._scrubbing = true; });
  timeline.addEventListener("input", () => {
    timeCurrent.textContent = formatTime(parseInt(timeline.value), state.video.fps);
  });
  timeline.addEventListener("change", async () => {
    const frame = parseInt(timeline.value);
    await apiSeek(frame);
    // Pause playback when user seeks (per spec)
    await apiPlayback("pause");
    state.video.playing = false;
    setMode("FREEZE");
    state._scrubbing = false;
    updateVideoUI();
  });
  timeline.addEventListener("mouseup", () => { state._scrubbing = false; });

  speedSelect.addEventListener("change", async () => {
    const speed = parseFloat(speedSelect.value);
    await apiPlayback(state.video.playing ? "play" : "pause", speed);
    state.video.speed = speed;
  });

  // SVG canvas
  overlayEl.addEventListener("click", onOverlayClick);
  overlayEl.addEventListener("dblclick", onOverlayDblClick);
  overlayEl.addEventListener("mousemove", onOverlayMouseMove);
  overlayEl.addEventListener("mouseup", onOverlayMouseUp);
  overlayEl.addEventListener("mousedown", e => {
    if (state.dragState || state.mode === "DRAW") e.preventDefault();
  });

  // Keyboard
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
      if (state.mode === "DRAW") {
        state.drawVertices = [];
        clearDrawPreview();
        setMode("FREEZE");
      } else if (state.mode === "SELECT") {
        state.selectedId = null;
        setMode("FREEZE");
      } else if (state.mode === "CORNERS") {
        setMode("FREEZE");
      }
    }
    if (e.key === "Delete" && state.selectedId !== null && state.mode !== "LIVE") {
      removeTile(state.selectedId);
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
      e.preventDefault();
      undo();
    }
    // Space: play/pause in video mode
    if (e.key === " " && state.video.sourceType === "video" && e.target.tagName !== "INPUT") {
      e.preventDefault();
      btnPlayPause.click();
    }
  });

  window.addEventListener("beforeunload", e => {
    if (state.dirty) { e.preventDefault(); e.returnValue = ""; }
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────
init().catch(err => {
  console.error("Init failed:", err);
  setStatus("Failed to load config: " + err.message, "err");
});
