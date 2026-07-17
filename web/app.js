const $ = (id) => document.getElementById(id);
const state = { file: null, dataUrl: "", svg: "", svgUrl: "", scale: 1, x: 0, y: 0, split: 50, view: "compare", naturalW: 1, naturalH: 1, cudaAvailable: true, requestId: 0, requestController: null };

const els = {
  sidebar: $("sidebar"), fileInput: $("fileInput"), dropZone: $("dropZone"), importCard: $("dropZone"),
  fileChip: $("fileChip"), fileThumb: $("fileThumb"), fileName: $("fileName"), fileMeta: $("fileMeta"),
  empty: $("emptyState"), canvas: $("canvas"), stage: $("stage"), comparison: $("comparison"), panLayer: $("panLayer"),
  input: $("inputPreview"), outputClip: $("outputClip"), svgPreview: $("svgPreview"), wipe: $("wipe"),
  processing: $("processing"), toast: $("toast"), vectorize: $("vectorizeButton"), download: $("downloadButton"),
  pathNum: $("pathNum"), refine: $("refinePaths"), optimize: $("optimizeIter")
};

function openPicker() { els.fileInput.click(); }
["browseButton", "emptyImport"].forEach(id => $(id).addEventListener("click", openPicker));
els.dropZone.addEventListener("click", e => { if (e.target.id !== "browseButton") openPicker(); });
els.fileInput.addEventListener("change", e => loadFile(e.target.files[0]));
["dragenter", "dragover"].forEach(type => els.dropZone.addEventListener(type, e => { e.preventDefault(); els.dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach(type => els.dropZone.addEventListener(type, e => { e.preventDefault(); els.dropZone.classList.remove("dragging"); }));
els.dropZone.addEventListener("drop", e => loadFile(e.dataTransfer.files[0]));

function loadFile(file) {
  if (!file || !/^image\/(png|jpeg|webp|bmp)$/.test(file.type)) return showToast("Choose a PNG, JPG, WebP, or BMP image.");
  if (file.size > 25 * 1024 * 1024) return showToast("That image is larger than 25 MB.");
  const reader = new FileReader();
  reader.onload = () => {
    clearSvgPreview();
    state.file = file; state.dataUrl = reader.result; state.svg = "";
    els.input.onload = () => {
      state.naturalW = els.input.naturalWidth; state.naturalH = els.input.naturalHeight;
      fitToView();
      els.fileMeta.textContent = `${state.naturalW} × ${state.naturalH} · ${formatBytes(file.size)}`;
    };
    els.input.src = state.dataUrl; els.fileThumb.src = state.dataUrl; els.fileName.textContent = file.name;
    els.importCard.classList.add("hidden"); els.fileChip.classList.remove("hidden");
    els.empty.classList.add("hidden"); els.canvas.classList.remove("hidden");
    els.svgPreview.innerHTML = `<img src="${state.dataUrl}" alt="">`;
    els.vectorize.disabled = false; els.download.disabled = true; setView("input");
  };
  reader.readAsDataURL(file);
}

function removeFile() {
  clearSvgPreview();
  state.file = null; state.dataUrl = ""; state.svg = ""; els.fileInput.value = "";
  els.fileChip.classList.add("hidden"); els.importCard.classList.remove("hidden"); els.canvas.classList.add("hidden");
  els.empty.classList.remove("hidden"); els.vectorize.disabled = true; els.download.disabled = true;
}
$("removeFile").addEventListener("click", removeFile);

function clearSvgPreview() {
  if (state.svgUrl) URL.revokeObjectURL(state.svgUrl);
  state.svgUrl = "";
  els.svgPreview.replaceChildren();
}

function showSvgPreview(svg) {
  clearSvgPreview();
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  state.svgUrl = URL.createObjectURL(blob);
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.alt = "SVG output preview";
    image.onload = () => {
      els.svgPreview.replaceChildren(image);
      resolve();
    };
    image.onerror = () => reject(new Error("The SVG was generated but the browser could not render its preview."));
    image.src = state.svgUrl;
  });
}

function formatBytes(bytes) { return bytes < 1048576 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / 1048576).toFixed(1)} MB`; }
function setRange(input, output, formatter = v => v) {
  const update = () => {
    output.textContent = formatter(input.value);
    const pct = (input.value - input.min) / (input.max - input.min) * 100;
    input.style.setProperty("--range", `${pct}%`);
  };
  input.addEventListener("input", update); update();
}
setRange(els.pathNum, $("pathNumOutput"), v => Number(v).toLocaleString());
setRange(els.refine, $("refineOutput"));
setRange(els.optimize, $("optimizeOutput"));
[els.pathNum, els.optimize].forEach(el => el.addEventListener("input", updateSettingsLimits));
function updateSettingsLimits() {
  if (!state.cudaAvailable) {
    const safePasses = Math.max(0, Math.min(100, Math.floor(80000 / Number(els.pathNum.value)) - 1));
    const newMax = Math.floor(safePasses / 5) * 5;
    const maxChanged = Number(els.optimize.max) !== newMax;
    els.optimize.max = newMax;
    const valueChanged = Number(els.optimize.value) > newMax;
    if (valueChanged) els.optimize.value = newMax;
    if (maxChanged || valueChanged) els.optimize.dispatchEvent(new Event("input", { bubbles: false }));
  }
  updateSettingsWarning();
}
function updateSettingsWarning() {
  const paths = Number(els.pathNum.value), passes = Number(els.optimize.value);
  const warning = $("settingsWarning");
  if (!state.cudaAvailable && paths >= 2500) {
    warning.textContent = `CPU safety limit: fine-tuning is capped at ${els.optimize.max} passes for ${paths.toLocaleString()} paths.`;
    warning.classList.remove("hidden");
  } else warning.classList.add("hidden");
}
updateSettingsLimits();
$("resetButton").addEventListener("click", () => {
  els.pathNum.value = 1000; els.refine.value = 8; els.optimize.value = 0;
  [els.pathNum, els.refine, els.optimize].forEach(el => el.dispatchEvent(new Event("input")));
  $("device").value = $("device").querySelector('option[value="cuda"]').disabled ? "cpu" : "cuda";
  $("batchSize").value = "64"; $("seed").value = 0;
  $("coarseRegionSize").value = 64; $("coarseMargin").value = 2; $("refineMargin").value = 0; $("workingResolution").value = "512";
  $("coarseCompactness").value = 50; $("refineCompactness").value = 20; $("slicSigma").value = 5;
  $("learningRate").value = 0.001; $("pathPenalty").value = 0.000001;
});

$("collapseButton").addEventListener("click", () => els.sidebar.classList.toggle("collapsed"));
$("openButton").addEventListener("click", () => els.sidebar.classList.remove("collapsed"));

function renderTransform() {
  els.panLayer.style.transform = `translate(calc(-50% + ${state.x}px), calc(-50% + ${state.y}px)) scale(${state.scale})`;
  els.wipe.style.setProperty("--wipe-width", `${1 / state.scale}px`);
  els.wipe.style.setProperty("--wipe-hit-width", `${24 / state.scale}px`);
  $("zoomReadout").textContent = `${Math.round(state.scale * 100)}%`;
}
function fitToView() {
  if (!state.file) return;
  const rect = els.stage.getBoundingClientRect(), maxW = rect.width - 80, maxH = rect.height - 80;
  const fitted = Math.min(maxW / state.naturalW, maxH / state.naturalH, 1);
  els.comparison.style.width = `${state.naturalW}px`; els.comparison.style.height = `${state.naturalH}px`;
  state.scale = Math.max(.05, fitted); state.x = 0; state.y = 0; renderTransform();
}
function zoomBy(factor) { state.scale = Math.max(.05, Math.min(8, state.scale * factor)); renderTransform(); }
$("zoomIn").addEventListener("click", () => zoomBy(1.2));
$("zoomOut").addEventListener("click", () => zoomBy(1 / 1.2));
$("fitButton").addEventListener("click", fitToView);
els.stage.addEventListener("wheel", e => { if (!state.file) return; e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.1 : 1 / 1.1); }, { passive: false });

let interaction = null;
els.canvas.addEventListener("pointerdown", e => {
  if (e.target.closest(".wipe")) interaction = { type: "wipe" };
  else { interaction = { type: "pan", sx: e.clientX, sy: e.clientY, x: state.x, y: state.y }; els.canvas.classList.add("panning"); }
  els.canvas.setPointerCapture(e.pointerId);
});
els.canvas.addEventListener("pointermove", e => {
  if (!interaction) return;
  if (interaction.type === "pan") { state.x = interaction.x + e.clientX - interaction.sx; state.y = interaction.y + e.clientY - interaction.sy; renderTransform(); }
  else {
    const rect = els.comparison.getBoundingClientRect();
    state.split = Math.max(0, Math.min(100, (e.clientX - rect.left) / rect.width * 100));
    updateSplit();
  }
});
els.canvas.addEventListener("pointerup", () => { interaction = null; els.canvas.classList.remove("panning"); });

function updateSplit() {
  const split = state.view === "input" ? 100 : state.view === "output" ? 0 : state.split;
  els.outputClip.style.left = "0";
  els.outputClip.style.clipPath = `inset(0 0 0 ${split}%)`;
  els.svgPreview.style.left = "0";
  els.svgPreview.style.width = "100%";
  els.wipe.style.left = `${split}%`;
  els.wipe.style.display = state.view === "compare" ? "" : "none";
}
function setView(view) {
  state.view = view;
  document.querySelectorAll(".view-tabs button").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  updateSplit();
}
document.querySelectorAll(".view-tabs button").forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));

els.vectorize.addEventListener("click", vectorize);
async function vectorize() {
  if (!state.file) return;
  if (state.requestController) state.requestController.abort();
  const requestId = ++state.requestId;
  const controller = new AbortController();
  let completed = false;
  state.requestController = controller;
  els.processing.classList.remove("hidden", "preview-ready", "idle"); els.vectorize.disabled = false; hideToast();
  els.vectorize.querySelector("span").textContent = "Restart vectorization";
  $("progressBar").style.width = "0%"; $("progressPercent").textContent = "0%";
  $("progressMessage").textContent = "Starting vectorization…"; $("processLog").textContent = "";
  try {
    const response = await fetch("/api/vectorize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        image: state.dataUrl, mime_type: state.file.type, path_num: Number(els.pathNum.value),
        optimize_iter: Number(els.optimize.value), refine_paths_per_segment: Number(els.refine.value),
        refine_batch_size: Number($("batchSize").value), seed: Number($("seed").value), device: $("device").value,
        coarse_paths_per_segment: Number($("coarseRegionSize").value), coarse_margin: Number($("coarseMargin").value),
        refine_margin: Number($("refineMargin").value), working_resolution: Number($("workingResolution").value),
        coarse_compactness: Number($("coarseCompactness").value), refine_compactness: Number($("refineCompactness").value),
        slic_sigma: Number($("slicSigma").value), learning_rate: Number($("learningRate").value), path_penalty: Number($("pathPenalty").value)
      })
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "Vectorization failed.");
    }
    if (!response.body) throw new Error("Streaming is unavailable in this browser.");
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "", result = null;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "progress") {
          $("progressBar").style.width = `${event.percent}%`;
          $("progressPercent").textContent = `${event.percent}%`;
          $("progressMessage").textContent = event.message;
          appendLog(`[${event.percent}%] ${event.message}`);
        } else if (event.type === "log") appendLog(event.message);
        else if (event.type === "preview") {
          state.svg = event.svg;
          await showSvgPreview(state.svg);
          els.download.disabled = false;
          state.split = 50;
          setView("compare");
          $("progressMessage").textContent = "Initial SVG ready · Fine-tuning…";
          appendLog(`Initial SVG displayed (${Math.round(state.svg.length / 1024).toLocaleString()} KB)`);
        }
        else if (event.type === "error") throw new Error(event.message);
        else if (event.type === "cancelled") throw new DOMException(event.message, "AbortError");
        else if (event.type === "result") result = event;
      }
      if (done) break;
    }
    if (!result) throw new Error("Vectorization ended without returning an SVG.");
    state.svg = result.svg;
    $("progressMessage").textContent = "Loading SVG preview…";
    appendLog(`SVG received (${Math.round(state.svg.length / 1024).toLocaleString()} KB)`);
    await showSvgPreview(state.svg);
    appendLog("SVG preview loaded");
    completed = true;
    els.download.disabled = false; state.split = 50; setView("compare");
  } catch (error) {
    if (error.name !== "AbortError") {
      $("progressMessage").textContent = "Vectorization failed";
      showToast(error.message);
    } else if (requestId === state.requestId) $("progressMessage").textContent = "Vectorization cancelled";
  } finally {
    if (requestId === state.requestId) {
      state.requestController = null;
      els.processing.classList.add("idle");
      if (completed) $("progressMessage").textContent = "SVG ready";
      els.vectorize.disabled = false;
      els.vectorize.querySelector("span").textContent = "Vectorize image";
    }
  }
}

function appendLog(message) {
  const log = $("processLog");
  log.textContent += `${message}\n`;
  log.scrollTop = log.scrollHeight;
}

els.download.addEventListener("click", () => {
  const blob = new Blob([state.svg], { type: "image/svg+xml" }), link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = `${state.file.name.replace(/\.[^.]+$/, "")}.svg`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
function showToast(message) { els.toast.textContent = message; els.toast.classList.remove("hidden"); clearTimeout(showToast.timer); showToast.timer = setTimeout(hideToast, 6000); }
function hideToast() { els.toast.classList.add("hidden"); }
window.addEventListener("resize", () => state.file && fitToView());
window.addEventListener("beforeunload", () => {
  if (state.requestController) state.requestController.abort();
  if (state.svgUrl) URL.revokeObjectURL(state.svgUrl);
});
window.addEventListener("keydown", e => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") vectorize(); });
fetch("/api/health").then(response => response.json()).then(data => {
  state.cudaAvailable = data.cuda_available;
  if (data.cuda_required) {
    $("device").value = "cuda";
    $("device").querySelector('option[value="cpu"]').disabled = true;
    $("device").querySelector('option[value="cuda"]').textContent = data.cuda_device || "CUDA GPU";
  }
  if (!data.cuda_available) {
    $("device").value = "cpu";
    $("device").querySelector('option[value="cuda"]').disabled = true;
    $("device").querySelector('option[value="cuda"]').textContent = "CUDA unavailable";
    updateSettingsLimits();
  }
}).catch(() => {});
updateSplit();
