const $ = (id) => document.getElementById(id);
const state = { file: null, dataUrl: "", svg: "", svgUrl: "", diagnostics: {}, scale: 1, x: 0, y: 0, split: 50, view: "compare", naturalW: 1, naturalH: 1, cudaAvailable: true, requestId: 0, requestController: null };

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
    resetDiagnostics();
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
  resetDiagnostics();
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
  const documentNode = new DOMParser().parseFromString(svg, "image/svg+xml");
  if (documentNode.querySelector("parsererror")) throw new Error("The generated SVG could not be parsed for preview.");
  const svgNode = documentNode.documentElement;
  svgNode.querySelectorAll("script, foreignObject").forEach(node => node.remove());
  svgNode.querySelectorAll("*").forEach(node => {
    for (const attribute of [...node.attributes]) if (attribute.name.toLowerCase().startsWith("on")) node.removeAttribute(attribute.name);
  });
  if (!svgNode.hasAttribute("viewBox")) {
    const width = Number.parseFloat(svgNode.getAttribute("width"));
    const height = Number.parseFloat(svgNode.getAttribute("height"));
    if (width > 0 && height > 0) svgNode.setAttribute("viewBox", `0 0 ${width} ${height}`);
  }
  svgNode.setAttribute("width", "100%");
  svgNode.setAttribute("height", "100%");
  svgNode.setAttribute("preserveAspectRatio", "none");
  svgNode.setAttribute("aria-label", "SVG output preview");
  els.svgPreview.replaceChildren(document.importNode(svgNode, true));
  return Promise.resolve();
}

function showImagePreview(src) {
  clearSvgPreview();
  const image = new Image(); image.alt = "Diagnostic preview"; image.src = src;
  els.svgPreview.replaceChildren(image);
}
const diagnosticLabels = { slic: "Coarse SLIC regions", crops: "Coarse model crops", coarse: "Coarse SVG", refined: "Neural refined SVG", final: "Final output" };
function resetDiagnostics() {
  state.diagnostics = {}; const select = $("diagnosticSelect");
  select.innerHTML = '<option value="final">Final output</option>'; select.disabled = true;
}
function addDiagnostic(event) {
  state.diagnostics[event.name] = event; const select = $("diagnosticSelect");
  if (!select.querySelector(`option[value="${event.name}"]`)) select.add(new Option(diagnosticLabels[event.name] || event.name, event.name));
  select.disabled = false;
}
$("diagnosticSelect").addEventListener("change", async e => {
  if (e.target.value === "final") { if (state.svg) await showSvgPreview(state.svg); return; }
  const item = state.diagnostics[e.target.value]; if (!item) return;
  if (item.kind === "svg") await showSvgPreview(item.data); else showImagePreview(item.data);
});

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
const settingHelp = {
  pathNum: ["Path count", "Sets the final number of SVG paths. Low: simpler, smaller and faster output with less detail. High: captures more detail, but takes longer and uses more GPU memory."],
  refinePaths: ["Refine density", "Estimated paths assigned to each refinement superpixel. Low: more, smaller regions and finer placement. High: fewer, larger regions and faster refinement."],
  optimizeIter: ["Fine-tune passes", "DiffVG optimization passes after neural vectorization. Low: fast and close to the neural result. High: closer pixel matching with substantially longer processing."],
  device: ["Device", "Chooses the inference processor. CPU is slower and restricted. CUDA uses the GPU and is strongly recommended."],
  batchSize: ["Batch size", "Refinement regions processed together. Low: less GPU memory but more batches. High: faster when memory allows, but may cause an out-of-memory error."],
  seed: ["Seed", "Controls deterministic initialization. Different values may produce slightly different paths; it is not a quality scale."],
  coarseRegionSize: ["Coarse region size", "Target paths represented by each initial superpixel. Low: more, smaller coarse regions. High: fewer, larger regions and faster processing."],
  coarseMargin: ["Coarse margin", "Extra render pixels around coarse crops. Low: paths stay close to region boundaries. High: paths can cross farther, reducing seams but increasing overlap."],
  coarseContext: ["Context strength", "Neighboring RGB information shown inside the coarse margin. Low: isolates each exact superpixel. High: gives the model surrounding visual context, which may improve continuity but can create overlapping predictions."],
  refineMargin: ["Refine margin", "Extra render pixels around refinement crops. Low: tightly localized paths. High: more cross-boundary freedom and overlap."],
  workingResolution: ["Working resolution", "Maximum internal raster dimension. Low: faster and lighter, but loses small details. High: sharper detail with much higher time and memory cost."],
  coarseCompactness: ["Coarse compactness", "Balances color boundaries against regular region shape. Low: follows image colors and edges. High: creates more uniform, geometric regions."],
  refineCompactness: ["Refine compactness", "Controls refinement superpixel shape. Low: follows local color boundaries. High: produces more regular spatial regions."],
  slicSigma: ["SLIC smoothing", "Blurs the image before segmentation. Low: preserves texture, noise and fine edges. High: suppresses noise and favors larger, smoother structures."],
  segmentationMode: ["Segmentation mode", "SLIC uses the selected global compactness. SLICO adapts compactness locally, helping images that mix smooth areas with detailed boundaries."],
  adaptiveSplit: ["Adaptive allocation", "Redistributes the fixed coarse-region budget using edge and color detail. Low: uniform global SLIC allocation. High: fewer regions on smooth areas and more around text, edges and complex color changes."],
  learningRate: ["Learning rate", "How far DiffVG moves parameters per pass. Low: stable, subtle changes needing more passes. High: faster changes that may overshoot or become unstable."],
  pathPenalty: ["Path penalty", "Penalizes active paths during fine-tuning. Low: preserves paths and detail. High: disables more paths for a simpler SVG, potentially losing detail."]
};
function showSettingHelp(target) {
  const field = target.closest("label")?.querySelector("input, select"), help = field && settingHelp[field.id];
  if (!help) return;
  $("settingInfoTitle").textContent = help[0]; $("settingInfoBody").textContent = help[1];
}
document.querySelectorAll(".control, .advanced-grid label").forEach(label => {
  label.addEventListener("mouseenter", e => showSettingHelp(e.currentTarget));
  label.addEventListener("focusin", e => showSettingHelp(e.currentTarget));
});
$("resetButton").addEventListener("click", () => {
  els.pathNum.value = 1000; els.refine.value = 8; els.optimize.value = 0;
  [els.pathNum, els.refine, els.optimize].forEach(el => el.dispatchEvent(new Event("input")));
  $("device").value = $("device").querySelector('option[value="cuda"]').disabled ? "cpu" : "cuda";
  $("batchSize").value = "64"; $("seed").value = 0;
  $("coarseRegionSize").value = 64; $("coarseMargin").value = 2; $("coarseContext").value = 0; $("refineMargin").value = 0; $("workingResolution").value = "512";
  $("coarseCompactness").value = 50; $("refineCompactness").value = 20; $("slicSigma").value = 5;
  $("segmentationMode").value = "slic"; $("adaptiveSplit").value = 0;
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
  resetDiagnostics();
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
        coarse_paths_per_segment: Number($("coarseRegionSize").value), coarse_margin: Number($("coarseMargin").value), coarse_context_strength: Number($("coarseContext").value),
        refine_margin: Number($("refineMargin").value), working_resolution: Number($("workingResolution").value),
        coarse_compactness: Number($("coarseCompactness").value), refine_compactness: Number($("refineCompactness").value),
        slic_sigma: Number($("slicSigma").value), slic_zero: $("segmentationMode").value === "slico", adaptive_split: Number($("adaptiveSplit").value),
        learning_rate: Number($("learningRate").value), path_penalty: Number($("pathPenalty").value)
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
        else if (event.type === "diagnostic") {
          addDiagnostic(event);
          appendLog(`Diagnostic ready: ${diagnosticLabels[event.name] || event.name}`);
        }
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
    state.diagnostics.final = { name: "final", kind: "svg", data: result.svg };
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
