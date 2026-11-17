"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const defaultTasks = [
    ["bottle", "瓶子"], ["bowl", "碗"], ["car", "汽车"],
    ["chair", "椅子"], ["clock", "时钟"], ["cup", "杯子"],
    ["fork", "叉子"], ["keyboard", "键盘"], ["knife", "刀"],
    ["laptop", "笔记本电脑"], ["microwave", "微波炉"],
    ["mouse", "鼠标"], ["oven", "烤箱"], ["potted plant", "盆栽"],
    ["sink", "水槽"], ["stop sign", "停车标志"],
    ["toilet", "马桶"], ["tv", "电视"]
  ].map(([id, label]) => ({ id, label: `${label} · ${id}` }));
  const state = {
    status: null, condition: "TP", image: null, imageData: null,
    exampleId: null, result: null, heatmaps: [], heatmapCards: [],
    visibleCount: 1,
    busy: false, timer: null, elapsedTimer: null, exampleRequest: 0,
    imageRequest: 0, imageLoading: false, examples: [], exampleOffset: 0
  };
  const ctx = $("result-canvas").getContext("2d");
  const maxFileBytes = 10 * 1024 * 1024;

  function message(text = "") {
    $("message").textContent = text;
    $("message").hidden = !text;
  }

  async function requestJson(url, options) {
    const response = await fetch(url, options);
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`服务返回了无效响应（HTTP ${response.status}）`); }
    if (!response.ok) {
      const error = payload.error || payload.detail || payload.message;
      throw new Error(typeof error === "string" ? error :
        error?.message || `请求失败（HTTP ${response.status}）`);
    }
    return payload;
  }

  function populateTasks(tasks) {
    const current = $("task").value;
    $("task").replaceChildren();
    tasks.forEach((task) => {
      const option = document.createElement("option");
      option.value = String(task.id);
      const label = task.label || String(task.id);
      option.textContent = label.includes(String(task.id)) ? label :
        `${label} · ${task.id}`;
      $("task").append(option);
    });
    if (tasks.some((task) => String(task.id) === current)) $("task").value = current;
  }

  function modelAvailable() {
    if (!state.status?.ready) return false;
    const model = state.status.models?.[state.condition];
    return model ? model.available !== false : true;
  }

  function updateControls() {
    $("predict-button").disabled = state.busy || state.imageLoading ||
      !state.image || !modelAvailable();
    $("predict-label").textContent = state.busy ? "正在推理…" : "开始推理";
    $("busy-overlay").hidden = !state.busy;
    $("predict-form").setAttribute("aria-busy", String(state.busy));
    document.querySelectorAll("#predict-form input, #predict-form select, " +
      "#predict-form button:not(#predict-button)").forEach((item) => {
      item.disabled = state.busy;
    });
    $("upload-zone").style.pointerEvents = state.busy ? "none" : "";
    $("clear-image").disabled = state.busy;
    if (state.busy) $("run-hint").textContent = "首轮推理包含模型加载，请保持页面打开";
    else if (state.imageLoading) $("run-hint").textContent = "正在读取图片，请稍候";
    else if (!state.status) $("run-hint").textContent = "正在连接推理服务";
    else if (!modelAvailable()) $("run-hint").textContent =
      state.status.error || `${state.condition} 模型暂不可用，请检查服务环境`;
    else if (!state.image) $("run-hint").textContent = "上传图片或选择示例后开始";
    else $("run-hint").textContent = `${state.condition} 预训练模型 · ${state.status.device || "自动选择设备"}`;
  }

  async function loadStatus() {
    try {
      state.status = await requestJson("/api/status");
      if (Array.isArray(state.status.tasks) && state.status.tasks.length) {
        populateTasks(state.status.tasks);
      }
      const ready = Boolean(state.status.ready);
      $("service-status").className = `service-status ${ready ? "ready" : "error"}`;
      $("service-label").textContent = ready ?
        `服务就绪 · ${state.status.device || "自动"}` : "推理环境未就绪";
      if (!ready && state.status.error) message(state.status.error);
      updateMaxSteps();
      await loadExamples();
    } catch (error) {
      state.status = { ready: false, error: error.message };
      $("service-status").className = "service-status error";
      $("service-label").textContent = "服务连接失败";
      message(`无法连接推理服务：${error.message}`);
    }
    updateControls();
  }

  function updateMaxSteps() {
    const limit = state.status?.max_steps?.[state.condition] ||
      (state.condition === "TP" ? 6 : 10);
    $("max-steps").max = String(limit);
    $("max-steps").value = String(limit);
  }

  function stopPlayback() {
    clearInterval(state.timer);
    state.timer = null;
    $("play-button").textContent = "▶";
    $("play-button").setAttribute("aria-label", "播放注视轨迹");
  }

  function clearResult() {
    stopPlayback();
    state.result = null;
    state.heatmaps = [];
    state.heatmapCards = [];
    $("heatmap-grid").replaceChildren();
    $("step-heatmaps").hidden = true;
    $("heatmap-step-count").textContent = "";
    state.visibleCount = 1;
    $("step-slider").disabled = true;
    $("step-slider").max = "1";
    $("step-slider").value = "1";
    $("play-button").disabled = true;
    $("step-value").textContent = "— / —";
    $("step-label").textContent = "轨迹回放";
    $("playback-hint").textContent = "推理后逐步查看";
    $("metric-fixations").replaceChildren(document.createTextNode("—"), unit("个"));
    $("metric-time").replaceChildren(document.createTextNode("—"), unit("秒"));
    $("metric-stop").textContent = "等待推理";
    $("result-detail").hidden = true;
    $("fixation-list").replaceChildren();
    $("export-png").disabled = true;
    $("export-json").disabled = true;
    $("result-note").textContent = "以图像中心为起点，预测后续注视位置。";
    $("viewer-tag").textContent = state.image ? "待推理" : "等待输入";
    $("canvas-caption").textContent = "输入图像";
    draw();
  }

  function unit(text) {
    const small = document.createElement("small");
    small.textContent = text;
    return small;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("无法读取图片，请换一张图片重试"));
      image.src = src;
    });
  }

  function setImage(image, name) {
    state.image = image;
    const maxCanvasSize = 2048;
    const scale = Math.min(1, maxCanvasSize / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = $("result-canvas");
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    canvas.hidden = false;
    $("empty-state").hidden = true;
    $("stage-caption").hidden = false;
    $("image-dimensions").textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
    $("file-name").textContent = name;
    $("file-summary").hidden = false;
    clearResult();
    updateControls();
  }

  async function uploadFile(file) {
    if (state.busy || !file) return;
    message();
    if (file.size > maxFileBytes) {
      message("图片超过 10 MB，请压缩后重试。"); return;
    }
    if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
      message("请选择 JPG、PNG 或 WebP 图片。"); return;
    }
    const requestId = ++state.imageRequest;
    state.imageLoading = true;
    updateControls();
    try {
      const data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("文件读取失败"));
        reader.readAsDataURL(file);
      });
      const image = await loadImage(data);
      if (requestId !== state.imageRequest) return;
      state.imageData = data;
      state.exampleId = null;
      markExample();
      setImage(image, file.name);
    } catch (error) { if (requestId === state.imageRequest) message(error.message); }
    finally {
      if (requestId === state.imageRequest) {
        state.imageLoading = false;
        updateControls();
      }
    }
  }

  function markExample() {
    document.querySelectorAll(".example-button").forEach((button) => {
      const selected = button.dataset.id === state.exampleId;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  async function selectExample(example) {
    if (state.busy || example.condition !== state.condition ||
        example.task !== $("task").value) return;
    const requestId = ++state.imageRequest;
    state.imageLoading = true;
    updateControls();
    message();
    try {
      const image = await loadImage(example.url);
      if (requestId !== state.imageRequest) return;
      state.exampleId = String(example.id);
      state.imageData = null;
      markExample();
      setImage(image, example.label || "数据集示例");
    } catch (error) { if (requestId === state.imageRequest) message(error.message); }
    finally {
      if (requestId === state.imageRequest) {
        state.imageLoading = false;
        updateControls();
      }
    }
  }

  async function loadExamples(nextPage = false) {
    const requestId = ++state.exampleRequest;
    $("examples").replaceChildren();
    const query = new URLSearchParams({
      condition: state.condition, task: $("task").value
    });
    $("examples-note").hidden = false;
    $("examples-note").textContent = "正在读取示例…";
    try {
      const payload = await requestJson(`/api/examples?${query}`);
      if (requestId !== state.exampleRequest) return;
      state.examples = Array.isArray(payload.examples) ? payload.examples : [];
      const length = state.examples.length;
      state.exampleOffset = nextPage && length ? (state.exampleOffset + 3) % length : 0;
      $("examples").replaceChildren();
      const shown = state.examples.slice(state.exampleOffset, state.exampleOffset + 3);
      shown.forEach((example) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "example-button";
        button.dataset.id = String(example.id);
        button.title = example.label || "选择示例";
        button.setAttribute("aria-label", `使用示例：${example.label || example.id}`);
        button.setAttribute("aria-pressed", "false");
        const image = document.createElement("img");
        image.src = example.url;
        image.alt = example.label || "数据集示例";
        image.loading = "lazy";
        button.append(image);
        button.addEventListener("click", () => selectExample(example));
        $("examples").append(button);
      });
      $("examples-note").hidden = Boolean(length);
      $("examples-note").textContent = "当前任务暂无可用示例，可直接上传图片。";
      $("refresh-examples").hidden = length <= 3;
      markExample();
    } catch (error) {
      if (requestId !== state.exampleRequest) return;
      $("examples").replaceChildren();
      $("examples-note").textContent = "示例暂时不可用，可直接上传图片。";
      $("refresh-examples").hidden = true;
    }
  }

  function draw() {
    if (!state.image) return;
    const canvas = $("result-canvas");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(state.image, 0, 0, width, height);
    if (!state.result) return;
    const visible = state.result.fixations.slice(0, state.visibleCount);
    const modelWidth = state.result.image?.model_width || 512;
    const modelHeight = state.result.image?.model_height || 320;
    const heatmap = state.heatmaps[state.visibleCount - 1];
    if ($("show-heatmap").checked && heatmap) {
      ctx.save();
      ctx.globalAlpha = heatmapOpacity();
      ctx.drawImage(heatmap, 0, 0, width, height);
      ctx.restore();
    }
    if (!$("show-path").checked || !visible.length) return;
    const scale = Math.max(width, height) / 900;
    const radius = Math.max(7, 11 * scale);
    const points = visible.map((point) => ({
      x: point.x / modelWidth * width, y: point.y / modelHeight * height
    }));
    ctx.save();
    ctx.lineWidth = Math.max(2, 3 * scale);
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(255, 255, 255, .9)";
    ctx.shadowColor = "rgba(0,0,0,.45)";
    ctx.shadowBlur = 4 * scale;
    points.forEach((point, index) => {
      if (!index) return;
      const previous = points[index - 1];
      const angle = Math.atan2(point.y - previous.y, point.x - previous.x);
      const length = Math.hypot(point.y - previous.y, point.x - previous.x);
      if (length < radius * 2) return;
      const endX = point.x - Math.cos(angle) * (radius + 3 * scale);
      const endY = point.y - Math.sin(angle) * (radius + 3 * scale);
      ctx.beginPath();
      ctx.moveTo(previous.x, previous.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.beginPath();
      const head = 8 * scale;
      ctx.moveTo(endX - head * Math.cos(angle - .45), endY - head * Math.sin(angle - .45));
      ctx.lineTo(endX, endY);
      ctx.lineTo(endX - head * Math.cos(angle + .45), endY - head * Math.sin(angle + .45));
      ctx.stroke();
    });
    points.forEach((point, index) => {
      const active = index === points.length - 1;
      ctx.beginPath();
      ctx.arc(point.x, point.y, active ? radius * 1.15 : radius, 0, Math.PI * 2);
      ctx.fillStyle = index === 0 ? "#f6d06e" : active ? "#166f62" : "#f5fffb";
      ctx.fill();
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = active ? "white" : "#166f62";
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = active ? "white" : "#18594e";
      ctx.font = `600 ${Math.max(10, 12 * scale)}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(index + 1), point.x, point.y + .5 * scale);
    });
    ctx.restore();
  }

  function heatmapOpacity() {
    return Number($("heatmap-opacity").value) / 100;
  }

  function drawHeatmapStep(canvas, index) {
    if (!state.result || !state.image) return;
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);
    context.drawImage(state.image, 0, 0, width, height);
    const heatmap = state.heatmaps[index];
    if (heatmap) {
      context.save();
      context.globalAlpha = heatmapOpacity();
      context.drawImage(heatmap, 0, 0, width, height);
      context.restore();
    }
    // 第 i 张热图以前 i + 1 个注视点为历史，候选点尚未加入历史。
    const history = state.result.fixations.slice(0, index + 1);
    const modelWidth = state.result.image?.model_width || 512;
    const modelHeight = state.result.image?.model_height || 320;
    const position = (point) => ({
      x: point.x / modelWidth * width,
      y: point.y / modelHeight * height
    });
    const points = history.map(position);
    const scale = Math.max(width, height) / 640;
    const radius = Math.max(3, 5 * scale);
    context.save();
    context.lineWidth = Math.max(1.5, 2 * scale);
    context.strokeStyle = "rgba(255,255,255,.95)";
    context.shadowColor = "rgba(0,0,0,.6)";
    context.shadowBlur = 3 * scale;
    if (points.length) {
      context.beginPath();
      points.forEach((point, pointIndex) => {
        if (pointIndex === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.stroke();
    }
    points.forEach((point, pointIndex) => {
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = pointIndex === 0 ? "#f6d06e" : "white";
      context.fill();
      context.strokeStyle = "#18594e";
      context.stroke();
    });
    const step = state.result.steps[index];
    if (Number.isFinite(step.x) && Number.isFinite(step.y)) {
      const next = position(step);
      const previous = points[points.length - 1];
      context.strokeStyle = "#b7ffe2";
      context.setLineDash([5 * scale, 4 * scale]);
      if (previous) {
        context.beginPath();
        context.moveTo(previous.x, previous.y);
        context.lineTo(next.x, next.y);
        context.stroke();
      }
      context.setLineDash([]);
      context.beginPath();
      context.arc(next.x, next.y, radius * 1.8, 0, Math.PI * 2);
      context.fillStyle = "#16796e";
      context.fill();
      context.strokeStyle = "white";
      context.stroke();
    }
    context.restore();
  }

  function renderStepHeatmaps() {
    const steps = Array.isArray(state.result?.steps) ? state.result.steps : [];
    $("heatmap-grid").replaceChildren();
    state.heatmapCards = [];
    $("step-heatmaps").hidden = steps.length === 0;
    $("heatmap-step-count").textContent = `共 ${steps.length} 步`;
    steps.forEach((step, index) => {
      const card = document.createElement("div");
      card.className = "heatmap-card";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "heatmap-select";
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `查看第 ${index + 1} 步热图`);
      const canvas = document.createElement("canvas");
      canvas.className = "heatmap-canvas";
      canvas.setAttribute("aria-label", `第 ${index + 1} 步预测热图`);
      const source = $("result-canvas");
      const scale = Math.min(1, 640 / Math.max(source.width, source.height));
      canvas.width = Math.max(1, Math.round(source.width * scale));
      canvas.height = Math.max(1, Math.round(source.height * scale));
      const title = document.createElement("span");
      title.className = "heatmap-card-title";
      const hasNext = Number.isFinite(step.x) && Number.isFinite(step.y);
      const stopped = !hasNext && state.result.meta?.stop_reason === "model_stop";
      title.textContent = `第 ${index + 1} 步${stopped ? " · 最终停止" : ""}`;
      const info = document.createElement("span");
      info.className = "heatmap-card-info";
      const probability = Number.isFinite(step.stop_probability) ?
        `终止概率 ${(step.stop_probability * 100).toFixed(1)}%` : "终止概率未提供";
      const destination = hasNext ?
        `下一点 (${Math.round(step.x)}, ${Math.round(step.y)})` :
        stopped ? "模型主动终止" : "未产生下一注视点";
      info.textContent = `${probability} · ${destination}`;
      button.append(canvas, title, info);
      button.addEventListener("click", () => {
        if (!state.result || state.heatmapCards[index]?.button !== button) return;
        stopPlayback();
        state.visibleCount = index + 1;
        $("show-heatmap").checked = true;
        updatePlayback();
      });
      const download = document.createElement("button");
      download.type = "button";
      download.className = "heatmap-download";
      download.textContent = "下载此步热图 ↓";
      download.setAttribute("aria-label", `下载第 ${index + 1} 步热图`);
      download.disabled = !state.heatmaps[index];
      download.addEventListener("click", () => {
        if (!state.result || state.heatmapCards[index]?.button !== button ||
            !state.heatmaps[index]) return;
        const output = document.createElement("canvas");
        output.width = $("result-canvas").width;
        output.height = $("result-canvas").height;
        drawHeatmapStep(output, index);
        const condition = state.result.meta?.condition || state.condition;
        const filename = `gldas-${condition.toLowerCase()}-step-` +
          `${String(index + 1).padStart(2, "0")}.png`;
        output.toBlob((blob) => {
          if (blob) saveBlob(blob, "png", filename);
        }, "image/png");
      });
      card.append(button, download);
      if (!state.heatmaps[index]) {
        const error = document.createElement("span");
        error.className = "heatmap-card-error";
        error.textContent = "此步热图无法读取";
        card.append(error);
      }
      $("heatmap-grid").append(card);
      state.heatmapCards.push({ card, button, canvas });
      drawHeatmapStep(canvas, index);
    });
  }

  function updatePlayback() {
    if (!state.result) return;
    const total = state.result.fixations.length;
    $("step-slider").value = String(state.visibleCount);
    $("step-value").textContent = `${state.visibleCount} / ${total}`;
    $("step-label").textContent = state.visibleCount === 1 ? "中心起点" :
      `第 ${state.visibleCount - 1} 次预测`;
    $("canvas-caption").textContent = `注视点 ${state.visibleCount} / ${total}`;
    const probability = state.result.fixations[state.visibleCount - 1]?.stop_probability;
    $("playback-hint").textContent = Number.isFinite(probability) ?
      `终止概率 ${(probability * 100).toFixed(1)}%` : "此步尚未评估终止概率";
    [...$("fixation-list").children].forEach((item, index) => {
      item.classList.toggle("active", index === state.visibleCount - 1);
    });
    state.heatmapCards.forEach(({ card, button }, index) => {
      const selected = index === state.visibleCount - 1;
      card.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    draw();
  }

  async function showResult(result) {
    if (!Array.isArray(result.fixations) || !result.fixations.length) {
      throw new Error("模型未返回有效注视点，请检查推理服务。");
    }
    const processedImage = result.image?.data_url ?
      await loadImage(result.image.data_url) : state.image;
    const sources = Array.isArray(result.steps) ?
      result.steps.map((step, index) => step.heatmap || result.heatmaps?.[index]) :
      result.heatmaps || [];
    state.heatmaps = await Promise.all(sources.map(async (source) => {
      if (!source) return null;
      try { return await loadImage(typeof source === "string" ? source : source.data_url); }
      catch (_) { return null; }
    }));
    state.result = result;
    state.image = processedImage;
    if (result.meta?.device && state.status) {
      state.status.device = result.meta.device;
      $("service-label").textContent = `服务就绪 · ${result.meta.device}`;
    }
    const total = result.fixations.length;
    state.visibleCount = total;
    $("step-slider").max = String(total);
    $("step-slider").disabled = total < 2;
    $("play-button").disabled = total < 2;
    $("metric-fixations").replaceChildren(document.createTextNode(String(total)), unit("个"));
    const seconds = (Number(result.meta?.elapsed_ms || 0) / 1000).toFixed(2);
    $("metric-time").replaceChildren(document.createTextNode(seconds), unit("秒"));
    $("metric-stop").textContent = result.meta?.stop_reason === "model_stop" ?
      "模型主动终止" : result.meta?.stop_reason === "max_steps" ? "达到步数上限" : "预测完成";
    $("viewer-tag").textContent = `${result.meta?.condition || state.condition} · 推理完成`;
    $("fixation-list").replaceChildren();
    result.fixations.forEach((point, index) => {
      const li = document.createElement("li");
      const number = document.createElement("span");
      number.textContent = index ? String(index + 1).padStart(2, "0") : "起点";
      const coordinate = document.createElement("span");
      coordinate.textContent = `(${Math.round(point.x)}, ${Math.round(point.y)})`;
      li.append(number, coordinate);
      $("fixation-list").append(li);
    });
    $("result-detail").hidden = false;
    $("export-png").disabled = false;
    $("export-json").disabled = false;
    $("result-note").textContent = "热图表示当前注视历史下的下一步预测分布。";
    renderStepHeatmaps();
    updatePlayback();
  }

  async function predict(event) {
    event.preventDefault();
    if (state.busy || state.imageLoading || !state.image || !modelAvailable()) return;
    if (!$("predict-form").reportValidity()) return;
    message();
    clearResult();
    state.busy = true;
    updateControls();
    const started = performance.now();
    state.elapsedTimer = setInterval(() => {
      const elapsed = Math.floor((performance.now() - started) / 1000);
      $("busy-detail").textContent = elapsed < 8 ?
        "首次推理需要加载权重，请稍候" : `已等待 ${elapsed} 秒 · 请保持页面打开`;
    }, 1000);
    const body = {
      task: $("task").value, condition: state.condition,
      decoding: $("decoding").value, max_steps: Number($("max-steps").value),
      stop_threshold: Number($("stop-threshold").value), seed: Number($("seed").value)
    };
    if (state.exampleId) body.example_id = state.exampleId;
    else body.image = state.imageData;
    try {
      const result = await requestJson("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      await showResult(result);
    } catch (error) {
      message(`推理未完成：${error.message}`);
      $("viewer-tag").textContent = "推理失败";
    } finally {
      clearInterval(state.elapsedTimer);
      state.busy = false;
      updateControls();
    }
  }

  function saveBlob(blob, extension, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const condition = state.result?.meta?.condition || state.condition;
    link.download = filename ||
      `gldas-${condition.toLowerCase()}-${Date.now()}.${extension}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  $("predict-form").addEventListener("submit", predict);
  $("image-file").addEventListener("change", (event) => {
    uploadFile(event.target.files[0]);
    event.target.value = "";
  });
  $("upload-zone").addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!state.busy) $("upload-zone").classList.add("dragover");
  });
  $("upload-zone").addEventListener("dragleave", () => $("upload-zone").classList.remove("dragover"));
  $("upload-zone").addEventListener("drop", (event) => {
    event.preventDefault();
    $("upload-zone").classList.remove("dragover");
    uploadFile(event.dataTransfer.files[0]);
  });
  $("clear-image").addEventListener("click", () => {
    if (state.busy) return;
    ++state.imageRequest;
    state.imageLoading = false;
    state.image = null;
    state.imageData = null;
    state.exampleId = null;
    clearResult();
    markExample();
    $("result-canvas").hidden = true;
    $("empty-state").hidden = false;
    $("stage-caption").hidden = true;
    $("file-summary").hidden = true;
    updateControls();
  });
  document.querySelectorAll("[data-condition]").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.busy || state.condition === button.dataset.condition) return;
      ++state.imageRequest;
      state.imageLoading = false;
      state.condition = button.dataset.condition;
      document.querySelectorAll("[data-condition]").forEach((item) => {
        const selected = item.dataset.condition === state.condition;
        item.classList.toggle("selected", selected);
        item.setAttribute("aria-pressed", String(selected));
      });
      if (state.exampleId) $("clear-image").click();
      else clearResult();
      message();
      updateMaxSteps();
      updateControls();
      loadExamples();
    });
  });
  $("task").addEventListener("change", () => {
    if (state.busy) return;
    ++state.imageRequest;
    state.imageLoading = false;
    if (state.exampleId) $("clear-image").click();
    else clearResult();
    updateControls();
    loadExamples();
  });
  ["decoding", "max-steps", "stop-threshold", "seed"].forEach((id) => {
    $(id).addEventListener("change", () => {
      if (!state.busy) clearResult();
    });
  });
  $("refresh-examples").addEventListener("click", () => {
    if (!state.busy) loadExamples(true);
  });
  $("show-path").addEventListener("change", draw);
  $("show-heatmap").addEventListener("change", draw);
  $("heatmap-opacity").addEventListener("input", () => {
    $("heatmap-opacity-value").textContent = `${$("heatmap-opacity").value}%`;
    state.heatmapCards.forEach(({ canvas }, index) => {
      drawHeatmapStep(canvas, index);
    });
    draw();
  });
  $("step-slider").addEventListener("input", () => {
    stopPlayback();
    state.visibleCount = Number($("step-slider").value);
    updatePlayback();
  });
  $("play-button").addEventListener("click", () => {
    if (!state.result) return;
    if (state.timer) { stopPlayback(); return; }
    if (state.visibleCount >= state.result.fixations.length) state.visibleCount = 1;
    updatePlayback();
    $("play-button").textContent = "Ⅱ";
    $("play-button").setAttribute("aria-label", "暂停注视轨迹");
    state.timer = setInterval(() => {
      if (state.visibleCount < state.result.fixations.length) {
        state.visibleCount += 1;
        updatePlayback();
      }
      if (state.visibleCount >= state.result.fixations.length) stopPlayback();
    }, 750);
  });
  $("export-png").addEventListener("click", () => {
    if (!state.result) return;
    $("result-canvas").toBlob((blob) => { if (blob) saveBlob(blob, "png"); }, "image/png");
  });
  $("export-json").addEventListener("click", () => {
    if (!state.result) return;
    const { image, heatmaps, heatmap, steps, ...data } = state.result;
    const imageInfo = image ? { ...image } : {};
    delete imageInfo.data_url;
    const stepInfo = (steps || []).map(({ heatmap, ...step }) => step);
    const content = { ...data, image: imageInfo, steps: stepInfo };
    saveBlob(new Blob([JSON.stringify(content, null, 2)], {
      type: "application/json;charset=utf-8"
    }), "json");
  });
  populateTasks(defaultTasks);
  loadStatus();
})();
