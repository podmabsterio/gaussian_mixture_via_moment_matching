const state = {
  declarations: null,
  model: null,
  dataset: null,
  advanced: false,
  runs: [],
  sessionRuns: [],
  savedRuns: [],
  currentRunId: null,
  currentTimer: null,
  compareResults: null,
  comparisonStatistic: "mean_std",
  comparisonVersion: 0,
  preview: null,
  previewVersion: 0,
  quickModel: null,
  quickDataset: null,
  quickRun: null,
  quickRunId: null,
  quickData: null,
  quickSnapshots: [],
  quickSelectedIndex: 0,
  quickFollowLatest: true,
  quickEventSource: null,
  quickLastEvent: 0,
};

const plotColors = ["#21745f", "#3b6f9d", "#bd7420", "#9a4f78", "#678441", "#725ca8", "#b84c45", "#27818a"];
const outlierColor = "#6f7774";

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function toast(message, error = false) {
  const element = $("toast");
  element.textContent = message;
  element.className = `toast visible${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.className = "toast"; }, 4200);
}

function entityById(items, id) {
  return items.find((item) => item.id === id);
}

function inputValue(value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function applyNumericAttributes(input, spec) {
  if (spec.minimum !== undefined) input.min = spec.minimum;
  if (spec.maximum !== undefined) input.max = spec.maximum;
  if (spec.exclusive_minimum !== undefined) input.min = spec.exclusive_minimum;
  if (spec.exclusive_maximum !== undefined) input.max = spec.exclusive_maximum;
  input.step = spec.type === "integer" ? "1" : String(spec.step ?? "any");
}

function makeControl(spec, value = spec.default, readOnly = false) {
  let control;
  if (spec.type === "boolean") {
    control = document.createElement("label");
    control.className = "checkbox-control";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = value === true;
    input.disabled = readOnly;
    input.dataset.key = spec.key;
    input.dataset.type = spec.type;
    const text = document.createElement("span");
    text.textContent = value === true ? "Enabled" : "Disabled";
    if (readOnly) control.classList.add("readonly-control");
    control.append(input, text);
    return control;
  }
  if (spec.type === "enum") {
    control = document.createElement("select");
    let hasValue = false;
    for (const optionSpec of spec.options) {
      const option = document.createElement("option");
      option.value = JSON.stringify(optionSpec.value);
      option.textContent = optionSpec.display_name;
      option.title = optionSpec.description || "";
      option.selected = Object.is(optionSpec.value, value);
      hasValue ||= option.selected;
      control.appendChild(option);
    }
    if (!hasValue) {
      const option = document.createElement("option");
      option.value = JSON.stringify(value);
      option.textContent = inputValue(value) || "auto";
      option.selected = true;
      control.appendChild(option);
    }
    control.disabled = readOnly;
  } else {
    control = document.createElement("input");
    if (spec.type === "integer" || spec.type === "number") {
      control.type = "number";
      applyNumericAttributes(control, spec);
    } else {
      control.type = "text";
      if (spec.maximum_length) control.maxLength = spec.maximum_length;
    }
    control.value = inputValue(value);
    control.readOnly = readOnly;
    if (spec.nullable) control.placeholder = "auto";
    if (spec.type.endsWith("_array")) control.placeholder = spec.nullable ? "auto or comma-separated values" : "comma-separated values";
  }
  control.dataset.key = spec.key;
  control.dataset.type = spec.type;
  if (readOnly) control.classList.add("readonly-control");
  return control;
}

function renderParameters(container, specs, options = {}) {
  const values = options.values || {};
  const readOnly = options.readOnly === true;
  container.innerHTML = "";
  container.classList.toggle("readonly-parameters", readOnly);
  for (const spec of specs) {
    if (spec.hidden) continue;
    const field = document.createElement("div");
    field.className = `field${spec.advanced ? " advanced-field" : ""}`;
    const label = document.createElement("label");
    label.textContent = spec.display_name;
    if (spec.advanced) {
      const chip = document.createElement("span");
      chip.className = "advanced-chip";
      chip.textContent = "advanced";
      label.appendChild(chip);
    }
    const value = Object.prototype.hasOwnProperty.call(values, spec.key) ? values[spec.key] : spec.default;
    const control = makeControl(spec, value, readOnly);
    const input = control.matches?.("input,select") ? control : control.querySelector("input,select");
    if (!readOnly) {
      input.addEventListener("change", () => {
        if (input.value !== "" || input.type === "checkbox") {
          for (const conflictingKey of spec.conflicts_with || []) {
            const other = container.querySelector(`[data-key="${conflictingKey}"]`);
            if (!other) continue;
            if (other.type === "checkbox") other.checked = false;
            else other.value = "";
          }
        }
        if (container.id === "datasetParameters" || spec.key === "dataset_seed_start") invalidateDataPreview();
        updateWorkload();
        options.onChange?.();
      });
      input.addEventListener("input", () => {
        if (container.id === "datasetParameters" || spec.key === "dataset_seed_start") invalidateDataPreview();
        updateWorkload();
        options.onChange?.();
      });
    }
    const help = document.createElement("p");
    help.className = "field-help";
    help.textContent = spec.description;
    field.append(label, control, help);
    container.appendChild(field);
  }
}

function readParameters(container, specs) {
  const values = {};
  for (const spec of specs) {
    if (spec.hidden) continue;
    const input = container.querySelector(`[data-key="${spec.key}"]`);
    if (!input) continue;
    let value;
    if (spec.type === "boolean") {
      value = input.checked;
    } else if (spec.type === "enum") {
      value = JSON.parse(input.value);
    } else if (input.value.trim() === "" && spec.nullable) {
      value = null;
    } else if (spec.type === "integer") {
      value = Number.parseInt(input.value, 10);
    } else if (spec.type === "number") {
      value = Number.parseFloat(input.value);
    } else if (spec.type === "integer_array" || spec.type === "number_array") {
      value = input.value.split(",").map((item) => item.trim()).filter(Boolean).map((item) => (
        spec.type === "integer_array" ? Number.parseInt(item, 10) : Number.parseFloat(item)
      ));
    } else {
      value = input.value;
    }
    values[spec.key] = value;
  }
  return values;
}

function describeEntity(element, entity) {
  element.textContent = entity.description;
  const target = document.createElement("code");
  target.textContent = entity.target;
  element.appendChild(target);
}

function selectDataset(id) {
  invalidateDataPreview();
  state.dataset = entityById(state.declarations.datasets, id);
  $("datasetInstanceName").value = state.dataset.default_instance_name;
  describeEntity($("datasetDescription"), state.dataset);
  renderParameters($("datasetParameters"), state.dataset.parameters);
  updateWorkload();
}

function selectModel(id) {
  state.model = entityById(state.declarations.models, id);
  $("modelInstanceName").value = state.model.default_instance_name;
  describeEntity($("modelDescription"), state.model);
  renderParameters($("modelParameters"), state.model.parameters);
  updateWorkload();
}

function selectQuickDataset(id) {
  state.quickDataset = entityById(state.declarations.datasets, id);
  $("quickDatasetInstanceName").value = state.quickDataset.default_instance_name;
  describeEntity($("quickDatasetDescription"), state.quickDataset);
  renderParameters($("quickDatasetParameters"), state.quickDataset.parameters);
}

function selectQuickModel(id) {
  state.quickModel = entityById(state.declarations.models, id);
  $("quickModelInstanceName").value = state.quickModel.default_instance_name;
  describeEntity($("quickModelDescription"), state.quickModel);
  renderParameters($("quickModelParameters"), state.quickModel.parameters);
}

function populateEntitySelect(select, items) {
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.display_name;
    select.appendChild(option);
  }
}

function executionValue(key) {
  const spec = state.declarations.execution.parameters.find((item) => item.key === key);
  const input = $("executionParameters").querySelector(`[data-key="${key}"]`);
  if (!spec || !input) return spec?.default;
  return readParameters($("executionParameters"), [spec])[key];
}

function updateWorkload() {
  if (!state.declarations) return;
  const datasets = Number(executionValue("num_datasets")) || 0;
  const seeds = Number(executionValue("model_seeds_per_dataset")) || 0;
  $("workloadBadge").querySelector("strong").textContent = (datasets * seeds).toLocaleString();
}

function setAdvanced(enabled) {
  state.advanced = enabled;
  document.body.classList.toggle("show-advanced", enabled);
  $("advancedToggle").setAttribute("aria-pressed", String(enabled));
}

function buildRequest() {
  return {
    models: [{
      declaration_id: state.model.id,
      instance_name: $("modelInstanceName").value,
      parameters: readParameters($("modelParameters"), state.model.parameters),
    }],
    datasets: [{
      ...buildDatasetRequest(),
      instance_name: $("datasetInstanceName").value,
    }],
    execution: readParameters($("executionParameters"), state.declarations.execution.parameters),
  };
}

function buildDatasetRequest() {
  return {
    declaration_id: state.dataset.id,
    parameters: readParameters($("datasetParameters"), state.dataset.parameters),
  };
}

function buildQuickRequest() {
  return {
    model: {
      declaration_id: state.quickModel.id,
      instance_name: $("quickModelInstanceName").value,
      parameters: readParameters($("quickModelParameters"), state.quickModel.parameters),
    },
    dataset: {
      declaration_id: state.quickDataset.id,
      instance_name: $("quickDatasetInstanceName").value,
      parameters: readParameters($("quickDatasetParameters"), state.quickDataset.parameters),
    },
    execution: readParameters(
      $("quickExecutionParameters"),
      state.declarations.execution.quick_parameters,
    ),
  };
}

function invalidateDataPreview() {
  state.previewVersion += 1;
  state.preview = null;
  const preview = $("dataPreview");
  if (preview) preview.hidden = true;
}

function pointColor(label) {
  return label < 0 ? outlierColor : plotColors[label % plotColors.length];
}

function previewBounds(preview) {
  const xs = preview.points.map((point) => point.x);
  const ys = preview.points.map((point) => point.y);
  for (const component of preview.components) {
    const cosine = Math.cos(component.angle);
    const sine = Math.sin(component.angle);
    const xRadius = Math.hypot(component.radii[0] * cosine, component.radii[1] * sine);
    const yRadius = Math.hypot(component.radii[0] * sine, component.radii[1] * cosine);
    xs.push(component.center[0] - xRadius, component.center[0] + xRadius);
    ys.push(component.center[1] - yRadius, component.center[1] + yRadius);
  }
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  if (maxX - minX < 1e-12) { minX -= 1; maxX += 1; }
  if (maxY - minY < 1e-12) { minY -= 1; maxY += 1; }
  return {minX, maxX, minY, maxY};
}

function drawDataPreview() {
  const preview = state.preview;
  if (!preview?.points?.length) return;
  const canvas = $("previewCanvas");
  const width = canvas.parentElement.clientWidth;
  const height = Math.min(560, Math.max(360, width * 0.52));
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  canvas.style.height = `${height}px`;

  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const bounds = previewBounds(preview);
  const padding = 42;
  const spanX = bounds.maxX - bounds.minX;
  const spanY = bounds.maxY - bounds.minY;
  const scale = Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY);
  const middleX = (bounds.minX + bounds.maxX) / 2;
  const middleY = (bounds.minY + bounds.maxY) / 2;
  const screenX = (value) => width / 2 + (value - middleX) * scale;
  const screenY = (value) => height / 2 - (value - middleY) * scale;

  context.strokeStyle = "rgba(35, 57, 50, .10)";
  context.lineWidth = 1;
  if (bounds.minX <= 0 && bounds.maxX >= 0) {
    context.beginPath(); context.moveTo(screenX(0), padding / 2); context.lineTo(screenX(0), height - padding / 2); context.stroke();
  }
  if (bounds.minY <= 0 && bounds.maxY >= 0) {
    context.beginPath(); context.moveTo(padding / 2, screenY(0)); context.lineTo(width - padding / 2, screenY(0)); context.stroke();
  }

  for (const point of preview.points) {
    context.beginPath();
    context.arc(screenX(point.x), screenY(point.y), point.label < 0 ? 2.1 : 1.8, 0, Math.PI * 2);
    context.fillStyle = pointColor(point.label);
    context.globalAlpha = point.label < 0 ? 0.45 : 0.52;
    context.fill();
  }
  context.globalAlpha = 1;

  for (const component of preview.components) {
    const color = pointColor(component.index);
    const centerX = screenX(component.center[0]);
    const centerY = screenY(component.center[1]);
    context.save();
    context.translate(centerX, centerY);
    context.rotate(-component.angle);
    context.beginPath();
    context.ellipse(0, 0, component.radii[0] * scale, component.radii[1] * scale, 0, 0, Math.PI * 2);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.stroke();
    context.restore();

    context.beginPath();
    context.moveTo(centerX - 6, centerY - 6); context.lineTo(centerX + 6, centerY + 6);
    context.moveTo(centerX + 6, centerY - 6); context.lineTo(centerX - 6, centerY + 6);
    context.strokeStyle = color;
    context.lineWidth = 2.4;
    context.stroke();
  }

  context.fillStyle = "#6b7672";
  context.font = '11px "SFMono-Regular", Consolas, monospace';
  context.fillText("PC1", width - padding + 8, height / 2 + 4);
  context.save();
  context.translate(width / 2 - 4, padding - 9);
  context.rotate(-Math.PI / 2);
  context.fillText("PC2", 0, 0);
  context.restore();
}

function renderDataPreview(preview) {
  state.preview = preview;
  $("dataPreview").hidden = false;
  $("previewTitle").textContent = `${preview.dataset.display_name} · PCA projection`;
  $("previewSeed").textContent = `seed ${preview.seed}`;
  const variance = preview.explained_variance_ratio.map((value) => `${(value * 100).toFixed(1)}%`);
  const pointSummary = preview.displayed_points === preview.total_points
    ? `${preview.total_points.toLocaleString()} points`
    : `${preview.displayed_points.toLocaleString()} of ${preview.total_points.toLocaleString()} points shown`;
  $("previewMeta").textContent = `${pointSummary} · ${preview.n_features} dimensions · PCA variance ${variance[0]} + ${variance[1]}`;

  const hasOutliers = preview.points.some((point) => point.label < 0);
  $("previewLegend").innerHTML = preview.components.map((component) => {
    const weight = component.weight === null ? "" : ` · ${(component.weight * 100).toFixed(1)}%`;
    return `<span><i style="background:${pointColor(component.index)}"></i>Component ${component.index + 1}${weight}</span>`;
  }).concat(hasOutliers ? [`<span><i style="background:${outlierColor}"></i>Outliers</span>`] : []).join("");
  requestAnimationFrame(drawDataPreview);
}

async function previewData() {
  const button = $("previewButton");
  const label = button.querySelector("span");
  button.disabled = true;
  label.textContent = "Generating…";
  const version = state.previewVersion;
  try {
    const preview = await api("/api/data-preview", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        dataset: buildDatasetRequest(),
        seed: executionValue("dataset_seed_start"),
      }),
    });
    if (version !== state.previewVersion) {
      toast("Parameters changed — generate the preview again");
      return;
    }
    renderDataPreview(preview);
    $("dataPreview").scrollIntoView({behavior: "smooth", block: "nearest"});
    toast("Data preview generated");
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    label.textContent = "Preview data";
  }
}

async function startRun() {
  const button = $("runButton");
  button.disabled = true;
  try {
    const run = await api("/api/runs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(buildRequest()),
    });
    state.currentRunId = run.id;
    $("liveRun").hidden = false;
    $("liveRun").scrollIntoView({behavior: "smooth", block: "nearest"});
    renderLive(run);
    pollCurrentRun();
    toast(`Run ${run.metadata.run_name} started`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

function renderLive(run) {
  $("liveRunName").textContent = run.metadata.run_name;
  const status = $("liveStatus");
  status.textContent = run.status;
  status.className = `status-pill status-${run.status}`;
  const started = run.started_at ? new Date(run.started_at * 1000).toLocaleString() : "waiting";
  $("liveMeta").textContent = `${run.id} · ${run.metadata.dataset_names.join(", ")} · ${run.metadata.model_names.join(", ")} · ${started}`;
  $("liveLog").textContent = run.log_tail || "Waiting for worker output…";
  $("liveLog").scrollTop = $("liveLog").scrollHeight;
  $("cancelButton").hidden = !["starting", "running", "cancelling"].includes(run.status);
}

function pollCurrentRun() {
  clearTimeout(state.currentTimer);
  const tick = async () => {
    if (!state.currentRunId) return;
    try {
      const run = await api(`/api/runs/${state.currentRunId}`);
      renderLive(run);
      await refreshRuns(false);
      if (["starting", "running", "cancelling"].includes(run.status)) {
        state.currentTimer = setTimeout(tick, 900);
      } else {
        toast(run.status === "completed" ? "Experiment completed" : `Experiment ${run.status}`, run.status === "error");
      }
    } catch (error) {
      toast(error.message, true);
    }
  };
  tick();
}

function formatDuration(run) {
  if (!run.started_at) return "—";
  const end = run.finished_at || Date.now() / 1000;
  return `${(end - run.started_at).toFixed(1)}s`;
}

function renderRuns() {
  const tbody = $("runsTable");
  if (!state.runs.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">No saved or active experiments were found.</td></tr>';
    return;
  }
  tbody.innerHTML = state.runs.map((run) => {
    const source = run.source || "session";
    const timestamp = run.started_at || run.saved_at || run.created_at;
    const identifier = source === "saved" ? "saved result" : run.id;
    return `
    <tr class="clickable-run" data-run-id="${escapeHtml(run.id)}" data-run-source="${source}" tabindex="0" aria-label="Open ${escapeHtml(run.metadata.run_name)}">
      <td><button class="run-link" data-run-id="${escapeHtml(run.id)}" data-run-source="${source}">${escapeHtml(run.metadata.run_name)}</button><br><span class="muted">${escapeHtml(identifier)}</span></td>
      <td><span class="status-pill status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></td>
      <td>${escapeHtml(run.metadata.dataset_names.join(", "))}</td>
      <td>${escapeHtml(run.metadata.model_names.join(", "))}</td>
      <td>${timestamp ? escapeHtml(new Date(timestamp * 1000).toLocaleString()) : "—"}</td>
      <td>${formatDuration(run)}</td>
      <td><button class="run-link" data-run-id="${escapeHtml(run.id)}" data-run-source="${source}">Open</button></td>
    </tr>`;
  }).join("");
}

function updateRunningCount() {
  const count = state.runs.filter((run) => ["starting", "running", "cancelling"].includes(run.status)).length + (quickIsActive() ? 1 : 0);
  $("runningCount").textContent = count ? `${count} active run${count === 1 ? "" : "s"}` : "No active runs";
}

function populateFilterSelect(id, values, allLabel) {
  const select = $(id);
  const previous = select.value;
  select.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>${values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  if (values.includes(previous)) select.value = previous;
}

function mergeRuns() {
  const livePaths = new Set(state.sessionRuns.map((run) => run.metadata?.result_path));
  state.runs = [
    ...state.sessionRuns.map((run) => ({...run, source: "session"})),
    ...state.savedRuns.filter((run) => !livePaths.has(run.metadata?.result_path)),
  ].sort((left, right) => {
    const leftTime = left.started_at || left.saved_at || left.created_at || 0;
    const rightTime = right.started_at || right.saved_at || right.created_at || 0;
    return rightTime - leftTime;
  });
}

async function refreshRuns(showError = true, includeSaved = false) {
  try {
    const requests = [api("/api/runs")];
    if (includeSaved) requests.push(api("/api/saved-runs"));
    const [sessionRuns, savedRuns] = await Promise.all(requests);
    state.sessionRuns = sessionRuns;
    if (includeSaved) state.savedRuns = savedRuns;
    mergeRuns();
    renderRuns();
    updateRunningCount();
  } catch (error) {
    if (showError) toast(error.message, true);
  }
}

function aggregateTable(rows) {
  if (!rows?.length) return '<p class="muted">No aggregate rows were saved.</p>';
  const columns = Object.keys(rows[0]);
  return `<div class="table-scroll"><table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((column) => `<td>${typeof row[column] === "number" ? escapeHtml(Number(row[column]).toPrecision(5)) : escapeHtml(row[column])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function configValue(config, path) {
  return path.split(".").reduce((value, key) => value?.[key], config);
}

function entityByTarget(items, target) {
  return items.find((item) => item.target === target);
}

function readonlyField(labelText, value, wide = false) {
  const field = document.createElement("div");
  field.className = `field${wide ? " wide" : ""}`;
  const label = document.createElement("label");
  label.textContent = labelText;
  const display = document.createElement("div");
  display.className = "readonly-value";
  display.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  field.append(label, display);
  return field;
}

function unknownParameters(container, target) {
  container.innerHTML = "";
  container.classList.add("readonly-parameters");
  for (const [key, value] of Object.entries(target || {})) {
    if (key === "_target_") continue;
    const field = document.createElement("div");
    field.className = "field";
    const label = document.createElement("label");
    label.textContent = key.replaceAll("_", " ");
    const display = document.createElement("div");
    display.className = "readonly-value readonly-raw-value";
    display.textContent = value === null ? "null" : (typeof value === "object" ? JSON.stringify(value) : String(value));
    const help = document.createElement("p");
    help.className = "field-help";
    help.textContent = "Not present in the current UI declaration.";
    field.append(label, display, help);
    container.appendChild(field);
  }
}

function readonlyEntityCard(kind, entry, index) {
  const isModel = kind === "model";
  const items = isModel ? state.declarations.models : state.declarations.datasets;
  const target = entry.target || {};
  const declaration = entityByTarget(items, target._target_);
  const instanceName = entry[isModel ? "model_name" : "name"] || `Unnamed ${kind}`;
  const card = document.createElement("article");
  card.className = "config-card readonly-config-card";
  card.dataset.accent = isModel ? "model" : "dataset";

  const heading = document.createElement("div");
  heading.className = "card-heading";
  const step = document.createElement("div");
  step.className = "step-number";
  step.textContent = `${isModel ? "M" : "D"}${index + 1}`;
  const headingText = document.createElement("div");
  const eyebrow = document.createElement("span");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = isModel ? "Estimator snapshot" : "Data source snapshot";
  const title = document.createElement("h3");
  title.textContent = declaration?.display_name || (isModel ? "Unrecognized model" : "Unrecognized dataset");
  headingText.append(eyebrow, title);
  heading.append(step, headingText);

  const selector = document.createElement("div");
  selector.className = "entity-selector-row";
  selector.append(
    readonlyField(isModel ? "Model target" : "Dataset generator", declaration?.display_name || target._target_, true),
    readonlyField("Instance name", instanceName),
  );

  const description = document.createElement("div");
  description.className = "entity-description";
  description.textContent = declaration?.description || "This target is not described by the current UI declarations.";
  const targetCode = document.createElement("code");
  targetCode.textContent = target._target_ || "Target not recorded";
  description.appendChild(targetCode);

  const parameters = document.createElement("div");
  parameters.className = "parameter-grid";
  if (declaration) {
    renderParameters(parameters, declaration.parameters, {values: target, readOnly: true});
  } else {
    unknownParameters(parameters, target);
  }
  card.append(heading, selector, description, parameters);
  return card;
}

function readonlyExecutionCard(run) {
  const quick = run.config?.quick_experiment === true;
  const specs = quick
    ? state.declarations.execution.quick_parameters
    : state.declarations.execution.parameters;
  const card = document.createElement("article");
  card.className = "config-card readonly-config-card";
  card.dataset.accent = "execution";
  card.innerHTML = `
    <div class="card-heading">
      <div class="step-number">E</div>
      <div><span class="eyebrow">${quick ? "Single-run snapshot" : "Resources &amp; output snapshot"}</span><h3>Execution</h3></div>
    </div>`;
  const parameters = document.createElement("div");
  parameters.className = "parameter-grid";
  const values = {};
  for (const spec of specs) {
    if (quick) {
      values[spec.key] = run.config.quick_execution?.[spec.key];
      continue;
    }
    if (spec.scope === "manager") {
      if (spec.key === "save_result") values[spec.key] = run.metadata.save_result;
      continue;
    }
    if (spec.path) values[spec.key] = configValue(run.config, spec.path);
  }
  renderParameters(parameters, specs, {values, readOnly: true});
  card.appendChild(parameters);
  return card;
}

function runConfiguration(run) {
  const wrapper = document.createElement("div");
  wrapper.className = "readonly-config-view";
  const notice = document.createElement("div");
  notice.className = "readonly-notice";
  notice.innerHTML = "<strong>Read-only configuration snapshot</strong><span>Values below are from the compiled Hydra config used by this worker. Use the global Advanced switch to reveal advanced fields.</span>";
  const stack = document.createElement("div");
  stack.className = "configuration-stack readonly-config-stack";
  for (const [index, dataset] of (run.config?.datasets || []).entries()) {
    stack.appendChild(readonlyEntityCard("dataset", dataset, index));
  }
  for (const [index, model] of (run.config?.models || []).entries()) {
    stack.appendChild(readonlyEntityCard("model", model, index));
  }
  stack.appendChild(readonlyExecutionCard(run));

  const raw = document.createElement("details");
  raw.className = "raw-config";
  const summary = document.createElement("summary");
  summary.textContent = "Raw compiled configuration";
  const pre = document.createElement("pre");
  pre.className = "log-view raw-config-view";
  pre.textContent = JSON.stringify(run.config, null, 2);
  raw.append(summary, pre);
  wrapper.append(notice, stack, raw);
  return wrapper;
}

function activateResultPanel(details, name) {
  details.querySelectorAll("[data-result-tab]").forEach((button) => {
    const active = button.dataset.resultTab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  details.querySelectorAll("[data-result-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.resultPanel !== name;
  });
}

async function showRunDetails(runId, source = "session") {
  try {
    let run;
    let result;
    if (source === "saved") {
      const saved = await api(`/api/saved-runs/${encodeURIComponent(runId)}`);
      run = saved.run;
      result = saved.results;
    } else {
      [run, result] = await Promise.all([api(`/api/runs/${runId}`), api(`/api/runs/${runId}/results`)]);
    }
    const details = $("resultDetails");
    const datasets = result.datasets.map((dataset) => `
      <div class="aggregate-block">
        <h4>${escapeHtml(dataset.name)} · mean</h4>
        ${aggregateTable(dataset.aggregates.mean)}
      </div>`).join("");
    details.innerHTML = `
      <div class="result-header">
        <div><span class="eyebrow">Run ${escapeHtml(run.id)}</span><h3>${escapeHtml(run.metadata.run_name)}</h3><div class="result-path">${escapeHtml(result.result_path)}</div></div>
        <div class="result-header-actions"><span class="status-pill status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span><button class="text-button" data-close-details type="button">Close</button></div>
      </div>
      <div class="result-tabs" role="tablist" aria-label="Run details">
        <button class="active" data-result-tab="configuration" role="tab" aria-selected="true" type="button">Configuration</button>
        <button data-result-tab="metrics" role="tab" aria-selected="false" type="button">Metrics</button>
        <button data-result-tab="log" role="tab" aria-selected="false" type="button">Worker log</button>
      </div>
      <section class="result-panel" data-result-panel="configuration"></section>
      <section class="result-panel" data-result-panel="metrics" hidden>${datasets || '<div class="empty-state">Aggregate results are not available yet.</div>'}</section>
      <section class="result-panel" data-result-panel="log" hidden><pre class="log-view result-log">${escapeHtml(run.log_tail || "No output")}</pre></section>`;
    details.querySelector('[data-result-panel="configuration"]').appendChild(runConfiguration(run));
    details.querySelectorAll("[data-result-tab]").forEach((button) => {
      button.addEventListener("click", () => activateResultPanel(details, button.dataset.resultTab));
    });
    details.querySelector("[data-close-details]").addEventListener("click", () => { details.hidden = true; });
    details.hidden = false;
    details.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadComparison() {
  const version = ++state.comparisonVersion;
  const parameters = new URLSearchParams();
  if ($("compareRun").value) parameters.set("run", $("compareRun").value);
  if ($("compareDataset").value) parameters.set("dataset", $("compareDataset").value);
  if ($("compareModel").value) parameters.set("model", $("compareModel").value);
  parameters.set("group_by", $("compareGroupBy").value);
  parameters.set("statistic", state.comparisonStatistic);
  $("refreshComparison").disabled = true;
  $("compareSummary").textContent = "Reading saved aggregate files…";
  try {
    const result = await api(`/api/comparison?${parameters.toString()}`);
    if (version !== state.comparisonVersion) return;
    state.compareResults = result;
    populateFilterSelect("compareRun", result.available.runs, "All runs");
    populateFilterSelect("compareDataset", result.available.datasets, "All datasets");
    populateFilterSelect("compareModel", result.available.models, "All models");
    renderComparison();
  } catch (error) {
    if (version === state.comparisonVersion) toast(error.message, true);
  } finally {
    if (version === state.comparisonVersion) $("refreshComparison").disabled = false;
  }
}

function setComparisonStatistic(statistic) {
  state.comparisonStatistic = statistic;
  document.querySelectorAll("[data-statistic]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.statistic === statistic));
  });
  loadComparison();
}

function formatMetricValue(value) {
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute >= 10000 || absolute < 0.0001)) return value.toExponential(3);
  return Number(value.toPrecision(6)).toLocaleString(undefined, {maximumSignificantDigits: 6});
}

function metricCard(metric) {
  const result = state.compareResults;
  const extents = metric.values.flatMap((entry) => {
    if (!Number.isFinite(entry.spread)) return [entry.value];
    const halfRange = entry.spread * result.spread_half_width;
    return [entry.value - halfRange, entry.value + halfRange];
  });
  const minimum = Math.min(0, ...extents);
  const maximum = Math.max(0, ...extents);
  const span = maximum - minimum || 1;
  const zero = ((0 - minimum) / span) * 100;
  const rows = metric.values.map((entry) => {
    const endpoint = ((entry.value - minimum) / span) * 100;
    const left = Math.min(zero, endpoint);
    const width = Math.abs(endpoint - zero);
    const observations = entry.observations === 1 ? "1 stored row" : `${entry.observations} stored rows`;
    const hasSpread = Number.isFinite(entry.spread);
    const halfRange = hasSpread ? entry.spread * result.spread_half_width : 0;
    const intervalStart = ((entry.value - halfRange - minimum) / span) * 100;
    const intervalWidth = ((2 * halfRange) / span) * 100;
    const spreadText = hasSpread
      ? (result.statistic === "mean_std" ? `± ${formatMetricValue(entry.spread)}` : `IQR ${formatMetricValue(entry.spread)}`)
      : `${result.spread_label} —`;
    const spreadRows = entry.spread_observations === 1 ? "1 spread row" : `${entry.spread_observations} spread rows`;
    return `
      <div class="metric-row">
        <div class="metric-label" title="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}<small>${escapeHtml(observations)}</small></div>
        <div class="metric-track">
          <i class="metric-zero" style="left:${zero.toFixed(3)}%"></i>
          <b class="metric-fill${entry.value < 0 ? " negative" : ""}" style="left:${left.toFixed(3)}%;width:${Math.max(width, entry.value === 0 ? 0 : 0.7).toFixed(3)}%"></b>
          ${hasSpread ? `<span class="metric-whisker" title="${escapeHtml(`${result.spread_label}: ${formatMetricValue(entry.spread)} · ${spreadRows}`)}" style="left:${intervalStart.toFixed(3)}%;width:${Math.max(intervalWidth, 0.25).toFixed(3)}%"></span>` : ""}
        </div>
        <div class="metric-value" title="${escapeHtml(`${result.center_label}: ${formatMetricValue(entry.value)} · ${spreadText}`)}"><strong>${escapeHtml(formatMetricValue(entry.value))}</strong><small>${escapeHtml(spreadText)}</small></div>
      </div>`;
  }).join("");
  return `
    <article class="metric-card">
      <div class="metric-heading"><h3>${escapeHtml(metric.name)}</h3><span>${metric.values.length} groups</span></div>
      <div class="metric-rows">${rows}</div>
    </article>`;
}

function renderComparison() {
  const result = state.compareResults;
  const metrics = $("compareMetrics");
  const empty = $("compareEmpty");
  if (!result) {
    metrics.innerHTML = "";
    empty.hidden = false;
    return;
  }
  const groupLabel = {model: "model name", dataset: "dataset name", run: "run name"}[result.group_by];
  const intervalDescription = result.statistic === "mean_std" ? "whiskers: mean ± std" : "whisker width: IQR, centered on median";
  $("compareSummary").textContent = `${result.matched_records.toLocaleString()} of ${result.total_records.toLocaleString()} stored ${result.center_label} rows · averaged by ${groupLabel} · ${intervalDescription} · source: ${result.source}`;
  empty.hidden = result.metrics.length > 0;
  metrics.innerHTML = result.metrics.map(metricCard).join("");
}

function quickIsActive() {
  return ["starting", "running", "cancelling"].includes(state.quickRun?.status);
}

function setQuickControlsDisabled(disabled) {
  document.querySelectorAll("#panel-quick .quick-settings input, #panel-quick .quick-settings select").forEach((control) => {
    control.disabled = disabled;
  });
  $("quickRunButton").disabled = disabled;
}

function formatDeclaredMetric(value, declaration) {
  if (!Number.isFinite(value)) return "—";
  const match = /^\.(\d+)g$/.exec(declaration?.format || "");
  if (match) return Number(value.toPrecision(Number(match[1]))).toLocaleString();
  return formatMetricValue(value);
}

function prepareCanvas(canvas) {
  const width = Math.max(240, canvas.clientWidth || canvas.parentElement.clientWidth);
  const height = Math.max(120, canvas.clientHeight || canvas.parentElement.clientHeight);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  return {context, width, height};
}

function drawLineChart(canvas, series, selectedIteration, color = "#21745f") {
  const {context, width, height} = prepareCanvas(canvas);
  const values = series.filter((point) => Number.isFinite(point.y));
  if (!values.length) {
    context.fillStyle = "#8a9490";
    context.font = '11px "SFMono-Regular", Consolas, monospace';
    context.textAlign = "center";
    context.fillText("Waiting for values…", width / 2, height / 2);
    return;
  }

  const padding = {left: 48, right: 14, top: 14, bottom: 27};
  let minX = Math.min(...values.map((point) => point.x));
  let maxX = Math.max(...values.map((point) => point.x));
  if (minX === maxX) maxX = minX + 1;
  const rawY = values.map((point) => point.y);
  const positiveMin = Math.min(...rawY);
  const positiveMax = Math.max(...rawY);
  const logScale = positiveMin > 0 && positiveMax / positiveMin >= 100;
  const yTransform = (value) => logScale ? Math.log10(value) : value;
  let minY = Math.min(...rawY.map(yTransform));
  let maxY = Math.max(...rawY.map(yTransform));
  if (minY === maxY) {
    const paddingY = Math.abs(minY) * 0.08 || 1;
    minY -= paddingY;
    maxY += paddingY;
  } else {
    const paddingY = (maxY - minY) * 0.08;
    minY -= paddingY;
    maxY += paddingY;
  }
  const screenX = (value) => padding.left + ((value - minX) / (maxX - minX)) * (width - padding.left - padding.right);
  const screenY = (value) => height - padding.bottom - ((yTransform(value) - minY) / (maxY - minY)) * (height - padding.top - padding.bottom);

  context.strokeStyle = "rgba(35,57,50,.09)";
  context.fillStyle = "#7b8581";
  context.font = '9px "SFMono-Regular", Consolas, monospace';
  context.lineWidth = 1;
  context.textAlign = "right";
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (index / 4) * (height - padding.top - padding.bottom);
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
    const transformed = maxY - (index / 4) * (maxY - minY);
    const labelValue = logScale ? 10 ** transformed : transformed;
    context.fillText(formatMetricValue(labelValue), padding.left - 6, y + 3);
  }
  context.textAlign = "left";
  context.fillText(String(minX), padding.left, height - 8);
  context.textAlign = "right";
  context.fillText(String(Math.max(...values.map((point) => point.x))), width - padding.right, height - 8);
  if (logScale) {
    context.textAlign = "left";
    context.fillStyle = "#9a6a28";
    context.fillText("log scale", padding.left + 4, padding.top + 10);
  }

  const snapshots = state.quickSnapshots;
  for (let index = 1; index < snapshots.length; index += 1) {
    const current = snapshots[index].metadata?.bandwidth_index;
    const previous = snapshots[index - 1].metadata?.bandwidth_index;
    if (current === undefined || current === previous) continue;
    const x = screenX(snapshots[index].iteration);
    context.save();
    context.setLineDash([3, 4]);
    context.strokeStyle = "rgba(169,102,24,.45)";
    context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
    context.restore();
  }

  if (Number.isFinite(selectedIteration)) {
    const x = screenX(selectedIteration);
    context.strokeStyle = "rgba(22,32,30,.23)";
    context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
  }

  context.strokeStyle = color;
  context.lineWidth = 2;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();
  values.forEach((point, index) => {
    const x = screenX(point.x);
    const y = screenY(point.y);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  for (const point of values) {
    context.beginPath();
    context.arc(screenX(point.x), screenY(point.y), point.x === selectedIteration ? 3.4 : 2, 0, Math.PI * 2);
    context.fillStyle = point.x === selectedIteration ? "#172522" : color;
    context.fill();
  }
}

function quickSelectedSnapshot() {
  return state.quickSnapshots[state.quickSelectedIndex] || null;
}

function quickProjectionBounds(data, snapshot) {
  const components = [
    ...(data?.true_components || []),
    ...(snapshot?.estimated_components || []),
  ];
  return previewBounds({points: data?.points || [], components});
}

function strokeEllipse(context, component, screenX, screenY, scale, options = {}) {
  const centerX = screenX(component.center[0]);
  const centerY = screenY(component.center[1]);
  context.save();
  context.translate(centerX, centerY);
  context.rotate(-component.angle);
  context.beginPath();
  context.ellipse(0, 0, component.radii[0] * scale, component.radii[1] * scale, 0, 0, Math.PI * 2);
  context.strokeStyle = options.color;
  context.lineWidth = options.width || 2;
  context.globalAlpha = options.alpha ?? 1;
  context.setLineDash(options.dashed ? [6, 5] : []);
  context.stroke();
  context.restore();
  context.globalAlpha = 1;
  return {centerX, centerY};
}

function drawQuickProjection() {
  const data = state.quickData;
  if (!data?.points?.length) return;
  const snapshot = quickSelectedSnapshot();
  const canvas = $("quickPcaCanvas");
  const {context, width, height} = prepareCanvas(canvas);
  const bounds = quickProjectionBounds(data, snapshot);
  const padding = 34;
  const spanX = bounds.maxX - bounds.minX;
  const spanY = bounds.maxY - bounds.minY;
  const scale = Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY);
  const middleX = (bounds.minX + bounds.maxX) / 2;
  const middleY = (bounds.minY + bounds.maxY) / 2;
  const screenX = (value) => width / 2 + (value - middleX) * scale;
  const screenY = (value) => height / 2 - (value - middleY) * scale;

  context.strokeStyle = "rgba(35,57,50,.09)";
  context.lineWidth = 1;
  if (bounds.minX <= 0 && bounds.maxX >= 0) {
    context.beginPath(); context.moveTo(screenX(0), 0); context.lineTo(screenX(0), height); context.stroke();
  }
  if (bounds.minY <= 0 && bounds.maxY >= 0) {
    context.beginPath(); context.moveTo(0, screenY(0)); context.lineTo(width, screenY(0)); context.stroke();
  }
  for (const point of data.points) {
    context.beginPath();
    context.arc(screenX(point.x), screenY(point.y), point.label < 0 ? 2 : 1.65, 0, Math.PI * 2);
    context.fillStyle = pointColor(point.label);
    context.globalAlpha = point.label < 0 ? 0.35 : 0.42;
    context.fill();
  }
  context.globalAlpha = 1;

  for (const component of data.true_components || []) {
    const color = pointColor(component.index);
    const center = strokeEllipse(context, component, screenX, screenY, scale, {color, width: 1.8, dashed: true, alpha: 0.76});
    context.beginPath();
    context.moveTo(center.centerX - 5, center.centerY - 5); context.lineTo(center.centerX + 5, center.centerY + 5);
    context.moveTo(center.centerX + 5, center.centerY - 5); context.lineTo(center.centerX - 5, center.centerY + 5);
    context.strokeStyle = color; context.lineWidth = 2; context.globalAlpha = 0.82; context.stroke(); context.globalAlpha = 1;
  }
  for (const component of snapshot?.estimated_components || []) {
    const color = pointColor(component.index);
    const center = strokeEllipse(context, component, screenX, screenY, scale, {color, width: 2.5});
    context.beginPath(); context.arc(center.centerX, center.centerY, 4, 0, Math.PI * 2);
    context.fillStyle = "white"; context.fill(); context.strokeStyle = color; context.lineWidth = 2.4; context.stroke();
  }

  context.fillStyle = "#6b7672";
  context.font = '10px "SFMono-Regular", Consolas, monospace';
  context.fillText("PC1", width - 29, height / 2 + 13);
  context.save(); context.translate(width / 2 - 11, 28); context.rotate(-Math.PI / 2); context.fillText("PC2", 0, 0); context.restore();
}

function renderQuickMetricCharts() {
  const selected = quickSelectedSnapshot();
  const selectedIteration = selected?.iteration;
  const declarations = state.declarations.metrics;
  $("quickMetricCharts").innerHTML = declarations.map((metric, index) => {
    const value = selected?.metrics?.[metric.id];
    const direction = metric.direction === "minimize" ? "lower is better" : (metric.direction === "maximize" ? "higher is better" : "diagnostic");
    return `<div class="quick-metric-chart">
      <div class="quick-metric-chart-heading"><strong>${escapeHtml(metric.display_name)}</strong><span>${escapeHtml(formatDeclaredMetric(value, metric))}</span></div>
      <p>${escapeHtml(metric.description)} · ${escapeHtml(direction)}</p>
      <canvas id="quickMetricCanvas${index}" aria-label="${escapeHtml(metric.display_name)} by iteration"></canvas>
    </div>`;
  }).join("");
  declarations.forEach((metric, index) => {
    const series = state.quickSnapshots
      .map((snapshot) => ({x: snapshot.iteration, y: snapshot.metrics?.[metric.id]}))
      .filter((point) => Number.isFinite(point.y));
    drawLineChart($("quickMetricCanvas" + index), series, selectedIteration, plotColors[(index + 1) % plotColors.length]);
  });
}

function renderQuickFinalMetrics() {
  const metrics = state.quickRun?.final_metrics;
  const card = $("quickFinalCard");
  if (!metrics) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  $("quickFitTime").textContent = Number.isFinite(state.quickRun.fit_time) ? `${state.quickRun.fit_time.toFixed(2)}s fit time` : "";
  $("quickFinalMetrics").innerHTML = metrics.map((metric) => {
    const declaration = state.declarations.metrics.find((item) => item.id === metric.id) || metric;
    const failed = !Number.isFinite(metric.value);
    const direction = metric.direction === "minimize" ? "↓ lower is better" : (metric.direction === "maximize" ? "↑ higher is better" : "diagnostic");
    return `<div class="quick-final-metric${failed ? " metric-failed" : ""}">
      <span>${escapeHtml(metric.name)} · ${escapeHtml(direction)}</span>
      <strong>${failed ? "Unavailable" : escapeHtml(formatDeclaredMetric(metric.value, declaration))}</strong>
      <p>${escapeHtml(metric.error || metric.description)}</p>
    </div>`;
  }).join("");
}

function renderQuickVisuals() {
  if ($("quickDashboard").hidden) return;
  const selected = quickSelectedSnapshot();
  $("quickIterationSlider").disabled = state.quickSnapshots.length === 0;
  $("quickIterationSlider").max = String(Math.max(0, state.quickSnapshots.length - 1));
  $("quickIterationSlider").value = String(state.quickSelectedIndex);
  if (selected) {
    const bandwidth = selected.metadata?.bandwidth;
    const stage = bandwidth === undefined ? selected.phase : `${selected.phase} · s=${formatMetricValue(bandwidth)}`;
    $("quickIterationLabel").textContent = `#${selected.iteration} · ${stage}`;
    $("quickLossValue").textContent = formatMetricValue(selected.loss);
  } else {
    $("quickIterationLabel").textContent = "Waiting…";
    $("quickLossValue").textContent = "—";
  }
  const lossSeries = state.quickSnapshots.map((snapshot) => ({x: snapshot.iteration, y: snapshot.loss}));
  drawLineChart($("quickLossCanvas"), lossSeries, selected?.iteration);
  renderQuickMetricCharts();
  drawQuickProjection();
  $("quickLatestButton").hidden = state.quickFollowLatest || !state.quickSnapshots.length;
}

function renderQuickRun() {
  const run = state.quickRun;
  if (!run) return;
  $("quickEmpty").hidden = true;
  $("quickDashboard").hidden = false;
  $("quickRunTitle").textContent = run.metadata?.run_name || "Quick experiment";
  const iterationCount = state.quickSnapshots.length;
  const dataset = run.metadata?.dataset_display_name || run.metadata?.dataset_name || "dataset";
  const model = run.metadata?.model_display_name || run.metadata?.model_name || "model";
  $("quickRunMeta").textContent = `${run.id} · ${dataset} · ${model} · ${iterationCount} reported state${iterationCount === 1 ? "" : "s"}`;
  const status = $("quickStatus");
  status.textContent = run.status;
  status.className = `status-pill status-${run.status}`;
  $("quickCancelButton").hidden = !quickIsActive();
  $("quickSaveButton").hidden = run.status !== "completed" || Boolean(run.saved_path);
  $("quickSaveButton").textContent = run.saved_path ? "Saved" : "Save to results";
  const error = $("quickError");
  error.hidden = !run.error || run.status === "cancelled";
  error.textContent = run.error || "";
  setQuickControlsDisabled(quickIsActive());
  updateRunningCount();
  renderQuickFinalMetrics();
  requestAnimationFrame(renderQuickVisuals);
}

function hydrateQuickRun(run) {
  state.quickRun = run;
  state.quickRunId = run.id;
  state.quickData = run.data || state.quickData;
  if (Array.isArray(run.snapshots)) state.quickSnapshots = run.snapshots;
  if (state.quickFollowLatest && state.quickSnapshots.length) {
    state.quickSelectedIndex = state.quickSnapshots.length - 1;
  } else {
    state.quickSelectedIndex = Math.min(state.quickSelectedIndex, Math.max(0, state.quickSnapshots.length - 1));
  }
  if (state.quickData) {
    const variance = state.quickData.explained_variance_ratio.map((value) => `${(value * 100).toFixed(1)}%`);
    $("quickPcaMeta").textContent = `${state.quickData.displayed_points.toLocaleString()} points · ${state.quickData.n_features}D · variance ${variance.join(" + ")}`;
  }
  renderQuickRun();
}

function closeQuickStream() {
  if (state.quickEventSource) state.quickEventSource.close();
  state.quickEventSource = null;
}

function applyQuickEvent(event) {
  state.quickLastEvent = Math.max(state.quickLastEvent, event.sequence || 0);
  if (event.type === "data") state.quickData = event.data;
  if (event.type === "iteration") {
    const existing = state.quickSnapshots.findIndex((snapshot) => snapshot.iteration === event.snapshot.iteration);
    if (existing >= 0) state.quickSnapshots[existing] = event.snapshot;
    else state.quickSnapshots.push(event.snapshot);
    if (state.quickFollowLatest) state.quickSelectedIndex = state.quickSnapshots.length - 1;
  }
  if (event.run) state.quickRun = {...state.quickRun, ...event.run};
  if (event.type === "complete") state.quickRun.final_metrics = event.final_metrics;
  if (event.type === "saved") state.quickRun.saved_path = event.result_path;
  if (state.quickData) {
    const variance = state.quickData.explained_variance_ratio.map((value) => `${(value * 100).toFixed(1)}%`);
    $("quickPcaMeta").textContent = `${state.quickData.displayed_points.toLocaleString()} points · ${state.quickData.n_features}D · variance ${variance.join(" + ")}`;
  }
  renderQuickRun();
  if (!quickIsActive()) closeQuickStream();
}

function connectQuickStream() {
  closeQuickStream();
  if (!state.quickRunId || !quickIsActive()) return;
  const runId = state.quickRunId;
  const source = new EventSource(`/api/quick-runs/${encodeURIComponent(runId)}/events?after=${state.quickLastEvent}`);
  state.quickEventSource = source;
  source.onmessage = (message) => {
    if (runId !== state.quickRunId) return;
    try { applyQuickEvent(JSON.parse(message.data)); }
    catch (error) { toast(`Could not read live update: ${error.message}`, true); }
  };
  source.onerror = async () => {
    source.close();
    if (runId !== state.quickRunId) return;
    try {
      hydrateQuickRun(await api(`/api/quick-runs/${encodeURIComponent(runId)}`));
      if (quickIsActive()) setTimeout(connectQuickStream, 900);
    } catch (error) {
      toast(error.message, true);
    }
  };
}

async function startQuickRun() {
  const button = $("quickRunButton");
  button.disabled = true;
  closeQuickStream();
  state.quickData = null;
  state.quickSnapshots = [];
  state.quickSelectedIndex = 0;
  state.quickFollowLatest = true;
  state.quickLastEvent = 0;
  state.quickRun = null;
  $("quickFinalCard").hidden = true;
  $("quickError").hidden = true;
  try {
    const run = await api("/api/quick-runs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(buildQuickRequest()),
    });
    hydrateQuickRun(run);
    connectQuickStream();
    toast("Quick experiment started");
  } catch (error) {
    button.disabled = false;
    toast(error.message, true);
  }
}

async function cancelQuickRun() {
  if (!state.quickRunId) return;
  try {
    hydrateQuickRun(await api(`/api/quick-runs/${encodeURIComponent(state.quickRunId)}/cancel`, {method: "POST"}));
  } catch (error) {
    toast(error.message, true);
  }
}

async function saveQuickRun() {
  if (!state.quickRunId) return;
  const specs = state.declarations.execution.quick_parameters;
  const execution = readParameters($("quickExecutionParameters"), specs);
  $("quickSaveButton").disabled = true;
  try {
    const saved = await api(`/api/quick-runs/${encodeURIComponent(state.quickRunId)}/save`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({run_name: execution.run_name}),
    });
    state.quickRun.saved_path = saved.result_path;
    $("quickSaveButton").hidden = true;
    await refreshRuns(false, true);
    toast(`Saved as ${saved.run_name}`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("quickSaveButton").disabled = false;
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${name}`));
  if (name === "results") refreshRuns(true, true);
  if (name === "compare") loadComparison();
}

function tabFromLocation() {
  if (window.location.pathname === "/quick") return "quick";
  const candidate = window.location.hash.replace(/^#/, "");
  return ["experiment", "results", "compare"].includes(candidate) ? candidate : "experiment";
}

function navigateToTab(name) {
  const destination = name === "quick" ? "/quick" : (name === "experiment" ? "/" : `/#${name}`);
  window.history.pushState({}, "", destination);
  switchTab(name);
}

function bindEvents() {
  $("advancedToggle").addEventListener("click", () => setAdvanced(!state.advanced));
  $("datasetSelect").addEventListener("change", (event) => selectDataset(event.target.value));
  $("modelSelect").addEventListener("change", (event) => selectModel(event.target.value));
  $("quickDatasetSelect").addEventListener("change", (event) => selectQuickDataset(event.target.value));
  $("quickModelSelect").addEventListener("change", (event) => selectQuickModel(event.target.value));
  $("previewButton").addEventListener("click", previewData);
  $("runButton").addEventListener("click", startRun);
  $("quickRunButton").addEventListener("click", startQuickRun);
  $("quickCancelButton").addEventListener("click", cancelQuickRun);
  $("quickSaveButton").addEventListener("click", saveQuickRun);
  $("quickLatestButton").addEventListener("click", () => {
    state.quickFollowLatest = true;
    state.quickSelectedIndex = Math.max(0, state.quickSnapshots.length - 1);
    renderQuickVisuals();
  });
  $("quickIterationSlider").addEventListener("input", (event) => {
    state.quickFollowLatest = false;
    state.quickSelectedIndex = Number.parseInt(event.target.value, 10) || 0;
    renderQuickVisuals();
  });
  $("cancelButton").addEventListener("click", async () => {
    if (!state.currentRunId) return;
    try { renderLive(await api(`/api/runs/${state.currentRunId}/cancel`, {method: "POST"})); }
    catch (error) { toast(error.message, true); }
  });
  $("refreshResults").addEventListener("click", () => refreshRuns(true, true));
  $("runsTable").addEventListener("click", (event) => {
    const button = event.target.closest("[data-run-id]");
    if (button) showRunDetails(button.dataset.runId, button.dataset.runSource);
  });
  $("runsTable").addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    const row = event.target.closest("tr[data-run-id]");
    if (!row) return;
    event.preventDefault();
    showRunDetails(row.dataset.runId, row.dataset.runSource);
  });
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => navigateToTab(tab.dataset.tab)));
  $("refreshComparison").addEventListener("click", loadComparison);
  $("compareRun").addEventListener("change", loadComparison);
  $("compareDataset").addEventListener("change", loadComparison);
  $("compareModel").addEventListener("change", loadComparison);
  $("compareGroupBy").addEventListener("change", loadComparison);
  document.querySelectorAll("[data-statistic]").forEach((button) => {
    button.addEventListener("click", () => setComparisonStatistic(button.dataset.statistic));
  });
  window.addEventListener("resize", () => {
    clearTimeout(drawDataPreview.resizeTimer);
    drawDataPreview.resizeTimer = setTimeout(() => {
      drawDataPreview();
      renderQuickVisuals();
    }, 100);
  });
  window.addEventListener("beforeunload", closeQuickStream);
  window.addEventListener("popstate", () => switchTab(tabFromLocation()));
}

async function init() {
  bindEvents();
  try {
    state.declarations = await api("/api/declarations");
    populateEntitySelect($("datasetSelect"), state.declarations.datasets);
    populateEntitySelect($("modelSelect"), state.declarations.models);
    populateEntitySelect($("quickDatasetSelect"), state.declarations.datasets);
    populateEntitySelect($("quickModelSelect"), state.declarations.models);
    renderParameters($("executionParameters"), state.declarations.execution.parameters);
    renderParameters($("quickExecutionParameters"), state.declarations.execution.quick_parameters);
    selectDataset(state.declarations.datasets[0].id);
    selectModel(state.declarations.models[0].id);
    selectQuickDataset(state.declarations.datasets[0].id);
    selectQuickModel(state.declarations.models[0].id);
    updateWorkload();
    switchTab(tabFromLocation());
    await refreshRuns(false, true);
    setInterval(() => refreshRuns(false, false), 4000);
  } catch (error) {
    toast(`Dashboard initialization failed: ${error.message}`, true);
  }
}

init();
