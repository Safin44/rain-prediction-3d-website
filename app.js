// ATMOS Rain Observatory — app logic
// Verified results copied from the executed notebook DS422_Rain_Prediction.ipynb:
// DT & RF 1.0000 / AUC 1.0, SVC 0.9766 / 0.9965, KNN 0.9587 / 0.9879, LR 0.9311 / 0.9766.
// Discovered rule (verified on 100.00% of 55,996 rows): Humidity > 65 AND Cloud > 4.5 AND Rainfall > 2.
// NOTE: the rule is shown only as an explanation (the gates). The VERDICT comes
// live from the Python backend, which loads random_forest_rain_model.pkl —
// there is deliberately NO JavaScript-rule fallback.

const API_BASE = ""; // same origin — locally the backend serves this page, on Render the deployed service does
const RULE = { humidity: 65, cloud: 4.5, rainfall: 2 };
const BACKEND_HINT = "Backend not running. Locally, start it from the backend folder with: python -m uvicorn app:app --host 127.0.0.1 --port 8001";

const MODELS = [
  { name: "Decision Tree 🏆", test: 100.0, auc: "1.0000", winner: true, color: "#e9a426" },
  { name: "Random Forest 🏆", test: 100.0, auc: "1.0000", winner: true, color: "#0e7c7b" },
  { name: "SVC", test: 97.66, auc: "0.9965", winner: false, color: "#2f6f9f" },
  { name: "KNN (k = 5)", test: 95.87, auc: "0.9879", winner: false, color: "#8b5cf6" },
  { name: "Logistic Regression", test: 93.11, auc: "0.9766", winner: false, color: "#b45309" },
];

// ---- Model race (lollipop rows) ----
(function renderRace() {
  const list = document.getElementById("raceList");
  if (!list) return;
  const scale = (v) => Math.max(4, ((v - 90) / 10) * 100);
  list.innerHTML = MODELS.map((m) => `
    <div class="race-row ${m.winner ? "winner" : ""}">
      <span class="race-name">${m.name}</span>
      <span class="race-track"><span class="race-fill" style="width:${scale(m.test)}%;background:${m.color}"></span><span class="race-dot" style="left:${scale(m.test)}%;border-color:${m.color}"></span></span>
      <span class="race-val" style="color:${m.color}">${m.test.toFixed(2)}%</span>
    </div>`).join("") +
    `<p style="color:var(--muted);font-size:.85rem;margin:.2rem 0 0">Bar length scaled 90–100% so the gaps are visible; labels and the table show the exact test accuracy. AUC: winners 1.0000 · SVC 0.9965 · KNN 0.9879 · LR 0.9766.</p>`;
})();

// ---- Live predictor (backend / .pkl model) ----
const els = {};
["humidity", "cloud", "rainfall", "minTemp", "maxTemp", "pressure", "windSpeed", "windGust"]
  .forEach((id) => { els[id] = document.getElementById(id); });

function num(id, fallback) {
  const el = els[id];
  let v = parseFloat(el && el.value);
  if (!Number.isFinite(v)) return fallback;
  // Clamp to the dataset range (the input's min/max) — the model never saw values outside it.
  const lo = parseFloat(el.min), hi = parseFloat(el.max);
  if (Number.isFinite(lo) && v < lo) v = lo;
  if (Number.isFinite(hi) && v > hi) v = hi;
  return v;
}

// Write the clamped value back into number inputs when the user leaves the field.
["minTemp", "maxTemp", "pressure", "windSpeed", "windGust"].forEach((id) => {
  if (!els[id]) return;
  els[id].addEventListener("blur", () => {
    const v = num(id, parseFloat(els[id].defaultValue));
    if (String(v) !== els[id].value) { els[id].value = v; update(); }
  });
});

function readInputs() {
  return {
    sliders: {
      humidity: num("humidity", 0),
      cloud: num("cloud", 0),
      rainfall: num("rainfall", 0),
    },
    // Exact payload for POST /predict — all 8 model features, notebook names.
    payload: {
      MinTemp: num("minTemp", 14.2),
      MaxTemp: num("maxTemp", 24.8),
      Rainfall: num("rainfall", 0),
      Humidity: num("humidity", 0),
      Pressure: num("pressure", 1015),
      WindSpeed: num("windSpeed", 18),
      WindGustSpeed: num("windGust", 32),
      Cloud: num("cloud", 0),
    },
  };
}

function evaluateGates(v) {
  return {
    humidity: v.humidity > RULE.humidity,
    cloud: v.cloud > RULE.cloud,
    rainfall: v.rainfall > RULE.rainfall,
  };
}

function fmt(id, v) {
  if (id === "humidity") return `${v}%`;
  if (id === "cloud") return `${v} / 8`;
  return `${Number(v).toFixed(1)} mm`;
}

let debounceTimer = null;
let requestSeq = 0;

function showChecking() {
  document.getElementById("verdictSky").textContent = "⏳";
  document.getElementById("verdictLabel").textContent = "CHECKING MODEL…";
  document.getElementById("verdictSub").textContent = "Asking random_forest_rain_model.pkl via the local backend…";
}

function setGauge(pct, trees, total) {
  const fill = document.getElementById("gaugeFill");
  const val = document.getElementById("gaugeVal");
  const votes = document.getElementById("gaugeVotes");
  if (!fill) return;
  if (pct == null) {
    fill.style.width = "0%"; val.textContent = "—"; votes.textContent = "";
    return;
  }
  fill.style.width = `${pct}%`;
  val.textContent = `${pct}%`;
  votes.textContent = (trees != null && total) ? `${trees} of ${total} trees vote RAIN` : "";
}

function setDistances(d) {
  const box = document.getElementById("distList");
  if (!box) return;
  if (!d) { box.innerHTML = ""; return; }
  const units = { Humidity: "%", Cloud: " oktas", Rainfall: " mm" };
  const th = { Humidity: 65.01, Cloud: 4.5, Rainfall: 2.01 };
  box.innerHTML = Object.keys(th).map((k) => {
    const v = d[k];
    const pass = v > 0;
    return `<li class="${pass ? "pass" : "fail"}">${k}: ${v > 0 ? "+" : ""}${v}${units[k]} from the tree threshold ${th[k]}${units[k]} ${pass ? "(above)" : "(below)"}</li>`;
  }).join("");
}

function setRainVideo(show) {
  const wrap = document.getElementById("rainVideoWrap");
  const vid = document.getElementById("rainVideo");
  if (!wrap || !vid) return;
  wrap.hidden = !show;
  if (show) {
    vid.play().catch(() => {});
  } else {
    vid.pause();
    vid.currentTime = 0;
  }
}

function showBackendError(message) {
  const box = document.getElementById("verdictBox");
  box.classList.remove("is-rain");
  box.classList.add("is-dry");
  document.getElementById("verdictSky").textContent = "⚠️";
  document.getElementById("verdictLabel").textContent = message ? "INPUT REJECTED" : "BACKEND NOT RUNNING";
  document.getElementById("verdictSub").textContent = message || BACKEND_HINT;
  setRainVideo(false);
  setGauge(null); setDistances(null);
}

async function fetchVerdict(payload) {
  const seq = ++requestSeq;
  showChecking();
  let res;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    if (seq !== requestSeq) return;
    showBackendError();
    setHealth(false);
    return;
  }
  if (seq !== requestSeq) return;
  if (res.status === 422) {
    // Backend rejects values outside the dataset range.
    let msg = "A reading is outside the dataset range.";
    try {
      const body = await res.json();
      if (Array.isArray(body.detail)) {
        msg = body.detail.map((d) => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join(" · ");
      }
    } catch (_) { /* keep default */ }
    showBackendError(msg);
    return;
  }
  if (!res.ok) { showBackendError(`Backend error: HTTP ${res.status}`); return; }
  try {
    const data = await res.json();
    if (seq !== requestSeq) return; // a newer request superseded this one
    setHealth(true);
    const rain = data.prediction === 1;
    const pct = (data.rain_probability_percent != null)
      ? data.rain_probability_percent
      : Math.round((data.rain_probability || 0) * 1000) / 10;
    const box = document.getElementById("verdictBox");
    box.classList.toggle("is-rain", rain);
    box.classList.toggle("is-dry", !rain);
    document.getElementById("verdictSky").textContent = rain ? "🌧" : "☀️";
    document.getElementById("verdictLabel").textContent = rain ? "RAIN TOMORROW" : "NO RAIN TOMORROW";
    document.getElementById("verdictSub").textContent =
      `Model verdict: ${data.label} · rain probability ${pct}% — served live from ${data.model || "random_forest_rain_model.pkl"}.`;
    setRainVideo(rain);
    setGauge(pct, data.trees_voting_rain, data.n_estimators);
    setDistances(data.threshold_distance);
  } catch (err) {
    if (seq !== requestSeq) return;
    showBackendError("Backend returned an unreadable response.");
  }
}

// ---- Backend status badge (GET /health) ----
function setHealth(ok, info) {
  const b = document.getElementById("healthBadge");
  if (!b) return;
  b.classList.toggle("ok", !!ok);
  b.classList.toggle("down", !ok);
  if (!ok) { b.textContent = "● Backend offline"; return; }
  if (info) {
    b.textContent = `● Backend online · ${info.model_type || "model"} (${info.n_estimators || "?"} trees) · scaler: ${info.preprocessing_source || "?"} · scikit-learn ${info.sklearn_version || "?"}`;
  } else if (!b.dataset.loaded) {
    b.textContent = "● Backend online";
  }
}
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const info = await r.json();
    const b = document.getElementById("healthBadge");
    if (b) b.dataset.loaded = "1";
    setHealth(info.status === "ok", info);
  } catch (_) { setHealth(false); }
}
checkHealth();
setInterval(checkHealth, 15000);

function update() {
  const { sliders: v, payload } = readInputs();
  const gates = evaluateGates(v);

  for (const id of ["humidity", "cloud", "rainfall"]) {
    document.getElementById(id + "Out").textContent = fmt(id, v[id]);
  }
  const gateText = {
    humidity: ["gateHumidity", `Humidity ${v.humidity}% — ${gates.humidity ? "✓ gate passed" : "✗ gate not passed"}`],
    cloud: ["gateCloud", `Cloud ${v.cloud} oktas — ${gates.cloud ? "✓ gate passed" : "✗ gate not passed"}`],
    rainfall: ["gateRainfall", `Rainfall ${Number(v.rainfall).toFixed(1)} mm — ${gates.rainfall ? "✓ gate passed" : "✗ gate not passed"}`],
  };
  for (const id of ["humidity", "cloud", "rainfall"]) {
    const [elId, text] = gateText[id];
    const el = document.getElementById(elId);
    el.textContent = text;
    el.className = "gate " + (gates[id] ? "pass" : "fail");
  }

  const vGates = { humidity: "vGateHumidity", cloud: "vGateCloud", rainfall: "vGateRainfall" };
  const labels = { humidity: "Humidity", cloud: "Cloud", rainfall: "Rainfall" };
  for (const id of ["humidity", "cloud", "rainfall"]) {
    const li = document.getElementById(vGates[id]);
    li.textContent = `${labels[id]} gate (explanation): ${gates[id] ? "✓ PASS" : "✗ FAIL"} (${fmt(id, v[id])})`;
    li.className = gates[id] ? "pass" : "fail";
  }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => fetchVerdict(payload), 250);
}

["humidity", "cloud", "rainfall", "minTemp", "maxTemp", "pressure", "windSpeed", "windGust"]
  .forEach((id) => {
    if (!els[id]) return;
    els[id].addEventListener("input", update);
    els[id].addEventListener("change", update);
  });

function setPreset(p) {
  els.humidity.value = p.Humidity; els.cloud.value = p.Cloud; els.rainfall.value = p.Rainfall;
  els.minTemp.value = p.MinTemp; els.maxTemp.value = p.MaxTemp; els.pressure.value = p.Pressure;
  els.windSpeed.value = p.WindSpeed; els.windGust.value = p.WindGustSpeed;
  update();
}
document.getElementById("presetRain").addEventListener("click", () =>
  setPreset({ Humidity: 84, Cloud: 7, Rainfall: 12.6, MinTemp: 14.2, MaxTemp: 24.8, Pressure: 1015, WindSpeed: 18, WindGustSpeed: 32 }));
document.getElementById("presetDry").addEventListener("click", () =>
  setPreset({ Humidity: 38, Cloud: 1, Rainfall: 0.0, MinTemp: 18.0, MaxTemp: 32.0, Pressure: 1022, WindSpeed: 12, WindGustSpeed: 20 }));
document.getElementById("presetEdge").addEventListener("click", () =>
  setPreset({ Humidity: 81, Cloud: 7, Rainfall: 1.4, MinTemp: 15.0, MaxTemp: 25.0, Pressure: 1012, WindSpeed: 20, WindGustSpeed: 35 }));

update();
