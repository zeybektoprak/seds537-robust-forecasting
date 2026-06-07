const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.resolve(__dirname, "../results/figures");
const OUT = path.resolve(__dirname, "Zeybek_SEDS537_Presentation.pptx");

// ── Palette ────────────────────────────────────────────────────────────────────
const NAVY   = "1B2A4A";
const BLUE   = "2563EB";
const LBLUE  = "3B82F6";
const ACCENT = "F59E0B";
const WHITE  = "FFFFFF";
const LGRAY  = "F1F5F9";
const GRAY   = "64748B";
const DARK   = "0F172A";
const GREEN  = "10B981";
const RED    = "EF4444";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Toprak Zeybek";
pres.title  = "Robust Time-Series Forecasting – SEDS 537";

// ── Helpers ────────────────────────────────────────────────────────────────────
const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.18 });

function darkSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: DARK };
  return s;
}
function lightSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: LGRAY };
  return s;
}

function addTopBar(slide, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08, fill: { color: color || BLUE }, line: { color: color || BLUE }
  });
}
function addSlideTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.45, y: 0.18, w: 9.1, h: 0.55,
    fontSize: 22, bold: true, color: NAVY, fontFace: "Calibri",
    margin: 0
  });
  if (subtitle) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.45, y: 0.77, w: 0.42, h: 0.055,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });
    slide.addText(subtitle, {
      x: 0.95, y: 0.70, w: 8.5, h: 0.22,
      fontSize: 11, color: GRAY, fontFace: "Calibri", margin: 0
    });
  }
  slide.addShape(pres.shapes.LINE, {
    x: 0.45, y: 0.85, w: 9.1, h: 0,
    line: { color: "D1D5DB", width: 0.6 }
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 1 — Title
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = darkSlide(pres);
  // Left accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: BLUE }, line: { color: BLUE }
  });
  // Bottom accent
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.225, w: 10, h: 0.4, fill: { color: NAVY }, line: { color: NAVY }
  });

  // Title
  s.addText("Robust Time-Series Forecasting", {
    x: 0.6, y: 0.85, w: 8.8, h: 0.9,
    fontSize: 36, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });
  s.addText("Under Noise and Anomaly Injection", {
    x: 0.6, y: 1.65, w: 8.8, h: 0.65,
    fontSize: 28, bold: false, color: "93C5FD", fontFace: "Calibri", margin: 0
  });
  // Divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 2.45, w: 1.8, h: 0.07,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });
  // Subtitle info
  s.addText([
    { text: "ARIMA  ·  Vanilla RNN  ·  Stacked LSTM  ·  Temporal Transformer", options: { breakLine: true } },
    { text: "Jena Climate Dataset  |  Gaussian Noise + Point Anomaly Study", options: {} }
  ], {
    x: 0.6, y: 2.65, w: 8.5, h: 0.9,
    fontSize: 13, color: "94A3B8", fontFace: "Calibri", margin: 0
  });
  // Student info
  s.addText([
    { text: "Toprak Zeybek  |  ID: 323011022", options: { breakLine: true } },
    { text: "SEDS 537 Machine Learning  ·  Izmir Institute of Technology  ·  Spring 2026", options: { breakLine: true } },
    { text: "Instructor: Prof. Dr. Aytug Onan", options: {} }
  ], {
    x: 0.6, y: 4.7, w: 9, h: 0.6,
    fontSize: 10, color: "64748B", fontFace: "Calibri", margin: 0
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 2 — Motivation
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Motivation: Why Robustness Matters?", "Real-world sensors are never perfect");

  const boxes = [
    { icon: "⚡", title: "The Problem", text: "Models trained on clean data but deployed in noisy environments. Sensor faults, transmission errors, and measurement noise are ubiquitous." },
    { icon: "🔬", title: "The Gap", text: "Nearly all benchmarks evaluate on clean test sets. Robustness to corrupted inputs is largely unknown." },
    { icon: "💡", title: "Our Approach", text: "Train all models on clean data, then systematically corrupt only the test inputs. Measure how accuracy degrades." },
    { icon: "🎯", title: "The Hypothesis", text: "Transformer's global self-attention can 'outvote' corrupted positions. RNN/LSTM propagate corruption forward through hidden state." },
  ];

  boxes.forEach((b, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.3 + col * 4.85;
    const y = 1.1 + row * 1.9;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.5, h: 1.7,
      fill: { color: WHITE }, line: { color: "D1D5DB", width: 0.8 },
      shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.07, h: 1.7,
      fill: { color: BLUE }, line: { color: BLUE }
    });
    s.addText(b.icon + "  " + b.title, {
      x: x + 0.18, y: y + 0.12, w: 4.2, h: 0.35,
      fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(b.text, {
      x: x + 0.18, y: y + 0.48, w: 4.15, h: 1.1,
      fontSize: 10.5, color: GRAY, fontFace: "Calibri", margin: 0
    });
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 3 — Dataset & Methodology
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Dataset & Methodology", "Jena Climate Dataset — 4 models — 2 corruption types");

  // Left column – dataset
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.05, w: 4.5, h: 4.3,
    fill: { color: WHITE }, line: { color: "D1D5DB" }, shadow: makeShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.05, w: 4.5, h: 0.42,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("📊  Dataset", {
    x: 0.4, y: 1.07, w: 4.3, h: 0.36,
    fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });
  s.addText([
    { text: "Jena Climate 2009–2016", options: { bold: true, breakLine: true } },
    { text: "14 atmospheric variables, 10-min intervals", options: { breakLine: true } },
    { text: "Subsampled to hourly (70,091 rows)", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Target: Air Temperature T (°C)", options: { bold: true, breakLine: true } },
    { text: "Split: 70% train / 15% val / 15% test", options: { breakLine: true } },
    { text: "Window: W = 120 hours (5 days)", options: { breakLine: true } },
    { text: "Horizon: H = 1 step ahead", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Z-score normalisation (train stats only)", options: { italic: true } },
  ], {
    x: 0.45, y: 1.55, w: 4.25, h: 3.65,
    fontSize: 10.5, color: GRAY, fontFace: "Calibri", lineSpacingMultiple: 1.3, margin: 0
  });

  // Right column – corruption
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.05, w: 4.5, h: 4.3,
    fill: { color: WHITE }, line: { color: "D1D5DB" }, shadow: makeShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.05, w: 4.5, h: 0.42,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("⚠️  Corruption Types", {
    x: 5.3, y: 1.07, w: 4.3, h: 0.36,
    fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });

  // Gaussian box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.35, y: 1.6, w: 4.2, h: 1.55,
    fill: { color: "EFF6FF" }, line: { color: "BFDBFE" }
  });
  s.addText("Gaussian Noise", {
    x: 5.5, y: 1.67, w: 3.9, h: 0.3,
    fontSize: 11, bold: true, color: BLUE, fontFace: "Calibri", margin: 0
  });
  s.addText("x̃ = x + ε,   ε ~ N(0, σ²)\nσ ∈ {0, 0.25, 0.5, 1.0, 2.0}", {
    x: 5.5, y: 1.97, w: 3.9, h: 0.6,
    fontSize: 10, color: NAVY, fontFace: "Calibri", margin: 0
  });
  s.addText("σ = 1.0 → noise std = data std (severe)", {
    x: 5.5, y: 2.88, w: 3.8, h: 0.22, fontSize: 9, italic: true, color: GRAY, fontFace: "Calibri", margin: 0
  });

  // Anomaly box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.35, y: 3.35, w: 4.2, h: 1.7,
    fill: { color: "FFF7ED" }, line: { color: "FED7AA" }
  });
  s.addText("Point Anomaly Injection", {
    x: 5.5, y: 3.42, w: 3.9, h: 0.3,
    fontSize: 11, bold: true, color: "D97706", fontFace: "Calibri", margin: 0
  });
  s.addText("r fraction of positions → N(0, 25)\nr ∈ {0, 0.01, 0.05, 0.10, 0.20}", {
    x: 5.5, y: 3.72, w: 3.9, h: 0.6,
    fontSize: 10, color: NAVY, fontFace: "Calibri", margin: 0
  });
  s.addText("All models trained on CLEAN data only!", {
    x: 5.5, y: 4.6, w: 3.8, h: 0.22, fontSize: 9, italic: true, color: RED, bold: true, fontFace: "Calibri", margin: 0
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 4 — Example Input / Output
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Example: What the Model Sees & Predicts", "One sample from the Jena Climate test set");

  // LEFT — Input window visual
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.05, w: 5.5, h: 4.35,
    fill: { color: WHITE }, line: { color: "D1D5DB" }, shadow: makeShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.05, w: 5.5, h: 0.42,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("📥  INPUT  —  Last 120 hours (5 days)", {
    x: 0.42, y: 1.07, w: 5.3, h: 0.36,
    fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });

  // Mini table of input features
  const inputRows = [
    [{ text: "Feature", options: { bold: true, color: WHITE, fill: { color: "1d4ed8" } } },
     { text: "t-120 (5d ago)", options: { bold: true, color: WHITE, fill: { color: "1d4ed8" } } },
     { text: "t-60 (2.5d ago)", options: { bold: true, color: WHITE, fill: { color: "1d4ed8" } } },
     { text: "t-1 (1h ago)", options: { bold: true, color: WHITE, fill: { color: "1d4ed8" } } }],
    ["T (°C) ← target",    "-8.02",  "2.14",  "4.87"],
    ["p (mbar)",           "996.52", "998.10", "999.23"],
    ["rh (%)",             "93.3",   "76.4",   "71.2"],
    ["wv (m/s)",           "1.03",   "2.41",   "3.12"],
    ["... 9 more features","...",    "...",    "..."],
  ];
  s.addTable(inputRows, {
    x: 0.42, y: 1.58, w: 5.28, h: 2.1,
    fontFace: "Calibri", fontSize: 9.5, align: "center",
    rowH: 0.32, colW: [1.8, 1.16, 1.16, 1.16],
    border: { pt: 0.4, color: "D1D5DB" }, fill: { color: WHITE },
  });

  s.addText("→  Shape after preprocessing:  (120 steps  ×  13 features)  =  1,560 numbers", {
    x: 0.42, y: 3.78, w: 5.28, h: 0.28,
    fontSize: 9.5, italic: true, color: BLUE, fontFace: "Calibri", margin: 0
  });
  s.addText("→  All values z-score normalised using training statistics only", {
    x: 0.42, y: 4.1, w: 5.28, h: 0.28,
    fontSize: 9.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0
  });
  s.addText("→  Window slides by 1 hour each time  (10,394 test windows total)", {
    x: 0.42, y: 4.42, w: 5.28, h: 0.28,
    fontSize: 9.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0
  });

  // RIGHT — Output
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.0, y: 1.05, w: 3.7, h: 4.35,
    fill: { color: WHITE }, line: { color: "D1D5DB" }, shadow: makeShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.0, y: 1.05, w: 3.7, h: 0.42,
    fill: { color: "047857" }, line: { color: "047857" }
  });
  s.addText("📤  OUTPUT  —  Next hour", {
    x: 6.12, y: 1.07, w: 3.48, h: 0.36,
    fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });

  // Ground truth
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.12, y: 1.62, w: 3.46, h: 1.1,
    fill: { color: "F0FDF4" }, line: { color: "A7F3D0" }
  });
  s.addText("Ground Truth", {
    x: 6.22, y: 1.68, w: 3.26, h: 0.28,
    fontSize: 10, bold: true, color: "065F46", fontFace: "Calibri", margin: 0
  });
  s.addText("T = 5.21 °C  →  normalised: 0.41", {
    x: 6.22, y: 1.96, w: 3.26, h: 0.28,
    fontSize: 10.5, color: "047857", fontFace: "Calibri", margin: 0
  });

  // Model predictions
  const modelPreds = [
    { name: "RNN",         pred: "0.38", err: "|err| = 0.03", color: "EE854A" },
    { name: "LSTM",        pred: "0.40", err: "|err| = 0.01", color: "6ACC65" },
    { name: "Transformer", pred: "0.39", err: "|err| = 0.02", color: "D65F5F" },
  ];
  modelPreds.forEach((m, i) => {
    const y = 2.9 + i * 0.72;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.12, y, w: 3.46, h: 0.62,
      fill: { color: WHITE }, line: { color: "E2E8F0" }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.12, y, w: 0.06, h: 0.62,
      fill: { color: m.color }, line: { color: m.color }
    });
    s.addText(m.name, {
      x: 6.24, y: y + 0.04, w: 1.5, h: 0.28,
      fontSize: 10, bold: true, color: m.color, fontFace: "Calibri", margin: 0
    });
    s.addText("pred: " + m.pred, {
      x: 6.24, y: y + 0.31, w: 1.5, h: 0.24,
      fontSize: 9.5, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(m.err, {
      x: 7.78, y: y + 0.18, w: 1.6, h: 0.28,
      fontSize: 10, bold: true, color: "047857", fontFace: "Calibri", margin: 0
    });
  });

  s.addText("This is ONE sample. We do this for 10,394 test windows.", {
    x: 6.12, y: 5.06, w: 3.46, h: 0.28,
    fontSize: 8.5, italic: true, color: GRAY, fontFace: "Calibri", margin: 0
  });
}

// SLIDE 4 (old) → now SLIDE 5 — Models
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Four Models Compared", "3 Baselines + 1 Proposed Method");

  const models = [
    { color: "4878D0", label: "ARIMA", tag: "Baseline 1", desc: "ARIMA(5,1,0) fitted on last 2,000 training rows. Generative model — does NOT consume test input. Trivially robust to corruption but ignores multivariate context.", params: "Classical" },
    { color: "EE854A", label: "Vanilla RNN", tag: "Baseline 2", desc: "Single-layer RNN, 64 hidden units. Final hidden state → scalar. Sequential processing: corrupted step propagates forward in hidden state.", params: "~7K params" },
    { color: "6ACC65", label: "Stacked LSTM", tag: "Baseline 3", desc: "2-layer LSTM, 128 units, dropout = 0.2. MC-Dropout: 50 forward passes at inference → predictive uncertainty signal.", params: "~200K params" },
    { color: "D65F5F", label: "Transformer", tag: "Proposed", desc: "2 encoder layers, 4 attention heads, d_model=64, sinusoidal PE. Global attention dilutes corrupted positions — the core robustness hypothesis.", params: "~400K params" },
  ];

  models.forEach((m, i) => {
    const x = 0.3 + i * 2.38;
    const y = 1.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.22, h: 4.35,
      fill: { color: WHITE }, line: { color: "E2E8F0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.22, h: 0.08,
      fill: { color: m.color }, line: { color: m.color }
    });
    // Tag badge
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.12, y: y + 0.18, w: 1.98, h: 0.3,
      fill: { color: m.color, transparency: 85 }, line: { color: m.color, transparency: 50 }
    });
    s.addText(m.tag, {
      x: x + 0.12, y: y + 0.19, w: 1.98, h: 0.27,
      fontSize: 9, bold: true, color: m.color, align: "center", fontFace: "Calibri", margin: 0
    });
    s.addText(m.label, {
      x: x + 0.12, y: y + 0.55, w: 1.98, h: 0.5,
      fontSize: 17, bold: true, color: NAVY, align: "center", fontFace: "Calibri", margin: 0
    });
    s.addShape(pres.shapes.LINE, {
      x: x + 0.25, y: y + 1.1, w: 1.72, h: 0,
      line: { color: "E2E8F0", width: 0.7 }
    });
    s.addText(m.desc, {
      x: x + 0.12, y: y + 1.2, w: 1.98, h: 2.7,
      fontSize: 9.5, color: GRAY, fontFace: "Calibri", lineSpacingMultiple: 1.3, margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.12, y: y + 3.95, w: 1.98, h: 0.3,
      fill: { color: LGRAY }, line: { color: "E2E8F0" }
    });
    s.addText(m.params, {
      x: x + 0.12, y: y + 3.95, w: 1.98, h: 0.3,
      fontSize: 9, bold: true, color: BLUE, align: "center", fontFace: "Calibri", margin: 0
    });
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 5 — Clean Results
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Clean Test Set — Baseline Performance", "All models trained and evaluated on unmodified data");

  // Big stat callouts
  const stats = [
    { val: "0.0592", label: "LSTM  MAE", color: "6ACC65" },
    { val: "0.0824", label: "Transformer  RMSE", color: "D65F5F" },
    { val: "30.80%", label: "Transformer  MAPE", color: "D65F5F" },
    { val: "×14", label: "Neural vs ARIMA", color: BLUE },
  ];
  stats.forEach((st, i) => {
    const x = 0.3 + i * 2.38;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.05, w: 2.22, h: 1.5,
      fill: { color: WHITE }, line: { color: "E2E8F0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.05, w: 2.22, h: 0.07,
      fill: { color: st.color }, line: { color: st.color }
    });
    s.addText(st.val, {
      x: x + 0.1, y: 1.2, w: 2.02, h: 0.72,
      fontSize: 30, bold: true, color: st.color, align: "center", fontFace: "Calibri", margin: 0
    });
    s.addText(st.label, {
      x: x + 0.1, y: 1.93, w: 2.02, h: 0.4,
      fontSize: 9.5, color: GRAY, align: "center", fontFace: "Calibri", margin: 0
    });
  });

  // Table
  const rows = [
    [{ text: "Model",       options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "MAE",         options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "RMSE",        options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "MAPE (%)",    options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["ARIMA",               "0.8769", "1.0411", "350.87"],
    ["Vanilla RNN",         "0.0602", "0.0860", "34.05"],
    ["Stacked LSTM",        "0.0592", "0.0829", "31.32"],
    ["Temporal Transformer","0.0599", "0.0824", "30.80"],
  ];
  s.addTable(rows, {
    x: 0.3, y: 2.75, w: 9.4, h: 2.55,
    fontFace: "Calibri", fontSize: 11,
    align: "center",
    rowH: 0.46,
    border: { pt: 0.5, color: "D1D5DB" },
    fill: { color: WHITE },
    colW: [3.4, 2, 2, 2],
  });

  s.addText("Neural models outperform ARIMA by ~14× on MAE", {
    x: 0.3, y: 5.2, w: 9.4, h: 0.28,
    fontSize: 10, italic: true, color: BLUE, align: "center", fontFace: "Calibri", margin: 0
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 6 — Gaussian Noise
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Robustness to Gaussian Noise", "σ increases from 0 (clean) to 2.0 (noise std = data std × 2)");

  s.addImage({ path: FIG + "/mae_vs_gaussian_noise.png", x: 0.3, y: 1.05, w: 5.6, h: 3.1 });

  const findings = [
    { color: "4878D0", text: "ARIMA: flat curve (ignores test input) — input-immune but 14× worse on clean data" },
    { color: "EE854A", text: "RNN degrades fastest: +1,183% MAE at σ=2.0 — sequential hidden state corruption" },
    { color: "D65F5F", text: "Transformer most robust at σ ≤ 0.5 — attention dilutes localised corruptions" },
    { color: "6ACC65", text: "LSTM overtakes Transformer at σ ≥ 1.0 — gating mechanism filters extreme noise better" },
  ];
  findings.forEach((f, i) => {
    const y = 1.1 + i * 0.82;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.1, y, w: 3.65, h: 0.72,
      fill: { color: WHITE }, line: { color: "E2E8F0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.1, y, w: 0.07, h: 0.72,
      fill: { color: f.color }, line: { color: f.color }
    });
    s.addText(f.text, {
      x: 6.25, y: y + 0.05, w: 3.4, h: 0.62,
      fontSize: 9.5, color: GRAY, fontFace: "Calibri", margin: 0
    });
  });

  // Mini table
  s.addTable([
    [{ text: "Model", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "σ=0.25", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "σ=0.5", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "σ=2.0", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["RNN",         "0.1408","0.2516","0.7723"],
    ["LSTM",        "0.1078","0.1811","0.5313"],
    ["Transformer", "0.0971","0.1600","0.6035"],
  ], {
    x: 0.3, y: 4.28, w: 5.6, h: 1.15,
    fontFace: "Calibri", fontSize: 10, align: "center",
    rowH: 0.27, colW: [1.9, 1.2, 1.2, 1.3],
    border: { pt: 0.4, color: "D1D5DB" }, fill: { color: WHITE },
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 7 — Point Anomalies
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Robustness to Point Anomalies", "Random positions replaced with extreme outliers N(0, 25)");

  s.addImage({ path: FIG + "/mae_vs_anomaly_ratio.png", x: 0.3, y: 1.05, w: 5.6, h: 3.1 });

  const findings = [
    { color: "D65F5F", text: "Transformer leads at r ≤ 5%: attention downweights isolated outliers effectively" },
    { color: "6ACC65", text: "LSTM competitive at r ≥ 10%: gating provides better heavy-outlier filtering" },
    { color: "EE854A", text: "RNN most vulnerable: at r=20% nearly matches ARIMA (0.822 vs 0.877)" },
    { color: "4878D0", text: "ARIMA: again flat — trivially immune but cannot exploit multivariate context" },
  ];
  findings.forEach((f, i) => {
    const y = 1.1 + i * 0.82;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.1, y, w: 3.65, h: 0.72,
      fill: { color: WHITE }, line: { color: "E2E8F0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.1, y, w: 0.07, h: 0.72,
      fill: { color: f.color }, line: { color: f.color }
    });
    s.addText(f.text, {
      x: 6.25, y: y + 0.05, w: 3.4, h: 0.62,
      fontSize: 9.5, color: GRAY, fontFace: "Calibri", margin: 0
    });
  });

  s.addTable([
    [{ text: "Model", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "r=0.01", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "r=0.05", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "r=0.20", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["RNN",         "0.1692","0.4386","0.8219"],
    ["LSTM",        "0.1220","0.3000","0.6894"],
    ["Transformer", "0.1116","0.2865","0.7322"],
  ], {
    x: 0.3, y: 4.28, w: 5.6, h: 1.15,
    fontFace: "Calibri", fontSize: 10, align: "center",
    rowH: 0.27, colW: [1.9, 1.2, 1.2, 1.3],
    border: { pt: 0.4, color: "D1D5DB" }, fill: { color: WHITE },
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 8 — MC-Dropout Uncertainty
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, GREEN);
  addSlideTitle(s, "MC-Dropout as a Data-Quality Detector", "LSTM: 50 forward passes with dropout active → predictive std");

  s.addImage({ path: FIG + "/mc_dropout_bands.png", x: 0.3, y: 1.05, w: 5.8, h: 3.3 });
  s.addText("Clean test set: LSTM mean ± 2σ uncertainty bands", {
    x: 0.3, y: 4.4, w: 5.8, h: 0.22,
    fontSize: 9, italic: true, color: GRAY, align: "center", fontFace: "Calibri", margin: 0
  });

  // MC-std table
  s.addText("Mean MC-Dropout Std vs. Corruption Level", {
    x: 6.25, y: 1.05, w: 3.5, h: 0.35,
    fontSize: 11, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
  });
  s.addTable([
    [{ text: "Type", options: { bold: true, color: WHITE, fill: { color: "047857" } } },
     { text: "Level 0", options: { bold: true, color: WHITE, fill: { color: "047857" } } },
     { text: "Level 2", options: { bold: true, color: WHITE, fill: { color: "047857" } } },
     { text: "Level 4", options: { bold: true, color: WHITE, fill: { color: "047857" } } }],
    ["Gaussian",    "0.0305","0.0424","0.0868"],
    ["Anomaly",     "0.0308","0.0564","0.0799"],
  ], {
    x: 6.25, y: 1.45, w: 3.5, h: 0.9,
    fontFace: "Calibri", fontSize: 10, align: "center",
    rowH: 0.28, colW: [1.1, 0.8, 0.8, 0.8],
    border: { pt: 0.4, color: "D1D5DB" }, fill: { color: WHITE },
  });

  const insights = [
    "Uncertainty INCREASES monotonically with corruption severity",
    "Works without labelled anomaly data — zero extra cost",
    "Practical use: if MC-std > threshold → flag forecast as unreliable",
    "Can trigger fallback to ARIMA (immune model) when activated",
  ];
  insights.forEach((txt, i) => {
    const y = 2.55 + i * 0.68;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.25, y, w: 3.5, h: 0.58,
      fill: { color: "F0FDF4" }, line: { color: "A7F3D0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.25, y, w: 0.06, h: 0.58,
      fill: { color: GREEN }, line: { color: GREEN }
    });
    s.addText(txt, {
      x: 6.38, y: y + 0.06, w: 3.28, h: 0.46,
      fontSize: 9.5, color: "065F46", fontFace: "Calibri", margin: 0
    });
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 9 — Ablation Study
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, ACCENT);
  addSlideTitle(s, "Ablation Study — Transformer Components", "Which design choices contribute to robustness?");

  s.addTable([
    [{ text: "Variant", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "Clean", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "σ=0.5", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "σ=1.0", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "Anomaly 5%", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    [{ text: "Full model (proposed)", options: { bold: true, color: "1D4ED8" } }, "0.0715","0.1599","0.2916","0.2877"],
    ["No positional enc.",  "0.0927","0.1932","0.3278","0.2693"],
    ["1 encoder layer",     "0.0726","0.1724","0.3028","0.2897"],
    ["d_model = 32",        "0.0818","0.1589","0.2793","0.2592"],
    ["No dropout",          "0.0618","0.1734","0.3158","0.2673"],
  ], {
    x: 0.3, y: 1.05, w: 9.4, h: 2.7,
    fontFace: "Calibri", fontSize: 11, align: "center",
    rowH: 0.42, colW: [3.5, 1.4, 1.4, 1.4, 1.7],
    border: { pt: 0.4, color: "D1D5DB" }, fill: { color: WHITE },
  });

  const findings = [
    { emoji: "1️⃣", color: RED,   title: "Positional encoding is #1", text: "Removing PE → +30% MAE on clean, worst robustness. Temporal ordering is critical for weather." },
    { emoji: "2️⃣", color: BLUE,  title: "Dropout = robustness", text: "No dropout wins on clean (0.0618) but collapses under noise (0.3158). Dropout prevents overfitting corruption." },
    { emoji: "3️⃣", color: GREEN, title: "Smaller can be better", text: "d_model=32 beats full model under corruption! Smaller models generalize better to noisy inputs." },
  ];
  findings.forEach((f, i) => {
    const x = 0.3 + i * 3.17;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.95, w: 3.0, h: 1.5,
      fill: { color: WHITE }, line: { color: "E2E8F0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.95, w: 3.0, h: 0.08,
      fill: { color: f.color }, line: { color: f.color }
    });
    s.addText(f.emoji + "  " + f.title, {
      x: x + 0.12, y: 4.05, w: 2.76, h: 0.32,
      fontSize: 10.5, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(f.text, {
      x: x + 0.12, y: 4.38, w: 2.76, h: 0.95,
      fontSize: 9.5, color: GRAY, fontFace: "Calibri", margin: 0
    });
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 10 — Error Analysis
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addTopBar(s, BLUE);
  addSlideTitle(s, "Error Analysis — When Do Models Fail?", "Failure cases: error distribution, temporal pattern, clean vs corrupted");

  s.addImage({ path: FIG + "/error_distributions.png", x: 0.3, y: 1.05, w: 5.8, h: 2.6 });
  s.addText("Error distributions: right-skewed, most predictions accurate (median |e| ≈ 0.043)", {
    x: 0.3, y: 3.7, w: 5.8, h: 0.22, fontSize: 9, italic: true, color: GRAY, align: "center", fontFace: "Calibri", margin: 0
  });

  s.addImage({ path: FIG + "/clean_vs_corrupted_mae.png", x: 0.3, y: 3.98, w: 5.8, h: 1.5 });

  const boxes = [
    { title: "Worst Predictions", text: "Cluster at temperature turning points — rapid direction changes after plateaus. All models share this failure mode." },
    { title: "Temporal Pattern", text: "MAE broadly stable across test set. Slight increase (~10%) at end — possible seasonality mismatch." },
    { title: "P90 Error", text: "RNN: 0.1315 | LSTM: 0.1296 | Transformer: 0.1264. Transformer has most concentrated tail." },
  ];
  boxes.forEach((b, i) => {
    const y = 1.1 + i * 1.48;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.3, y, w: 3.45, h: 1.28,
      fill: { color: WHITE }, line: { color: "E2E8F0" }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.3, y, w: 0.07, h: 1.28,
      fill: { color: BLUE }, line: { color: BLUE }
    });
    s.addText(b.title, {
      x: 6.44, y: y + 0.1, w: 3.2, h: 0.3,
      fontSize: 10.5, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(b.text, {
      x: 6.44, y: y + 0.42, w: 3.2, h: 0.76,
      fontSize: 9.5, color: GRAY, fontFace: "Calibri", margin: 0
    });
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 11 — Conclusion
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = darkSlide(pres);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: ACCENT }, line: { color: ACCENT }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.225, w: 10, h: 0.4, fill: { color: NAVY }, line: { color: NAVY }
  });

  s.addText("Conclusions & Key Takeaways", {
    x: 0.6, y: 0.25, w: 9.1, h: 0.55,
    fontSize: 24, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 0.85, w: 2.2, h: 0.07,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const conclusions = [
    { num: "01", text: "All neural models achieve MAE ≈ 0.06 on clean data — ~14× better than ARIMA", color: "6ACC65" },
    { num: "02", text: "Transformer is most robust under LOW-TO-MODERATE corruption (σ ≤ 0.5, r ≤ 5%)", color: "D65F5F" },
    { num: "03", text: "LSTM gating outperforms Transformer under HEAVY corruption (σ ≥ 1.0, r ≥ 10%)", color: "6ACC65" },
    { num: "04", text: "MC-Dropout uncertainty increases with corruption — useful as data-quality indicator", color: "93C5FD" },
    { num: "05", text: "Positional encoding is the most critical Transformer component for robustness", color: "FCD34D" },
  ];

  conclusions.forEach((c, i) => {
    const y = 1.05 + i * 0.82;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 0.5, h: 0.62,
      fill: { color: c.color, transparency: 20 }, line: { color: c.color, transparency: 40 }
    });
    s.addText(c.num, {
      x: 0.6, y, w: 0.5, h: 0.62,
      fontSize: 16, bold: true, color: c.color, align: "center", valign: "middle", fontFace: "Calibri", margin: 0
    });
    s.addText(c.text, {
      x: 1.25, y: y + 0.1, w: 8.4, h: 0.45,
      fontSize: 12, color: "E2E8F0", fontFace: "Calibri", margin: 0
    });
  });

  s.addText("Future work: attention-based anomaly masking · adversarial training · multi-step forecasting", {
    x: 0.6, y: 5.0, w: 9.1, h: 0.22,
    fontSize: 9, italic: true, color: "94A3B8", fontFace: "Calibri", margin: 0
  });
}

// ────────────────────────────────────────────────────────────────────────────────
// SLIDE 12 — Thank You / Q&A
// ────────────────────────────────────────────────────────────────────────────────
{
  const s = darkSlide(pres);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.12, fill: { color: BLUE }, line: { color: BLUE }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.505, w: 10, h: 0.12, fill: { color: BLUE }, line: { color: BLUE }
  });

  s.addText("Thank You!", {
    x: 1, y: 0.8, w: 8, h: 1.0,
    fontSize: 52, bold: true, color: WHITE, align: "center", fontFace: "Calibri", margin: 0
  });
  s.addText("Questions & Discussion", {
    x: 1, y: 1.75, w: 8, h: 0.6,
    fontSize: 22, color: "93C5FD", align: "center", fontFace: "Calibri", margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 3.5, y: 2.5, w: 3.0, h: 0.07,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  s.addText([
    { text: "Toprak Zeybek  |  323011022", options: { breakLine: true } },
    { text: "SEDS 537 Machine Learning  ·  Spring 2026", options: { breakLine: true } },
    { text: "Izmir Institute of Technology  ·  Prof. Dr. Aytug Onan", options: {} }
  ], {
    x: 1, y: 2.75, w: 8, h: 0.9,
    fontSize: 12, color: "64748B", align: "center", fontFace: "Calibri", margin: 0
  });

  const summary = ["4 Models", "2 Corruption Types", "13 Figures", "Ablation Study"];
  summary.forEach((t, i) => {
    const x = 1.2 + i * 1.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.9, w: 1.7, h: 1.1,
      fill: { color: NAVY }, line: { color: BLUE }
    });
    s.addText(t, {
      x, y: 3.9, w: 1.7, h: 1.1,
      fontSize: 13, bold: true, color: BLUE, align: "center", valign: "middle", fontFace: "Calibri", margin: 0
    });
  });
}

// ── Write ──────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Done -> " + OUT);
}).catch(e => { console.error(e); process.exit(1); });
