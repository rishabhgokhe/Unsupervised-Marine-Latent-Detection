/**
 * Marine Latent - Interactive Web Engine
 * Controls: Ocean currents canvas, live regime telemetry simulator,
 * early warning risk calculator, and tab interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
  initOceanCanvas();
  initRegimeSimulator();
  initRiskCalculator();
  initPipelineTabs();
  initMatrixHover();
  checkDashboardStatus();
});

/* -----------------------------------------------------------
 * 1. Ambient Ocean Currents Background Canvas
 * ----------------------------------------------------------- */
function initOceanCanvas() {
  const canvas = document.getElementById('ocean-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  const particleCount = Math.min(width > 768 ? 65 : 30, 80);

  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 2.5 + 0.8,
      speedX: (Math.random() * 0.4 + 0.15) * (Math.random() > 0.5 ? 1 : -1),
      speedY: Math.random() * 0.35 + 0.1,
      opacity: Math.random() * 0.5 + 0.15,
      hue: Math.random() > 0.6 ? 186 : 172, // Cyan to Seafoam teal
    });
  }

  let step = 0;
  function animate() {
    ctx.clearRect(0, 0, width, height);
    step += 0.008;

    // Draw flowing connections between nearby nodes
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.speedX + Math.sin(step + p.y * 0.005) * 0.4;
      p.y -= p.speedY;

      if (p.y < -10) {
        p.y = height + 10;
        p.x = Math.random() * width;
      }
      if (p.x < -10) p.x = width + 10;
      if (p.x > width + 10) p.x = -10;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 90%, 65%, ${p.opacity})`;
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 110) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.12 * (1 - dist / 110)})`;
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
}

/* -----------------------------------------------------------
 * 2. Interactive Regime Simulator & Live Telemetry Chart
 * ----------------------------------------------------------- */
const REGIME_DATA = {
  calm: {
    id: 0,
    name: 'Calm Coastal Sea',
    type: 'Low Energy State',
    badgeClass: 'badge-calm',
    wave: 0.85,
    waveUnit: 'm',
    wind: 5.2,
    windUnit: 'kts',
    pres: 1018.4,
    presUnit: 'hPa',
    sst: 19.8,
    sstUnit: '°C',
    riskLevel: 'Low',
    riskColor: '#10b981',
    duration: '142h avg',
    entropy: '0.24',
    waveAmplitude: 12,
    waveFreq: 0.02,
    waveNoise: 1.5,
    description: 'Minimal wave height, light breezes, and high atmospheric pressure. Optimal conditions for marine transit and maintenance operations.',
    features: {
      'WAVE_s24_mean': '0.85 m',
      'WIND_s24_mean': '5.20 kts',
      'PRES_s24_mean': '1018.4 hPa',
      'WAVE_s72_trend': '-0.02',
      'ENERGY_s24': '0.74 m²',
    }
  },
  moderate: {
    id: 1,
    name: 'Moderate Ocean Swell',
    type: 'Equilibrium State',
    badgeClass: 'badge-moderate',
    wave: 2.35,
    waveUnit: 'm',
    wind: 16.8,
    windUnit: 'kts',
    pres: 1012.0,
    presUnit: 'hPa',
    sst: 18.2,
    sstUnit: '°C',
    riskLevel: 'Medium',
    riskColor: '#0ea5e9',
    duration: '98h avg',
    entropy: '0.41',
    waveAmplitude: 28,
    waveFreq: 0.035,
    waveNoise: 4.0,
    description: 'Steady trade wind swells and oceanic equilibrium. Noticeable wave periods with persistent moderate wave energy.',
    features: {
      'WAVE_s24_mean': '2.35 m',
      'WIND_s24_mean': '16.80 kts',
      'PRES_s24_mean': '1012.0 hPa',
      'WAVE_s72_trend': '+0.15',
      'ENERGY_s24': '5.62 m²',
    }
  },
  storm: {
    id: 2,
    name: 'Severe Storm / Cyclone',
    type: 'Critical Divergence',
    badgeClass: 'badge-storm',
    wave: 5.82,
    waveUnit: 'm',
    wind: 38.4,
    windUnit: 'kts',
    pres: 989.2,
    presUnit: 'hPa',
    sst: 16.4,
    sstUnit: '°C',
    riskLevel: 'Critical',
    riskColor: '#ef4444',
    duration: '34h avg',
    entropy: '0.68',
    waveAmplitude: 52,
    waveFreq: 0.06,
    waveNoise: 9.5,
    description: 'Steep pressure drops accompany gale-force winds and towering sea states. Elevated risk triggers automated warning alerts for offshore assets.',
    features: {
      'WAVE_s24_mean': '5.82 m',
      'WIND_s24_mean': '38.40 kts',
      'PRES_s24_mean': '989.2 hPa',
      'WAVE_s72_trend': '+1.42',
      'ENERGY_s24': '34.80 m²',
    }
  },
  transitional: {
    id: 3,
    name: 'High-Variance Front',
    type: 'Dynamic Boundary',
    badgeClass: 'badge-variance',
    wave: 3.45,
    waveUnit: 'm',
    wind: 27.1,
    windUnit: 'kts',
    pres: 1003.5,
    presUnit: 'hPa',
    sst: 17.5,
    sstUnit: '°C',
    riskLevel: 'Elevated',
    riskColor: '#f59e0b',
    duration: '48h avg',
    entropy: '0.59',
    waveAmplitude: 38,
    waveFreq: 0.048,
    waveNoise: 7.0,
    description: 'Rapid weather front passing through buoy arrays. Statistical changepoints align with sudden directional wind shifts and pressure volatility.',
    features: {
      'WAVE_s24_mean': '3.45 m',
      'WIND_s24_mean': '27.10 kts',
      'PRES_s24_mean': '1003.5 hPa',
      'WAVE_s72_trend': '-0.88',
      'ENERGY_s24': '12.40 m²',
    }
  }
};

let currentRegimeKey = 'calm';
let simCanvas, simCtx;
let simPhase = 0;

function initRegimeSimulator() {
  simCanvas = document.getElementById('telemetry-canvas');
  if (!simCanvas) return;
  simCtx = simCanvas.getContext('2d');

  function resizeSim() {
    const rect = simCanvas.parentElement.getBoundingClientRect();
    simCanvas.width = rect.width;
    simCanvas.height = 200;
  }
  resizeSim();
  window.addEventListener('resize', resizeSim);

  // Tab button events
  const buttons = document.querySelectorAll('.regime-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.dataset.regime;
      updateRegimeDisplay(key);
    });
  });

  updateRegimeDisplay('calm');
  requestAnimationFrame(drawWaveform);
}

function updateRegimeDisplay(key) {
  currentRegimeKey = key;
  const d = REGIME_DATA[key];
  if (!d) return;

  // Update text & indicators
  const badge = document.getElementById('regime-badge');
  if (badge) {
    badge.className = `status-badge ${d.badgeClass}`;
    badge.textContent = d.name;
  }

  const desc = document.getElementById('regime-desc');
  if (desc) desc.textContent = d.description;

  const valWave = document.getElementById('stat-wave');
  if (valWave) valWave.textContent = `${d.wave} ${d.waveUnit}`;

  const valWind = document.getElementById('stat-wind');
  if (valWind) valWind.textContent = `${d.wind} ${d.windUnit}`;

  const valPres = document.getElementById('stat-pres');
  if (valPres) valPres.textContent = `${d.pres} ${d.presUnit}`;

  const valRisk = document.getElementById('stat-risk');
  if (valRisk) {
    valRisk.textContent = d.riskLevel;
    valRisk.style.color = d.riskColor;
  }

  const valDur = document.getElementById('stat-duration');
  if (valDur) valDur.textContent = d.duration;

  const valEnt = document.getElementById('stat-entropy');
  if (valEnt) valEnt.textContent = d.entropy;

  // Update feature vector breakdown chips
  const featGrid = document.getElementById('feature-vector-chips');
  if (featGrid) {
    featGrid.innerHTML = '';
    for (const [feat, val] of Object.entries(d.features)) {
      const chip = document.createElement('div');
      chip.className = 'feature-chip';
      chip.innerHTML = `<span class="feat-name">${feat}</span><span class="feat-val">${val}</span>`;
      featGrid.appendChild(chip);
    }
  }
}

function drawWaveform() {
  if (!simCtx || !simCanvas) return;
  const w = simCanvas.width;
  const h = simCanvas.height;
  const d = REGIME_DATA[currentRegimeKey];

  simCtx.clearRect(0, 0, w, h);
  simPhase += 0.035;

  // Grid background
  simCtx.strokeStyle = 'rgba(14, 165, 233, 0.08)';
  simCtx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    simCtx.beginPath();
    simCtx.moveTo(x, 0);
    simCtx.lineTo(x, h);
    simCtx.stroke();
  }
  for (let y = 0; y < h; y += 35) {
    simCtx.beginPath();
    simCtx.moveTo(0, y);
    simCtx.lineTo(w, y);
    simCtx.stroke();
  }

  // Draw simulated Wave Height curve
  const midY = h * 0.55;
  simCtx.beginPath();
  simCtx.moveTo(0, midY);

  for (let x = 0; x <= w; x += 3) {
    const harmonic1 = Math.sin(x * d.waveFreq + simPhase) * d.waveAmplitude;
    const harmonic2 = Math.sin(x * (d.waveFreq * 2.3) - simPhase * 1.2) * (d.waveAmplitude * 0.35);
    const noise = Math.sin(x * 0.15 + simPhase * 3) * d.waveNoise;
    const y = midY + harmonic1 + harmonic2 + noise;
    simCtx.lineTo(x, y);
  }

  // Gradient fill under the curve
  const grad = simCtx.createLinearGradient(0, 0, 0, h);
  if (currentRegimeKey === 'storm') {
    grad.addColorStop(0, 'rgba(239, 68, 68, 0.45)');
    grad.addColorStop(1, 'rgba(239, 68, 68, 0.0)');
    simCtx.strokeStyle = '#ef4444';
  } else if (currentRegimeKey === 'transitional') {
    grad.addColorStop(0, 'rgba(245, 158, 11, 0.4)');
    grad.addColorStop(1, 'rgba(245, 158, 11, 0.0)');
    simCtx.strokeStyle = '#f59e0b';
  } else {
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
    grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
    simCtx.strokeStyle = '#00f0ff';
  }

  simCtx.lineWidth = 2.4;
  simCtx.stroke();

  simCtx.lineTo(w, h);
  simCtx.lineTo(0, h);
  simCtx.fillStyle = grad;
  simCtx.fill();

  requestAnimationFrame(drawWaveform);
}

/* -----------------------------------------------------------
 * 3. Early Warning Risk Calculator Simulator
 * ----------------------------------------------------------- */
function initRiskCalculator() {
  const currentSevSlider = document.getElementById('calc-current-sev');
  const nextProbSlider = document.getElementById('calc-next-prob');
  const scoreEl = document.getElementById('calc-score-display');
  const gaugeEl = document.getElementById('calc-gauge-fill');
  const pillEl = document.getElementById('calc-risk-pill');

  if (!currentSevSlider || !nextProbSlider || !scoreEl) return;

  function update() {
    const cur = parseFloat(currentSevSlider.value) / 100;
    const prob = parseFloat(nextProbSlider.value) / 100;

    // Mathematical formula from app/ui_helpers.py:L251
    // Risk Score = 0.7 * Current Severity + 0.3 * P(Next Regime High)
    const risk = 0.7 * cur + 0.3 * prob;
    const pct = Math.round(risk * 100);

    scoreEl.textContent = `${(risk).toFixed(2)} (${pct}%)`;
    if (gaugeEl) gaugeEl.style.width = `${pct}%`;

    const curValEl = document.getElementById('calc-cur-val');
    if (curValEl) curValEl.textContent = `${Math.round(cur * 100)}%`;
    const probValEl = document.getElementById('calc-prob-val');
    if (probValEl) probValEl.textContent = `${Math.round(prob * 100)}%`;

    if (pillEl) {
      if (risk >= 0.66) {
        pillEl.textContent = 'HIGH RISK';
        pillEl.className = 'pill-status pill-danger';
        gaugeEl.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
      } else if (risk >= 0.33) {
        pillEl.textContent = 'MODERATE';
        pillEl.className = 'pill-status pill-warning';
        gaugeEl.style.background = 'linear-gradient(90deg, #0ea5e9, #f59e0b)';
      } else {
        pillEl.textContent = 'LOW RISK';
        pillEl.className = 'pill-status pill-success';
        gaugeEl.style.background = 'linear-gradient(90deg, #10b981, #0ea5e9)';
      }
    }
  }

  currentSevSlider.addEventListener('input', update);
  nextProbSlider.addEventListener('input', update);
  update();
}

/* -----------------------------------------------------------
 * 4. Interactive Pipeline Stepper
 * ----------------------------------------------------------- */
function initPipelineTabs() {
  const stepBtns = document.querySelectorAll('.pipeline-step-btn');
  const stepPanels = document.querySelectorAll('.pipeline-card');

  stepBtns.forEach((btn, idx) => {
    btn.addEventListener('click', () => {
      stepBtns.forEach(b => b.classList.remove('active'));
      stepPanels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      if (stepPanels[idx]) stepPanels[idx].classList.add('active');
    });
  });
}

/* -----------------------------------------------------------
 * 5. Transition Matrix Hover Inspector
 * ----------------------------------------------------------- */
function initMatrixHover() {
  const cells = document.querySelectorAll('.matrix-cell');
  const tooltip = document.getElementById('matrix-tooltip');
  if (!tooltip) return;

  cells.forEach(cell => {
    cell.addEventListener('mouseenter', () => {
      const from = cell.dataset.from;
      const to = cell.dataset.to;
      const val = cell.dataset.prob;
      tooltip.textContent = `P(${from} → ${to}) = ${val} (${(parseFloat(val) * 100).toFixed(1)}%)`;
      tooltip.style.opacity = '1';
    });
    cell.addEventListener('mouseleave', () => {
      tooltip.style.opacity = '0.7';
      tooltip.textContent = 'Hover over any transition cell to inspect conditional probability';
    });
  });
}

/* -----------------------------------------------------------
 * 6. Live Dashboard Health Checker (Port 8501)
 * ----------------------------------------------------------- */
function checkDashboardStatus() {
  const badge = document.getElementById('dashboard-status-indicator');
  if (!badge) return;

  fetch('http://localhost:8501/_stcore/health', { mode: 'no-cors' })
    .then(() => {
      badge.innerHTML = '<span class="status-dot online"></span> Streamlit Live (:8501)';
      badge.classList.add('online');
    })
    .catch(() => {
      badge.innerHTML = '<span class="status-dot offline"></span> Dashboard Offline (run `make app`)';
      badge.classList.remove('online');
    });
}
