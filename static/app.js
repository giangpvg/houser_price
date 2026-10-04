// JavaScript logic for Housing Price Prediction Dashboard
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const sliderArea = document.getElementById("slider-area");
  const inputArea = document.getElementById("input-area");
  const valArea = document.getElementById("val-area");

  const sliderBeds = document.getElementById("slider-beds");
  const inputBeds = document.getElementById("input-beds");
  const valBeds = document.getElementById("val-beds");

  const sliderBaths = document.getElementById("slider-baths");
  const inputBaths = document.getElementById("input-baths");
  const valBaths = document.getElementById("val-baths");

  const sliderDist = document.getElementById("slider-dist");
  const inputDist = document.getElementById("input-dist");
  const valDist = document.getElementById("val-dist");

  const sliderAge = document.getElementById("slider-age");
  const inputAge = document.getElementById("input-age");
  const valAge = document.getElementById("val-age");

  const form = document.getElementById("prediction-form");
  const btnPredict = document.getElementById("btn-predict");
  const btnRetrain = document.getElementById("btn-retrain");

  const heroPriceDisplay = document.getElementById("hero-price-display");
  const unitRateChip = document.getElementById("unit-rate-chip");
  const intervalChip = document.getElementById("interval-chip");
  const contributionList = document.getElementById("contribution-list");
  const liveFormulaCode = document.getElementById("live-formula-code");

  const metricR2 = document.getElementById("metric-r2");
  const metricMae = document.getElementById("metric-mae");
  const metricRmse = document.getElementById("metric-rmse");

  const presetBtns = document.querySelectorAll(".preset-btn");

  let modelMetadata = null;

  // Helper: Synchronize slider and number input
  function bindInputPair(slider, input, badge, isFloat = false) {
    const update = (val) => {
      slider.value = val;
      input.value = val;
      badge.textContent = isFloat ? parseFloat(val).toFixed(1) : val;
    };

    slider.addEventListener("input", (e) => {
      update(e.target.value);
      debouncePredict();
    });

    input.addEventListener("input", (e) => {
      update(e.target.value);
      debouncePredict();
    });
  }

  bindInputPair(sliderArea, inputArea, valArea, false);
  bindInputPair(sliderBeds, inputBeds, valBeds, false);
  bindInputPair(sliderBaths, inputBaths, valBaths, false);
  bindInputPair(sliderDist, inputDist, valDist, true);
  bindInputPair(sliderAge, inputAge, valAge, true);

  // Debounce for smooth live prediction while dragging slider
  let debounceTimer = null;
  function debouncePredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      executePrediction();
    }, 250);
  }

  // Handle Preset selection
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const area = btn.getAttribute("data-area");
      const beds = btn.getAttribute("data-beds");
      const baths = btn.getAttribute("data-baths");
      const dist = btn.getAttribute("data-dist");
      const age = btn.getAttribute("data-age");

      sliderArea.value = area;
      inputArea.value = area;
      valArea.textContent = area;

      sliderBeds.value = beds;
      inputBeds.value = beds;
      valBeds.textContent = beds;

      sliderBaths.value = baths;
      inputBaths.value = baths;
      valBaths.textContent = baths;

      sliderDist.value = dist;
      inputDist.value = dist;
      valDist.textContent = dist;

      sliderAge.value = age;
      inputAge.value = age;
      valAge.textContent = age;

      executePrediction();
    });
  });

  // Fetch initial model metadata and metrics
  async function loadModelInfo() {
    try {
      const res = await fetch("/api/model/info");
      if (!res.ok) throw new Error("Không thể tải thông tin mô hình");
      const data = await res.json();
      modelMetadata = data.metadata;

      if (modelMetadata && modelMetadata.metrics) {
        metricR2.textContent = modelMetadata.metrics.r2_score.toFixed(4);
        metricMae.textContent = `~${modelMetadata.metrics.mae_billion_vnd.toFixed(3)} Tỷ`;
        metricRmse.textContent = `~${modelMetadata.metrics.rmse_billion_vnd.toFixed(3)} Tỷ`;
      }
      executePrediction();
    } catch (err) {
      console.error("Lỗi khi tải metadata mô hình:", err);
    }
  }

  // Execute Prediction API
  async function executePrediction() {
    const payload = {
      area_m2: parseFloat(inputArea.value),
      bedrooms: parseInt(inputBeds.value, 10),
      bathrooms: parseInt(inputBaths.value, 10),
      distance_to_center_km: parseFloat(inputDist.value),
      house_age_years: parseFloat(inputAge.value),
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Lỗi dự đoán");
      }

      const data = await res.json();
      renderPredictionResult(data);
    } catch (err) {
      console.error("Lỗi dự đoán:", err);
      heroPriceDisplay.textContent = "Lỗi tính toán";
    }
  }

  // Render prediction UI & Waterfall explanation
  function renderPredictionResult(data) {
    heroPriceDisplay.textContent = data.formatted_price;
    unitRateChip.textContent = `~ ${data.formatted_price_per_m2}`;
    intervalChip.textContent = `Ước tính 95%: ${data.prediction_interval.low_billion_vnd.toFixed(3)} - ${data.prediction_interval.high_billion_vnd.toFixed(3)} Tỷ VNĐ`;

    const c = data.feature_contributions;
    const items = [
      {
        name: "🏢 Đất nền cơ bản (Intercept)",
        val: c.base_intercept_billion,
        type: "base",
      },
      {
        name: `📐 Diện tích (${data.input_features.area_m2} m²)`,
        val: c.area_contribution_billion,
        type: "positive",
      },
      {
        name: `🛏️ Số phòng ngủ (${data.input_features.bedrooms} PN)`,
        val: c.bedrooms_contribution_billion,
        type: "positive",
      },
      {
        name: `🚿 Số phòng tắm (${data.input_features.bathrooms} WC)`,
        val: c.bathrooms_contribution_billion,
        type: "positive",
      },
      {
        name: `📍 Khoảng cách trung tâm (${data.input_features.distance_to_center_km} km)`,
        val: c.distance_contribution_billion,
        type: "negative",
      },
      {
        name: `⏳ Khấu hao tuổi nhà (${data.input_features.house_age_years} năm)`,
        val: c.age_contribution_billion,
        type: "negative",
      },
    ];

    contributionList.innerHTML = "";
    items.forEach((item) => {
      const el = document.createElement("div");
      el.className = `contrib-item ${item.type}`;

      const sign = item.val >= 0 ? "+" : "";
      const valClass =
        item.type === "base" ? "base" : item.val >= 0 ? "pos" : "neg";

      el.innerHTML = `
        <span class="contrib-name">${item.name}</span>
        <span class="contrib-val ${valClass}">${sign}${item.val.toFixed(3)} Tỷ VNĐ</span>
      `;
      contributionList.appendChild(el);
    });

    // Update real-time mathematical formula
    liveFormulaCode.textContent =
      `y = ${c.base_intercept_billion.toFixed(3)} ` +
      `+ (${c.area_contribution_billion.toFixed(3)}) ` +
      `+ (${c.bedrooms_contribution_billion.toFixed(3)}) ` +
      `+ (${c.bathrooms_contribution_billion.toFixed(3)}) ` +
      `+ (${c.distance_contribution_billion.toFixed(3)}) ` +
      `+ (${c.age_contribution_billion.toFixed(3)}) ` +
      `= ${data.predicted_price_billion_vnd.toFixed(3)} Tỷ VNĐ`;
  }

  // Handle Form Submit
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    executePrediction();
  });

  // Handle Retrain Button
  btnRetrain.addEventListener("click", async () => {
    if (!confirm("Bạn có muốn tái huấn luyện mô hình Linear Regression với bộ dữ liệu mới?")) return;
    btnRetrain.disabled = true;
    btnRetrain.textContent = "⏳ Đang huấn luyện...";

    try {
      const res = await fetch("/api/model/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_samples: 1500, random_state: Math.floor(Math.random() * 1000) }),
      });
      if (!res.ok) throw new Error("Tái huấn luyện thất bại");
      alert("✅ Tái huấn luyện mô hình thành công!");
      await loadModelInfo();
    } catch (err) {
      alert(`❌ Lỗi: ${err.message}`);
    } finally {
      btnRetrain.disabled = false;
      btnRetrain.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
        Tái huấn luyện
      `;
    }
  });

  // Initial load
  loadModelInfo();
});
