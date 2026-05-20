const AQI_META = [
    { label: "Good", min: 0, max: 50, color: "#00c853", class: "good" },
    { label: "Satisfactory", min: 51, max: 100, color: "#69b34c", class: "satisfactory" },
    { label: "Moderate", min: 101, max: 200, color: "#fab733", class: "moderate" },
    { label: "Poor", min: 201, max: 300, color: "#ff6600", class: "poor" },
    { label: "Very Poor", min: 301, max: 400, color: "#e63946", class: "very-poor" },
    { label: "Severe", min: 401, max: 500, color: "#7b2d8b", class: "severe" }
];

function getAqiMeta(aqi) {
    for (const meta of AQI_META) {
        if (aqi >= meta.min && aqi <= meta.max) return meta;
    }
    return AQI_META[AQI_META.length - 1];
}

let mockCities = {};

const stateSelect = document.getElementById("state-select");
const citySelect = document.getElementById("city-select");
const runBtn = document.getElementById("run-prediction-btn");
const resultsDashboard = document.getElementById("results-dashboard");
const loadingState = document.getElementById("loading-state");
let aqiChartInstance = null;
let histChartInstance = null;

// Initialize States Dropdown from API
fetch('/api/locations')
    .then(res => res.json())
    .then(data => {
        mockCities = data;
        const sortedStates = Object.keys(mockCities).sort();
        sortedStates.forEach(state => {
            const opt = document.createElement("option");
            opt.value = state;
            opt.textContent = state;
            stateSelect.appendChild(opt);
        });
    })
    .catch(err => console.error("Error loading locations:", err));

// Populate cities on state change
stateSelect.addEventListener("change", (e) => {
    const state = e.target.value;
    const cities = mockCities[state] || [];
    citySelect.innerHTML = '<option value="" disabled selected>Select Area / City</option>';
    cities.forEach(city => {
        const opt = document.createElement("option");
        opt.value = city;
        opt.textContent = city;
        citySelect.appendChild(opt);
    });
});

// Sidebar selection (for popular.html)
document.querySelectorAll(".location-btn").forEach(btn => {
    btn.addEventListener("click", function() {
        document.querySelectorAll(".location-btn").forEach(b => b.classList.remove("active"));
        this.classList.add("active");
        const state = this.getAttribute("data-state");
        const city = this.getAttribute("data-city");
        
        stateSelect.value = state;
        stateSelect.dispatchEvent(new Event("change"));
        setTimeout(() => {
            citySelect.value = city;
            citySelect.dispatchEvent(new Event("change"));
        }, 50);
    });
});

// Sidebar & Auto-Predict update when City changes
citySelect.addEventListener("change", () => {
    const city = citySelect.value;
    if (city) {
        updateSidebarHistorical(city);
        
        // Auto-predict ONLY on the popular page (where the Run Prediction button is removed)
        if (document.body.getAttribute('data-page') === 'popular') {
            runPredictionLogic(stateSelect.value, city);
        }
    }
});

// Run Prediction (only if button exists - e.g. Home Page)
if (runBtn) {
    runBtn.addEventListener("click", () => {
        runPredictionLogic(stateSelect.value, citySelect.value);
    });
}

function runPredictionLogic(state, city) {
    if (!state || !city) {
        alert("Please select a State and City first.");
        return;
    }

    // Hide dashboard, show loading
    resultsDashboard.classList.add("hidden");
    resultsDashboard.classList.remove("fade-in-up");
    loadingState.classList.remove("hidden");
    
    document.getElementById("target-loc-display").innerHTML = `<i class="ph ph-map-pin"></i> ${city}, ${state}`;

    // Fetch real prediction from Python API
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: state, city: city })
    })
    .then(res => {
        if (!res.ok) throw new Error("Failed to fetch prediction");
        return res.json();
    })
    .then(data => {
        loadingState.classList.add("hidden");
        const forecast = data.forecast;
        
        renderCards(forecast);
        renderMetrics(forecast);
        renderChart(forecast);
        renderTable(forecast);
        
        resultsDashboard.classList.remove("hidden");
        resultsDashboard.classList.add("fade-in-up");
    })
    .catch(err => {
        console.error(err);
        loadingState.classList.add("hidden");
        alert("Error predicting AQI: " + err.message);
    });
}

function renderCards(forecast) {
    const container = document.getElementById("forecast-cards-container");
    container.innerHTML = "";
    
    forecast.forEach((day, index) => {
        const delay = index * 0.1;
        const html = `
            <div class="day-card ${day.meta.class}" style="animation: fadeInUp 0.5s ease ${delay}s forwards; opacity: 0;">
                <div class="day-label">${day.day}</div>
                <div class="date-label">${day.dateStr}</div>
                <div class="aqi-value">${day.aqi}</div>
                <div class="cat-label" style="color: ${day.meta.color}">${day.meta.label.toUpperCase()}</div>
            </div>
        `;
        container.innerHTML += html;
    });
}

function renderMetrics(forecast) {
    const aqis = forecast.map(f => f.aqi);
    const max = Math.max(...aqis);
    const min = Math.min(...aqis);
    const avg = Math.round(aqis.reduce((a, b) => a + b) / aqis.length);
    const maxMeta = getAqiMeta(max);

    document.getElementById("metric-max").textContent = max;
    document.getElementById("metric-min").textContent = min;
    document.getElementById("metric-avg").textContent = avg;
    
    const catEl = document.getElementById("metric-cat");
    catEl.textContent = maxMeta.label;
    catEl.style.color = maxMeta.color;
}

function renderTable(forecast) {
    const tbody = document.querySelector("#details-table tbody");
    tbody.innerHTML = "";
    
    forecast.forEach(day => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${day.day}</strong> <span style="color:#64748b; font-size:0.8rem; margin-left:6px">${day.dateStr}</span></td>
            <td><strong style="color:${day.meta.color}">${day.aqi}</strong></td>
            <td><span style="background:${day.meta.color}22; color:${day.meta.color}; padding:2px 8px; border-radius:4px; font-size:0.75rem">${day.meta.label}</span></td>
            <td>${day.pm25}</td>
            <td>${day.pm10}</td>
            <td>${day.no2}</td>
            <td>${day.o3}</td>
            <td>${day.so2}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderChart(forecast) {
    const ctx = document.getElementById('aqiChart').getContext('2d');
    
    if (aqiChartInstance) {
        aqiChartInstance.destroy();
    }
    
    const labels = forecast.map(f => f.day === 'TODAY' ? 'Today' : f.day);
    const data = forecast.map(f => f.aqi);
    
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Outfit';

    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    aqiChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Predicted AQI',
                data: data,
                borderColor: '#00f0ff',
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: '#0f172a',
                pointBorderColor: '#00f0ff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#00f0ff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(0, 240, 255, 0.3)',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    border: { display: false }
                },
                x: {
                    grid: { display: false },
                    border: { display: false }
                }
            }
        }
    });
}

// Sidebar Historical Updates
function updateSidebarHistorical(city) {
    const captionEl = document.getElementById("hist-caption");
    if (!captionEl) return; // Guard for popular.html where this sidebar doesn't exist

    captionEl.textContent = `Last 30 readings — ${city}`;
    
    // Generate mock historical data (30 days)
    const histData = [];
    let baseAqi = 150;
    if (city === "Delhi") baseAqi = 320;
    else if (city === "Bileipada") baseAqi = 220;
    
    for (let i = 0; i < 30; i++) {
        histData.push(Math.max(20, baseAqi + (Math.random() * 80 - 40)));
    }
    const latest = Math.round(histData[histData.length - 1]);
    const prev = Math.round(histData[histData.length - 2]);
    const delta = latest - prev;
    
    document.getElementById("latest-aqi-val").textContent = latest;
    
    const deltaEl = document.getElementById("latest-aqi-delta");
    if (delta > 0) {
        deltaEl.innerHTML = `↑ ${delta}`;
        deltaEl.style.background = 'rgba(230,57,70,0.2)';
        deltaEl.style.color = '#e63946';
    } else {
        deltaEl.innerHTML = `↓ ${Math.abs(delta)}`;
        deltaEl.style.background = 'rgba(0,200,83,0.2)';
        deltaEl.style.color = '#00c853';
    }

    renderHistoricalChart(histData);
}

function renderHistoricalChart(data) {
    const ctx = document.getElementById('historicalChart').getContext('2d');
    if (histChartInstance) histChartInstance.destroy();
    
    const labels = Array.from({length: 30}, (_, i) => i + 1);
    
    histChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                borderColor: '#8b5cf6',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            layout: { padding: 0 }
        }
    });
}
