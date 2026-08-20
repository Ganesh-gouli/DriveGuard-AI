/* backend/static/script.js */

// Global Chart Instances
let drowsinessChart = null;
let radarChart = null;

// Global State
let breakAlertDismissed = false;
let seatbeltSpeechInterval = null;

// --- Initialization ---
function initDashboard() {
    initCharts();

    // Start Polling
    setInterval(fetchLiveData, 1000); // 1 sec for live data
    setInterval(fetchAlerts, 2000);   // 2 sec for alerts

    // Initial Fetch
    fetchLiveData();
    fetchAlerts();

    // Update Profile Info (Mock)
    const name = localStorage.getItem('driverName') || 'Driver';
    const car = localStorage.getItem('carNumber') || 'XX-00-XX-0000';

    const nameEls = document.querySelectorAll('#userName, #profileName');
    nameEls.forEach(el => el.textContent = name);

    const carEls = document.querySelectorAll('#userCar, #profileCar');
    carEls.forEach(el => el.textContent = car);
}

// --- Chart.js Setup ---
function initCharts() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    // 1. Drowsiness Line Chart
    const ctx1 = document.getElementById('drowsinessChart').getContext('2d');

    // Gradient Fill
    const gradient = ctx1.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.5)'); // Blue
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

    drowsinessChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [{
                label: 'Drowsiness Probability',
                data: Array(20).fill(0),
                borderColor: '#3b82f6',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });

    // 2. Real-time Analysis (Radar Chart)
    const ctx2 = document.getElementById('radarChart').getContext('2d');
    radarChart = new Chart(ctx2, {
        type: 'radar',
        data: {
            labels: ['Eyes', 'Yawn', 'Head Tilt', 'Phone', 'Smoke'],
            datasets: [{
                label: 'Risk Level',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(16, 185, 129, 0.2)', // Green tint
                borderColor: '#10b981',
                pointBackgroundColor: '#10b981',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#10b981'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: '#94a3b8',
                        font: {
                            size: 12
                        }
                    },
                    suggestedMin: 0,
                    suggestedMax: 100,
                    ticks: {
                        display: false, // Hide numbers
                        backdropColor: 'transparent'
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// --- Data Fetching ---
async function fetchLiveData() {
    try {
        const res = await fetch('/live');
        const data = await res.json();
        updateDashboard(data);
    } catch (e) {
        console.error("Live data fetch error:", e);
    }
}

function updateDashboard(data) {
    // 1. Update Text Stats
    // Driving Duration
    if (data.driving_time_str) {
        document.getElementById('driveTime').textContent = data.driving_time_str;
    }

    // Safety Score (Inverse of Overall Risk)
    const overall = data.overall_risk || 0;
    const safetyScore = Math.max(0, 100 - overall);
    document.getElementById('safetyScore').textContent = safetyScore;

    // Next Break
    if (data.remaining_break_min !== undefined) {
        document.getElementById('breakTime').textContent = data.remaining_break_min + 'm';
        if (data.remaining_break_min <= 0) {
            document.getElementById('breakTime').style.color = 'var(--danger)';
            document.getElementById('breakTime').textContent = "NOW";
        } else {
            document.getElementById('breakTime').style.color = 'var(--text-main)';
        }
    }

    // 2. Update Charts

    // Update Line Chart (Drowsiness Probability = Overall Risk)
    const labels = drowsinessChart.data.labels;
    const values = drowsinessChart.data.datasets[0].data;

    labels.shift();
    labels.push(new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }));

    values.shift();
    values.push(overall);

    drowsinessChart.update('none');

    // Update Radar Chart
    // ['Eyes', 'Yawn', 'Head Tilt', 'Phone', 'Smoke']
    // Convert status to 0-100 score
    const phoneScore = data.phone_status === 'DETECTED' ? 100 : 0;
    const smokeScore = data.smoking_status === 'DETECTED' ? 100 : 0;

    // Eye score is 100 (Open) -> 0 (Closed). Risk is inverse.
    // Wait, detect.py sends eye_score as 100 (Open). So Risk = 100 - eye_score.
    const eyeRisk = 100 - (data.eye_score || 100);
    const yawnRisk = data.yawn_score || 0; // 0 (Closed) -> 100 (Open). Wait, yawn score logic in detect.py: 0 (Closed) -> 100 (Open).
    // Actually, yawn score in detect.py: max(0, min(100, int((mar - 0.1) / (0.6 - 0.1) * 100)))
    // If mar is high (yawning), score is high. So this IS risk.
    // Wait, let's check detect.py again.
    // eye_score: ear > 0.3 -> 100 (Open). ear < 0.15 -> 0 (Closed).
    // So eye_score is "Openness". Risk is "Closedness" => 100 - eye_score.

    // yawn_score: mar > 0.6 -> 100 (Open/Yawning). mar < 0.1 -> 0 (Closed).
    // So yawn_score is "Yawningness". This IS risk.

    // tilt_score: pitch > 40 -> 100 (Down). pitch < 10 -> 0 (Upright).
    // This IS risk.

    radarChart.data.datasets[0].data = [
        eyeRisk,
        yawnRisk,
        data.tilt_score || 0,
        phoneScore,
        smokeScore
    ];

    // Dynamic Color for Radar
    if (overall > 70) {
        radarChart.data.datasets[0].backgroundColor = 'rgba(239, 68, 68, 0.2)';
        radarChart.data.datasets[0].borderColor = '#ef4444';
        radarChart.data.datasets[0].pointBackgroundColor = '#ef4444';
    } else if (overall > 30) {
        radarChart.data.datasets[0].backgroundColor = 'rgba(245, 158, 11, 0.2)';
        radarChart.data.datasets[0].borderColor = '#f59e0b';
        radarChart.data.datasets[0].pointBackgroundColor = '#f59e0b';
    } else {
        radarChart.data.datasets[0].backgroundColor = 'rgba(16, 185, 129, 0.2)';
        radarChart.data.datasets[0].borderColor = '#10b981';
        radarChart.data.datasets[0].pointBackgroundColor = '#10b981';
    }

    radarChart.update();

    // 3. Live Monitors Logic
    updateMonitor('phoneStatus', data.phone_status === 'DETECTED', 'Detected', 'None');
    updateMonitor('smokingStatus', data.smoking_status === 'DETECTED', 'Detected', 'None');

    // Seatbelt Logic
    const seatbeltEl = document.getElementById('seatbeltStatus');
    const seatbeltIcon = document.getElementById('seatbeltMonitorIcon');
    const seatbeltAlertEl = document.getElementById('seatbeltAlert');

    if (data.seatbelt_status === 'NOT_WORN') {
        if (seatbeltEl) {
            seatbeltEl.textContent = 'Not Worn';
            seatbeltEl.style.color = 'var(--danger)';
        }
        if (seatbeltIcon) {
            seatbeltIcon.className = 'monitor-icon bg-red-soft pulse-animation';
            seatbeltIcon.innerHTML = '<i class="fas fa-user-slash"></i>';
        }
        if (seatbeltAlertEl) seatbeltAlertEl.style.display = 'flex';

        speakSeatbeltAlert();
    } else {
        if (seatbeltEl) {
            seatbeltEl.textContent = 'Worn';
            seatbeltEl.style.color = 'var(--success)';
        }
        if (seatbeltIcon) {
            seatbeltIcon.className = 'monitor-icon bg-green-soft';
            seatbeltIcon.innerHTML = '<i class="fas fa-user-shield"></i>';
        }
        if (seatbeltAlertEl) seatbeltAlertEl.style.display = 'none';

        stopSeatbeltAlert();
    }

    // 4. Break Logic
    if (!breakAlertDismissed && data.break_recommended) {
        document.getElementById('breakAlert').style.display = 'flex';
    }
}

function updateMonitor(elementId, isDetected, detectedText, normalText) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (isDetected) {
        el.textContent = detectedText;
        el.style.color = 'var(--danger)';
        // Find parent monitor-item and add pulse
        el.closest('.monitor-item').style.borderColor = 'var(--danger)';
    } else {
        el.textContent = normalText;
        el.style.color = 'var(--text-muted)';
        el.closest('.monitor-item').style.borderColor = 'var(--glass-border)';
    }
}

// --- Alerts Logic ---
async function fetchAlerts() {
    try {
        const res = await fetch('/alerts');
        const data = await res.json();
        updateAlertsTable(data);
        updateAllAlertsTable(data);

        // Update Total Alerts Count
        document.getElementById('totalAlerts').textContent = data.length;

    } catch (e) {
        console.error("Alerts fetch error:", e);
    }
}

function updateAlertsTable(alerts) {
    const tbody = document.getElementById('alertsBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding:20px;">No alerts yet</td></tr>';
        return;
    }

    alerts.slice(0, 5).forEach(alert => {
        tbody.appendChild(createAlertRow(alert));
    });
}

function updateAllAlertsTable(alerts) {
    const tbody = document.getElementById('allAlertsBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding:20px;">No alerts yet</td></tr>';
        return;
    }

    alerts.forEach(alert => {
        tbody.appendChild(createAlertRow(alert));
    });
}

function createAlertRow(alert) {
    const tr = document.createElement('tr');

    let severity = 'Low';
    let color = 'var(--success)';

    if (alert.status === 'Drowsy' || alert.status === 'Sleep' || alert.status === 'Phone Usage' || alert.status === 'Seatbelt') {
        severity = 'High';
        color = 'var(--danger)';
    } else if (alert.status === 'Yawn' || alert.status === 'Smoking') {
        severity = 'Medium';
        color = 'var(--warning)';
    }

    let details = '-';
    if (alert.extra && alert.extra.reason) {
        details = alert.extra.reason;
    }

    tr.innerHTML = `
        <td>${alert.timestamp.split(' ')[1]}</td>
        <td style="font-weight: 500;">${alert.status}</td>
        <td><span class="badge" style="background: ${color}20; color: ${color}; border: 1px solid ${color}40;">${severity}</span></td>
        <td style="color: var(--text-muted);">${details}</td>
    `;
    return tr;
}

// --- Voice Alert ---
function speakSeatbeltAlert() {
    if (seatbeltSpeechInterval) return;

    const msg = new SpeechSynthesisUtterance("Please wear your seat belt");
    window.speechSynthesis.speak(msg);

    seatbeltSpeechInterval = setInterval(() => {
        window.speechSynthesis.speak(msg);
    }, 180000); // 3 mins
}

function stopSeatbeltAlert() {
    if (seatbeltSpeechInterval) {
        clearInterval(seatbeltSpeechInterval);
        seatbeltSpeechInterval = null;
        window.speechSynthesis.cancel();
    }
}

function dismissBreakAlert() {
    document.getElementById('breakAlert').style.display = 'none';
    breakAlertDismissed = true;
}

// --- Navigation ---
function switchView(viewName) {
    const views = document.querySelectorAll('.view-section');
    views.forEach(v => v.style.display = 'none');

    const selected = document.getElementById('view-' + viewName);
    if (selected) selected.style.display = 'block';

    const links = document.querySelectorAll('.nav-link');
    links.forEach(l => l.classList.remove('active'));

    const activeLink = Array.from(links).find(l => l.getAttribute('onclick').includes(viewName));
    if (activeLink) activeLink.classList.add('active');
}

function logout() {
    if (confirm('Are you sure you want to logout?')) {
        window.location.href = '/';
    }
}
