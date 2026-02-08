/**
 * Dashboard JavaScript for CropAI
 * Chart configurations and visualizations
 */

// Color palette
const CHART_COLORS = {
    primary: 'rgba(74, 124, 35, 0.8)',
    secondary: 'rgba(13, 110, 253, 0.8)',
    success: 'rgba(40, 167, 69, 0.8)',
    warning: 'rgba(255, 193, 7, 0.8)',
    danger: 'rgba(220, 53, 69, 0.8)',
    info: 'rgba(23, 162, 184, 0.8)',
    purple: 'rgba(111, 66, 193, 0.8)'
};

const CHART_COLORS_ARRAY = [
    CHART_COLORS.success,
    CHART_COLORS.primary,
    CHART_COLORS.warning,
    CHART_COLORS.danger,
    CHART_COLORS.purple,
    CHART_COLORS.info
];

/**
 * Create probability bar chart for crop predictions
 */
function createProbabilityChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability',
                data: data,
                backgroundColor: data.map((v, i) => CHART_COLORS_ARRAY[i % CHART_COLORS_ARRAY.length]),
                borderColor: data.map((v, i) => CHART_COLORS_ARRAY[i % CHART_COLORS_ARRAY.length].replace('0.8', '1')),
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Crop Recommendation Probabilities' }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

/**
 * Create ROI comparison chart
 */
function createROIChart(canvasId, crops, roiValues) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: crops,
            datasets: [{
                label: 'ROI (%)',
                data: roiValues,
                backgroundColor: CHART_COLORS_ARRAY,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Return on Investment Comparison' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { callback: value => value + '%' }
                }
            }
        }
    });
}

/**
 * Create cost breakdown pie chart
 */
function createCostBreakdownChart(canvasId, costs) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const labels = Object.keys(costs);
    const data = Object.values(costs);

    return new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                data: data,
                backgroundColor: [
                    CHART_COLORS.success,
                    CHART_COLORS.primary,
                    CHART_COLORS.warning,
                    CHART_COLORS.secondary
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' },
                title: { display: true, text: 'Cost Breakdown' },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ₹${context.raw.toLocaleString()}`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create rotation timeline visualization
 */
function createRotationTimeline(containerId, rotationPlan) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = rotationPlan.map((item, index) => `
        <div class="rotation-step d-flex align-items-center mb-3">
            <div class="step-number rounded-circle ${index === 0 ? 'bg-success' : index === 1 ? 'bg-primary' : 'bg-secondary'
        } text-white d-flex align-items-center justify-content-center me-3"
                 style="width: 40px; height: 40px; font-weight: bold;">
                ${index + 1}
            </div>
            <div class="step-content">
                <strong>${item.season}</strong>: ${item.crop}
                <span class="badge bg-light text-dark ms-2">${item.category}</span>
            </div>
        </div>
    `).join('');
}

/**
 * Create sustainability gauge chart
 */
function createSustainabilityGauge(canvasId, score) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [
                    score >= 70 ? CHART_COLORS.success : score >= 40 ? CHART_COLORS.warning : CHART_COLORS.danger,
                    'rgba(200, 200, 200, 0.2)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '75%',
            rotation: -90,
            circumference: 180,
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });
}

/**
 * Create feature contribution chart
 */
function createFeatureChart(canvasId, features, values) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Contribution',
                data: values,
                backgroundColor: values.map(v => v > 0 ? CHART_COLORS.success : CHART_COLORS.danger),
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { x: { beginAtZero: true } }
        }
    });
}

/**
 * Create radar chart for multi-crop comparison
 */
function createRadarChart(canvasId, crops, dimensions) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx.getContext('2d'), {
        type: 'radar',
        data: {
            labels: dimensions.labels,
            datasets: crops.map((crop, i) => ({
                label: crop.name,
                data: crop.values,
                backgroundColor: CHART_COLORS_ARRAY[i].replace('0.8', '0.2'),
                borderColor: CHART_COLORS_ARRAY[i],
                borderWidth: 2
            }))
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'bottom' } },
            scales: { r: { beginAtZero: true, max: 100 } }
        }
    });
}

/**
 * Animate number counting
 */
function animateValue(element, start, end, duration) {
    if (!element) return;

    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease out

        element.textContent = Math.round(start + range * easeProgress);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/**
 * Format currency
 */
function formatCurrency(amount) {
    return '₹' + amount.toLocaleString('en-IN');
}

/**
 * Format percentage
 */
function formatPercentage(value) {
    return value.toFixed(1) + '%';
}

// Export functions for global use
window.CropAIDashboard = {
    createProbabilityChart,
    createROIChart,
    createCostBreakdownChart,
    createRotationTimeline,
    createSustainabilityGauge,
    createFeatureChart,
    createRadarChart,
    animateValue,
    formatCurrency,
    formatPercentage,
    CHART_COLORS
};

console.log('📊 CropAI Dashboard loaded');
