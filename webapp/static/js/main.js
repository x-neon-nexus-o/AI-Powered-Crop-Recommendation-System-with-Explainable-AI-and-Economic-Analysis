/**
 * Main JavaScript file for CropAI Crop Recommendation System
 */

// Document ready
document.addEventListener('DOMContentLoaded', function () {
    console.log('🌾 CropAI initialized');

    // Initialize tooltips
    initTooltips();

    // Initialize form validation
    initFormValidation();

    // Add smooth scroll
    initSmoothScroll();

    // Add loading states
    initLoadingStates();
});

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation
 */
function initFormValidation() {
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                e.preventDefault();
                const target = document.querySelector(targetId);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

/**
 * Add loading states to submit buttons
 */
function initLoadingStates() {
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        form.addEventListener('submit', function () {
            const submitBtn = form.querySelector('[type="submit"]');
            if (submitBtn && form.checkValidity()) {
                const originalText = submitBtn.innerHTML;
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                    Processing...
                `;

                // Re-enable after 10 seconds (fallback)
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }, 10000);
            }
        });
    });
}

/**
 * Input range validation helper
 */
function validateRange(input, min, max) {
    const value = parseFloat(input.value);
    const isValid = !isNaN(value) && value >= min && value <= max;

    if (isValid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }

    return isValid;
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed`;
    toast.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 300px;';
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close float-end" onclick="this.parentElement.remove()"></button>
    `;
    document.body.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Copy failed:', err);
        showToast('Failed to copy', 'danger');
    });
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString('en-IN', options);
}

/**
 * Animate number counting
 */
function animateCount(element, target, duration = 1000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

/**
 * Print prediction results
 */
function printResults() {
    window.print();
}

/**
 * Export results to JSON
 */
function exportToJSON(data, filename = 'crop_recommendation.json') {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Console branding
console.log('%c🌾 CropAI', 'font-size: 24px; font-weight: bold; color: #4a7c23;');
console.log('%cAI-Powered Crop Recommendation System', 'font-size: 12px; color: #666;');
