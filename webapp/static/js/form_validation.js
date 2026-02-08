/**
 * Form Validation JavaScript for CropAI
 * Input validation and slider synchronization
 */

// Validation ranges
const VALIDATION_RANGES = {
    N: { min: 0, max: 200, label: 'Nitrogen' },
    P: { min: 0, max: 200, label: 'Phosphorus' },
    K: { min: 0, max: 300, label: 'Potassium' },
    temperature: { min: 0, max: 50, label: 'Temperature' },
    humidity: { min: 0, max: 100, label: 'Humidity' },
    ph: { min: 3, max: 10, label: 'pH' },
    rainfall: { min: 0, max: 500, label: 'Rainfall' }
};

/**
 * Validate a numeric input against its range
 */
function validateNumericInput(fieldName, value) {
    const range = VALIDATION_RANGES[fieldName];
    if (!range) return { valid: true };

    const numValue = parseFloat(value);

    if (isNaN(numValue)) {
        return {
            valid: false,
            message: `${range.label} must be a valid number`
        };
    }

    if (numValue < range.min || numValue > range.max) {
        return {
            valid: false,
            message: `${range.label} must be between ${range.min} and ${range.max}`
        };
    }

    return { valid: true };
}

/**
 * Validate all form inputs
 */
function validateAllInputs(formElement) {
    const errors = [];

    for (const [fieldName, range] of Object.entries(VALIDATION_RANGES)) {
        const input = formElement.querySelector(`[name="${fieldName}"]`);
        if (input) {
            const result = validateNumericInput(fieldName, input.value);
            if (!result.valid) {
                errors.push(result.message);
            }
        }
    }

    return errors;
}

/**
 * Show error message for an input
 */
function showErrorMessage(inputElement, message) {
    // Remove existing error
    hideErrorMessage(inputElement);

    // Create error element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback d-block';
    errorDiv.textContent = message;

    inputElement.classList.add('is-invalid');
    inputElement.parentNode.appendChild(errorDiv);
}

/**
 * Hide error message for an input
 */
function hideErrorMessage(inputElement) {
    inputElement.classList.remove('is-invalid');
    inputElement.classList.add('is-valid');

    const existingError = inputElement.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
}

/**
 * Enable/disable submit button based on form validity
 */
function updateSubmitButton(formElement) {
    const submitBtn = formElement.querySelector('[type="submit"]');
    if (!submitBtn) return;

    const errors = validateAllInputs(formElement);
    submitBtn.disabled = errors.length > 0;
}

/**
 * Sync slider and number input values
 */
function syncSliderAndInput(sliderId, inputId) {
    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);

    if (!slider || !input) return;

    // Slider updates input
    slider.addEventListener('input', function () {
        input.value = this.value;
        validateAndUpdate(input);
    });

    // Input updates slider
    input.addEventListener('input', function () {
        slider.value = this.value;
        validateAndUpdate(this);
    });

    function validateAndUpdate(inputEl) {
        const fieldName = inputEl.name;
        const result = validateNumericInput(fieldName, inputEl.value);

        if (result.valid) {
            hideErrorMessage(inputEl);
        } else {
            showErrorMessage(inputEl, result.message);
        }

        updateSubmitButton(inputEl.form);
    }
}

/**
 * Initialize form validation
 */
function initFormValidation(formId) {
    const form = document.getElementById(formId) || document.querySelector('form');
    if (!form) return;

    // Add real-time validation to all inputs
    form.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('input', function () {
            const result = validateNumericInput(this.name, this.value);

            if (result.valid) {
                hideErrorMessage(this);
            } else {
                showErrorMessage(this, result.message);
            }

            updateSubmitButton(form);
        });

        input.addEventListener('blur', function () {
            const result = validateNumericInput(this.name, this.value);

            if (result.valid) {
                this.classList.add('is-valid');
            }
        });
    });

    // Form submission validation
    form.addEventListener('submit', function (e) {
        const errors = validateAllInputs(form);

        if (errors.length > 0) {
            e.preventDefault();

            // Show all errors
            errors.forEach((error, index) => {
                console.error(`Validation Error ${index + 1}: ${error}`);
            });

            // Focus on first invalid input
            const firstInvalid = form.querySelector('.is-invalid');
            if (firstInvalid) {
                firstInvalid.focus();
            }
        }

        form.classList.add('was-validated');
    });
}

/**
 * Set up slider syncs for prediction form
 */
function setupSliderSyncs() {
    const sliderPairs = [
        ['N_slider', 'N'],
        ['P_slider', 'P'],
        ['K_slider', 'K'],
        ['temperature_slider', 'temperature'],
        ['humidity_slider', 'humidity'],
        ['ph_slider', 'ph'],
        ['rainfall_slider', 'rainfall']
    ];

    sliderPairs.forEach(([sliderId, inputId]) => {
        syncSliderAndInput(sliderId, inputId);
    });
}

// Auto-initialize on DOM load
document.addEventListener('DOMContentLoaded', function () {
    initFormValidation();
    setupSliderSyncs();
    console.log('✅ Form validation initialized');
});

// Export for global use
window.CropAIFormValidation = {
    validateNumericInput,
    validateAllInputs,
    showErrorMessage,
    hideErrorMessage,
    updateSubmitButton,
    syncSliderAndInput,
    initFormValidation,
    VALIDATION_RANGES
};
