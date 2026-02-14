"""
Explainability module for Crop Recommendation System.
Handles SHAP-based explanations and visualizations.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_PATH, "models")
STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "images", "shap_plots")

# Ensure SHAP plots directory exists
os.makedirs(STATIC_PATH, exist_ok=True)

# SHAP explainer cache
_shap_explainer = None
SHAP_AVAILABLE = False

# Feature names for display
DISPLAY_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def load_shap_explainer():
    """Load SHAP explainer from pickle file."""
    global _shap_explainer, SHAP_AVAILABLE
    
    try:
        explainer_path = os.path.join(MODELS_PATH, "explainers", "shap_explainer.pkl")
        with open(explainer_path, 'rb') as f:
            _shap_explainer = pickle.load(f)
        SHAP_AVAILABLE = True
        print("[+] SHAP Explainer loaded")
        return _shap_explainer
    except FileNotFoundError:
        print("[!] SHAP Explainer not found")
        SHAP_AVAILABLE = False
        return None
    except Exception as e:
        print(f"[!] Error loading SHAP Explainer: {e}")
        SHAP_AVAILABLE = False
        return None


def get_shap_explainer():
    """Get the loaded SHAP explainer."""
    return _shap_explainer


def is_shap_available():
    """Check if SHAP is available."""
    return SHAP_AVAILABLE and _shap_explainer is not None


def generate_shap_explanation(features_scaled, predicted_class_idx, feature_names):
    """
    Generate SHAP explanation for the prediction.
    
    Args:
        features_scaled: Scaled feature array
        predicted_class_idx: Index of predicted class
        feature_names: List of feature names
    
    Returns:
        dict: SHAP explanation data
    """
    if not is_shap_available():
        return generate_fallback_explanation(features_scaled, feature_names)
    
    try:
        import shap
        shap_values = _shap_explainer.shap_values(features_scaled)
        
        # Get SHAP values for predicted class
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_class_idx][0]
        else:
            class_shap_values = shap_values[0, :, predicted_class_idx]
        
        # Get top contributing features (focus on original 7)
        feature_contributions = []
        for i, feat in enumerate(feature_names):
            if feat in DISPLAY_FEATURES:
                feature_contributions.append({
                    'feature': feat,
                    'value': float(features_scaled[0][i]),
                    'shap_value': float(class_shap_values[i]),
                    'contribution': f"{class_shap_values[i]:+.2f}"
                })
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'success': True,
            'top_features': feature_contributions[:7],
            'all_features': feature_contributions,
            'shap_values': class_shap_values.tolist(),
            'method': 'SHAP'
        }
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return generate_fallback_explanation(features_scaled, feature_names)


def generate_fallback_explanation(features_scaled, feature_names):
    """
    Generate fallback explanation when SHAP is not available.
    Uses feature value analysis.
    """
    contributions = []
    raw_values = features_scaled[0][:7]  # First 7 are original features
    
    # Simple importance based on normalized values
    for i, feat in enumerate(DISPLAY_FEATURES):
        val = raw_values[i]
        contributions.append({
            'feature': feat,
            'value': float(val),
            'shap_value': float(val),  # Use scaled value as proxy
            'contribution': f"{val:+.2f}"
        })
    
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    return {
        'success': True,
        'top_features': contributions[:5],
        'all_features': contributions,
        'method': 'Feature Analysis'
    }


def feature_contribution_text(crop_name, top_features, input_values):
    """
    Generate detailed, farmer-friendly explanation text with HTML formatting.

    Args:
        crop_name: Name of recommended crop
        top_features: List of top contributing features
        input_values: Original input values dict

    Returns:
        str: Human-readable HTML explanation
    """
    # --- Value range classifications ---
    ranges = {
        'N': [(0, 30, 'Low'), (30, 80, 'Moderate'), (80, 200, 'High')],
        'P': [(0, 25, 'Low'), (25, 60, 'Moderate'), (60, 200, 'High')],
        'K': [(0, 30, 'Low'), (30, 80, 'Moderate'), (80, 300, 'High')],
        'temperature': [(0, 15, 'Cool'), (15, 28, 'Warm'), (28, 50, 'Hot')],
        'humidity': [(0, 40, 'Low'), (40, 70, 'Moderate'), (70, 100, 'High')],
        'ph': [(3, 5.5, 'Acidic'), (5.5, 7.5, 'Neutral'), (7.5, 10, 'Alkaline')],
        'rainfall': [(0, 80, 'Low'), (80, 200, 'Moderate'), (200, 500, 'High')]
    }

    def classify_value(name, value):
        for lo, hi, label in ranges.get(name, []):
            if lo <= value <= hi:
                return label
        return 'Normal'

    # --- What each parameter means for a farmer ---
    param_meaning = {
        'N': (
            'Nitrogen (N)',
            'kg/ha',
            'Nitrogen is one of the most important nutrients for plant growth. '
            'It helps your crop develop healthy green leaves and strong stems. '
            'Plants use nitrogen to make chlorophyll, which captures sunlight and converts it into energy.'
        ),
        'P': (
            'Phosphorus (P)',
            'kg/ha',
            'Phosphorus helps your crop develop strong roots, healthy flowers, and good seed formation. '
            'It is especially important during the early stages of plant growth when roots are establishing, '
            'and again during flowering and fruit setting.'
        ),
        'K': (
            'Potassium (K)',
            'kg/ha',
            'Potassium strengthens your crop against diseases and drought stress. '
            'It helps regulate water movement inside the plant, improves the quality of fruits and grains, '
            'and makes your crop more resistant to pests and weather damage.'
        ),
        'temperature': (
            'Temperature',
            '°C',
            'Temperature controls how fast your crop grows. Each crop has an ideal temperature range. '
            'If it is too cold, growth slows down; if it is too hot, the plant gets stressed and yields drop.'
        ),
        'humidity': (
            'Humidity',
            '%',
            'Humidity is the amount of moisture in the air. '
            'Some crops like rice and banana need moist air, while others like wheat and chickpea prefer drier conditions. '
            'High humidity can also increase the risk of fungal diseases if the crop is not suited to it.'
        ),
        'ph': (
            'Soil pH',
            '',
            'Soil pH tells you whether your soil is acidic, neutral, or alkaline. '
            'Most nutrients are best absorbed by plants when pH is between 6.0 and 7.5. '
            'If your soil is too acidic or too alkaline, some nutrients become locked in the soil and the crop cannot use them.'
        ),
        'rainfall': (
            'Rainfall',
            'mm',
            'Rainfall is the primary source of water for rainfed farming. '
            'Different crops need different amounts of water. '
            'Crops like rice need heavy rainfall, while crops like millet and chickpea can grow with much less water.'
        )
    }

    # --- Crop-specific context ---
    crop_needs = {
        'rice':       {'N': 'high', 'humidity': 'high', 'rainfall': 'high', 'temperature': 'warm', 'ph': 'slightly acidic to neutral'},
        'wheat':      {'N': 'moderate', 'humidity': 'moderate', 'rainfall': 'moderate', 'temperature': 'cool to warm', 'ph': 'neutral'},
        'maize':      {'N': 'high', 'humidity': 'moderate', 'rainfall': 'moderate', 'temperature': 'warm', 'ph': 'neutral'},
        'chickpea':   {'N': 'low (fixes own)', 'P': 'moderate', 'rainfall': 'low', 'temperature': 'cool to warm', 'ph': 'neutral to alkaline'},
        'lentil':     {'N': 'low (fixes own)', 'P': 'moderate', 'rainfall': 'low', 'temperature': 'cool', 'ph': 'neutral'},
        'cotton':     {'N': 'high', 'K': 'high', 'rainfall': 'moderate', 'temperature': 'hot', 'ph': 'neutral to alkaline'},
        'coffee':     {'N': 'moderate', 'humidity': 'high', 'rainfall': 'high', 'temperature': 'warm', 'ph': 'acidic'},
        'mango':      {'K': 'high', 'humidity': 'moderate', 'rainfall': 'moderate', 'temperature': 'warm to hot', 'ph': 'slightly acidic'},
        'banana':     {'N': 'high', 'K': 'high', 'humidity': 'high', 'rainfall': 'high', 'temperature': 'warm', 'ph': 'slightly acidic'},
        'coconut':    {'K': 'high', 'humidity': 'high', 'rainfall': 'high', 'temperature': 'warm to hot', 'ph': 'neutral'},
        'jute':       {'N': 'high', 'humidity': 'high', 'rainfall': 'high', 'temperature': 'warm to hot', 'ph': 'slightly acidic'},
        'grapes':     {'K': 'high', 'P': 'moderate', 'rainfall': 'low to moderate', 'temperature': 'warm', 'ph': 'neutral'},
        'apple':      {'N': 'moderate', 'K': 'moderate', 'rainfall': 'moderate', 'temperature': 'cool', 'ph': 'slightly acidic'},
        'orange':     {'N': 'moderate', 'K': 'high', 'humidity': 'moderate', 'temperature': 'warm', 'ph': 'slightly acidic'},
        'papaya':     {'N': 'high', 'K': 'high', 'humidity': 'high', 'rainfall': 'moderate to high', 'temperature': 'warm to hot', 'ph': 'neutral'},
        'watermelon': {'N': 'moderate', 'K': 'high', 'rainfall': 'moderate', 'temperature': 'hot', 'ph': 'slightly acidic to neutral'},
        'muskmelon':  {'N': 'moderate', 'K': 'high', 'rainfall': 'low to moderate', 'temperature': 'hot', 'ph': 'neutral'},
        'pomegranate':{'K': 'high', 'rainfall': 'low', 'temperature': 'warm to hot', 'ph': 'neutral to alkaline'},
        'pigeonpeas': {'N': 'low (fixes own)', 'P': 'moderate', 'rainfall': 'moderate', 'temperature': 'warm', 'ph': 'neutral'},
        'mothbeans':  {'N': 'low (fixes own)', 'rainfall': 'low', 'temperature': 'hot', 'ph': 'neutral to alkaline'},
        'mungbean':   {'N': 'low (fixes own)', 'P': 'moderate', 'rainfall': 'low to moderate', 'temperature': 'warm', 'ph': 'neutral'},
        'blackgram':  {'N': 'low (fixes own)', 'P': 'moderate', 'rainfall': 'low to moderate', 'temperature': 'warm', 'ph': 'neutral'},
        'kidneybeans':{'P': 'moderate', 'K': 'moderate', 'rainfall': 'moderate', 'temperature': 'cool to warm', 'ph': 'slightly acidic to neutral'},
    }

    crop_key = crop_name.lower()
    needs = crop_needs.get(crop_key, {})

    # --- Build the explanation ---
    text = '<div class="farmer-explanation">'
    text += f'<h6 class="fw-bold text-success mb-3">Why is {crop_name} the best choice for your land?</h6>'
    text += (
        f'<p>Based on the soil and weather conditions you provided, our AI model has analyzed '
        f'your land and determined that <strong>{crop_name}</strong> is the most suitable crop. '
        f'Here is a detailed breakdown of the key factors that led to this recommendation:</p>'
    )

    for i, feat in enumerate(top_features, 1):
        name = feat['feature']
        desc, unit, meaning = param_meaning.get(name, (name, '', ''))
        raw_value = input_values.get(name, feat['value'])
        level = classify_value(name, float(raw_value))
        shap_val = feat['shap_value']
        is_positive = shap_val > 0

        unit_str = f' {unit}' if unit else ''

        # Impact sentence
        if is_positive:
            impact_class = 'text-success'
            impact_icon = 'bi-check-circle-fill'
            if abs(shap_val) > 0.5:
                strength = 'strongly supports'
            elif abs(shap_val) > 0.1:
                strength = 'supports'
            else:
                strength = 'slightly supports'
        else:
            impact_class = 'text-warning'
            impact_icon = 'bi-exclamation-triangle-fill'
            if abs(shap_val) > 0.5:
                strength = 'works against'
            elif abs(shap_val) > 0.1:
                strength = 'somewhat works against'
            else:
                strength = 'has a small negative effect on'

        # Crop-specific note
        crop_need = needs.get(name, '')
        crop_note = ''
        if crop_need:
            crop_note = f' {crop_name} typically needs <strong>{crop_need}</strong> {desc.lower().split("(")[0].strip()}.'

        # Farmer-friendly advice
        advice = ''
        if name == 'N':
            if level == 'Low':
                advice = 'You may want to add nitrogen-rich fertilizers like Urea or DAP before sowing.'
            elif level == 'High':
                advice = 'Your soil already has plenty of nitrogen. Avoid adding too much more, as excess nitrogen can delay fruit formation.'
        elif name == 'P':
            if level == 'Low':
                advice = 'Consider adding phosphorus through fertilizers like SSP (Single Super Phosphate) or DAP to help root growth.'
            elif level == 'High':
                advice = 'Your soil has good phosphorus reserves for strong root development.'
        elif name == 'K':
            if level == 'Low':
                advice = 'Adding MOP (Muriate of Potash) can help improve crop quality and disease resistance.'
            elif level == 'High':
                advice = 'Your potassium levels are high, which will help the crop resist disease and produce better quality yield.'
        elif name == 'temperature':
            if level == 'Cool':
                advice = 'Cool temperatures favour crops like wheat, chickpea, and apple. Protect seedlings from frost if needed.'
            elif level == 'Hot':
                advice = 'Hot conditions suit crops like cotton, watermelon, and millet. Ensure adequate irrigation to prevent heat stress.'
            else:
                advice = 'Your temperature range is comfortable for most major crops.'
        elif name == 'humidity':
            if level == 'High':
                advice = 'Watch out for fungal diseases in high humidity. Ensure good spacing between plants for air circulation.'
            elif level == 'Low':
                advice = 'Low humidity reduces disease risk but may increase water loss from leaves. Mulching can help.'
        elif name == 'ph':
            if level == 'Acidic':
                advice = 'Acidic soil is good for crops like coffee, tea, and blueberries. Adding lime can raise pH if needed for other crops.'
            elif level == 'Alkaline':
                advice = 'Alkaline soil suits crops like chickpea and pomegranate. Adding gypsum or organic matter can lower pH if needed.'
            else:
                advice = 'Your soil pH is in the ideal range where most nutrients are available to plants.'
        elif name == 'rainfall':
            if level == 'Low':
                advice = 'With low rainfall, choose drought-tolerant crops or arrange supplementary irrigation like drip or sprinkler systems.'
            elif level == 'High':
                advice = 'High rainfall is great for water-loving crops. Ensure your field has proper drainage to prevent waterlogging.'
            else:
                advice = 'Moderate rainfall works well for most crops. Plan irrigation for dry spells during critical growth stages.'

        text += f'''
        <div class="mb-4 p-3 border rounded bg-light">
            <div class="d-flex align-items-center mb-2">
                <span class="badge bg-primary rounded-pill me-2">{i}</span>
                <strong class="fs-6">{desc}: {raw_value}{unit_str}</strong>
                <span class="badge bg-secondary ms-2">{level}</span>
            </div>
            <p class="mb-2 small text-muted">{meaning}</p>
            <p class="mb-2">
                <i class="bi {impact_icon} {impact_class} me-1"></i>
                Your {desc.lower()} of <strong>{raw_value}{unit_str}</strong> ({level.lower()}) <strong class="{impact_class}">{strength}</strong>
                the recommendation for {crop_name}.{crop_note}
            </p>
            {"<p class='mb-0 small fst-italic'><i class=\"bi bi-lightbulb text-warning me-1\"></i><strong>Tip:</strong> " + advice + "</p>" if advice else ""}
        </div>'''

    # Overall summary
    positive_count = sum(1 for f in top_features if f['shap_value'] > 0)
    total = len(top_features)

    if positive_count == total:
        summary = f'All {total} key factors in your soil and climate strongly point towards {crop_name}. Your land conditions are an excellent match.'
    elif positive_count > total / 2:
        summary = (
            f'{positive_count} out of {total} key factors favour {crop_name}. '
            f'While some conditions are not perfect, the overall match is good. '
            f'You can improve results by addressing the factors marked with warnings above.'
        )
    else:
        summary = (
            f'While {crop_name} is still the best match among all 22 crops, '
            f'some of your soil or climate conditions are not ideal. '
            f'Pay attention to the tips above to improve your chances of a good harvest.'
        )

    text += f'''
    <div class="alert alert-success mt-4">
        <i class="bi bi-patch-check-fill me-2"></i>
        <strong>Summary:</strong> {summary}
    </div>
    </div>'''

    return text


def get_top_features(explanation_data, n=5):
    """
    Get top N contributing features.
    
    Returns:
        list: Top N features with contributions
    """
    if not explanation_data.get('success'):
        return []
    
    return explanation_data.get('top_features', [])[:n]


def create_shap_plots(features_scaled, predicted_class_idx, crop_name, session_id="default"):
    """
    Create and save SHAP visualization plots.
    
    Args:
        features_scaled: Scaled features
        predicted_class_idx: Predicted class index
        crop_name: Name of predicted crop
        session_id: Unique session identifier for filenames
    
    Returns:
        dict: Paths to generated plot files
    """
    plots = {
        'waterfall': None,
        'bar': None
    }
    
    if not is_shap_available():
        return plots
    
    try:
        import shap
        shap_values = _shap_explainer.shap_values(features_scaled)
        
        # Get values for predicted class
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        else:
            class_shap = shap_values[0, :, predicted_class_idx]
        
        # Create waterfall-style bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Focus on original 7 features
        indices = list(range(7))
        values = class_shap[:7]
        features = DISPLAY_FEATURES
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(values))[::-1]
        sorted_values = [values[i] for i in sorted_idx]
        sorted_features = [features[i] for i in sorted_idx]
        
        colors = ['#28a745' if v > 0 else '#dc3545' for v in sorted_values]
        
        ax.barh(range(len(sorted_features)), sorted_values, color=colors)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Feature Contributions for {crop_name} Recommendation')
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot
        waterfall_path = os.path.join(STATIC_PATH, f"waterfall_{session_id}.png")
        plt.savefig(waterfall_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        plots['waterfall'] = f"images/shap_plots/waterfall_{session_id}.png"
        
    except Exception as e:
        print(f"Error creating SHAP plots: {e}")
    
    return plots
