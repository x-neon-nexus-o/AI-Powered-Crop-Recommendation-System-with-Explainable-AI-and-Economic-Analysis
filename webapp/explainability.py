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
        print("✅ SHAP Explainer loaded")
        return _shap_explainer
    except FileNotFoundError:
        print("⚠️ SHAP Explainer not found")
        SHAP_AVAILABLE = False
        return None
    except Exception as e:
        print(f"⚠️ Error loading SHAP Explainer: {e}")
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
            'top_features': feature_contributions[:5],
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
    Generate human-readable explanation text with HTML formatting.
    
    Args:
        crop_name: Name of recommended crop
        top_features: List of top contributing features
        input_values: Original input values dict
    
    Returns:
        str: Human-readable HTML explanation
    """
    text = f"Your soil and climate conditions are suitable for <strong>{crop_name}</strong> because:<br><br>"
    
    feature_descriptions = {
        'N': ('Nitrogen level', 'kg/ha'),
        'P': ('Phosphorus level', 'kg/ha'),
        'K': ('Potassium level', 'kg/ha'),
        'temperature': ('Temperature', '°C'),
        'humidity': ('Humidity', '%'),
        'ph': ('pH level', ''),
        'rainfall': ('Rainfall', 'mm')
    }
    
    for i, feat in enumerate(top_features[:3], 1):
        name = feat['feature']
        desc, unit = feature_descriptions.get(name, (name, ''))
        value = input_values.get(name, feat['value'])
        impact = "contributes positively" if feat['shap_value'] > 0 else "is a consideration"
        
        text += f"{i}. <strong>{desc}</strong> ({value}{unit}) {impact}<br>"
    
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
