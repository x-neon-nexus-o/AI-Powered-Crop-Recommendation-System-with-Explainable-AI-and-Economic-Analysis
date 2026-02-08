"""
Prediction module for Crop Recommendation System.
Handles crop prediction logic, confidence scoring, and categorization.
"""

import numpy as np
from utils import get_model, get_scaler, get_label_encoder, engineer_features, FEATURE_NAMES


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the best crop based on input features.
    
    Returns:
        dict: Prediction results with top 3 crops, probabilities, and categories
    """
    # Engineer features (7 -> 39)
    features = engineer_features(N, P, K, temperature, humidity, ph, rainfall)
    
    # Scale features
    scaler = get_scaler()
    features_scaled = scaler.transform(features)
    
    # Get predictions
    model = get_model()
    label_encoder = get_label_encoder()
    
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    
    predictions = []
    for idx in top_3_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        prob = probabilities[idx]
        
        predictions.append({
            'crop': crop_name,
            'probability': float(prob),
            'confidence': confidence_score(prob),
            'category': categorize_recommendation(prob),
            'category_class': get_category_class(prob)
        })
    
    return {
        'success': True,
        'predictions': predictions,
        'recommended_crop': predictions[0]['crop'],
        'all_probabilities': get_crop_probabilities(probabilities, label_encoder)
    }


def get_crop_probabilities(probabilities, label_encoder):
    """
    Get all 22 crop probabilities sorted by confidence.
    
    Returns:
        list: All crops with their probabilities
    """
    all_crops = []
    for idx, prob in enumerate(probabilities):
        crop_name = label_encoder.inverse_transform([idx])[0]
        all_crops.append({
            'crop': crop_name,
            'probability': float(prob),
            'percentage': f"{prob * 100:.2f}%"
        })
    
    # Sort by probability descending
    all_crops.sort(key=lambda x: x['probability'], reverse=True)
    return all_crops


def confidence_score(probability):
    """
    Convert probability to confidence score string.
    
    Returns:
        str: Confidence as percentage string
    """
    return f"{probability * 100:.1f}%"


def categorize_recommendation(probability):
    """
    Categorize recommendation based on probability threshold.
    
    Thresholds:
        - Recommended: prob > 70%
        - Slightly Recommended: 40-70%
        - Not Recommended: < 40%
    
    Returns:
        str: Category name
    """
    if probability > 0.7:
        return "Highly Recommended"
    elif probability > 0.4:
        return "Recommended"
    else:
        return "Consider"


def get_category_class(probability):
    """
    Get Bootstrap CSS class for category badge.
    
    Returns:
        str: CSS class name
    """
    if probability > 0.7:
        return "success"
    elif probability > 0.4:
        return "warning"
    else:
        return "secondary"


def compare_crops(crop_names, economic_data=None, rotation_data=None):
    """
    Compare multiple crops side by side.
    
    Args:
        crop_names: List of crop names to compare
        economic_data: Optional economic analysis data
        rotation_data: Optional rotation planning data
    
    Returns:
        dict: Comparison data for all crops
    """
    comparison = []
    
    for crop in crop_names:
        crop_info = {
            'crop': crop,
            'economic': economic_data.get(crop.lower()) if economic_data else None,
            'rotation': rotation_data.get(crop.lower()) if rotation_data else None
        }
        comparison.append(crop_info)
    
    return {
        'crops': comparison,
        'count': len(comparison)
    }
