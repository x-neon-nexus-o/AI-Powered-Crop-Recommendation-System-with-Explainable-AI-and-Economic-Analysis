"""
Utility functions for the Crop Recommendation Flask Application.
Handles feature engineering, input validation, and model loading.
"""

import pickle
import numpy as np
import pandas as pd
import os

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_PATH, "models")
DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(DATA_PATH, "results")
ML_READY_PATH = os.path.join(DATA_PATH, "processed", "ml_ready")

# Feature names (39 engineered features)
FEATURE_NAMES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    'N_to_P_ratio', 'N_to_K_ratio', 'P_to_K_ratio', 'NPK_sum', 'NPK_product',
    'N_dominance', 'P_dominance', 'K_dominance', 'temp_humidity_interaction',
    'climate_index', 'heat_stress_index', 'ph_squared', 'ph_deviation',
    'N_ph_interaction', 'P_ph_interaction', 'K_ph_interaction',
    'water_stress_index', 'moisture_index', 'rainfall_per_temp',
    'water_availability', 'growing_condition_index', 'resource_availability',
    'environmental_stress', 'nutrient_balance', 'temp_category_Hot',
    'temp_category_Moderate', 'humidity_category_Low', 'humidity_category_Medium',
    'ph_category_Alkaline', 'ph_category_Neutral', 'rainfall_category_Low',
    'rainfall_category_Medium'
]

# Cached models
_model = None
_scaler = None
_label_encoder = None
_economic_df = None


def load_models():
    """Load all required models on application startup."""
    global _model, _scaler, _label_encoder, _economic_df
    
    # Load stacking ensemble model
    try:
        with open(os.path.join(MODELS_PATH, "ensemble", "stacking_ensemble.pkl"), 'rb') as f:
            _model = pickle.load(f)
        print("✅ Stacking Ensemble loaded")
    except FileNotFoundError:
        with open(os.path.join(MODELS_PATH, "random_forest_model.pkl"), 'rb') as f:
            _model = pickle.load(f)
        print("✅ Random Forest loaded (fallback)")
    
    # Load scaler
    with open(os.path.join(MODELS_PATH, "scaler_standard.pkl"), 'rb') as f:
        _scaler = pickle.load(f)
    print("✅ Scaler loaded")
    
    # Load label encoder
    with open(os.path.join(MODELS_PATH, "label_encoder.pkl"), 'rb') as f:
        _label_encoder = pickle.load(f)
    print("✅ Label Encoder loaded")
    
    # Load economic data
    try:
        _economic_df = pd.read_csv(os.path.join(RESULTS_PATH, "economic_analysis.csv"))
        print("✅ Economic data loaded")
    except FileNotFoundError:
        print("⚠️ Economic data not found")
        _economic_df = None
    
    return _model, _scaler, _label_encoder


def get_model():
    """Get the loaded model."""
    return _model


def get_scaler():
    """Get the loaded scaler."""
    return _scaler


def get_label_encoder():
    """Get the loaded label encoder."""
    return _label_encoder


def get_economic_data():
    """Get the economic data DataFrame."""
    return _economic_df


def validate_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """
    Validate input parameters are within acceptable ranges.
    
    Returns:
        tuple: (is_valid: bool, errors: list)
    """
    errors = []
    
    try:
        N = float(N)
        P = float(P)
        K = float(K)
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)
    except (ValueError, TypeError) as e:
        return False, ["All inputs must be valid numbers"]
    
    if not (0 <= N <= 200):
        errors.append(f"Nitrogen (N) must be between 0-200 kg/ha, got {N}")
    if not (0 <= P <= 200):
        errors.append(f"Phosphorus (P) must be between 0-200 kg/ha, got {P}")
    if not (0 <= K <= 300):
        errors.append(f"Potassium (K) must be between 0-300 kg/ha, got {K}")
    if not (0 <= temperature <= 50):
        errors.append(f"Temperature must be between 0-50°C, got {temperature}")
    if not (0 <= humidity <= 100):
        errors.append(f"Humidity must be between 0-100%, got {humidity}")
    if not (3 <= ph <= 10):
        errors.append(f"pH must be between 3-10, got {ph}")
    if not (0 <= rainfall <= 500):
        errors.append(f"Rainfall must be between 0-500mm, got {rainfall}")
    
    return len(errors) == 0, errors


def engineer_features(N, P, K, temperature, humidity, ph, rainfall):
    """
    Transform 7 raw inputs into 39 engineered features.
    
    Returns:
        np.array: Array of shape (1, 39) with engineered features
    """
    features = {}
    
    # 1. Original features (7)
    features['N'] = N
    features['P'] = P
    features['K'] = K
    features['temperature'] = temperature
    features['humidity'] = humidity
    features['ph'] = ph
    features['rainfall'] = rainfall
    
    # 2. Nutrient ratio features (8)
    features['N_to_P_ratio'] = N / (P + 1)
    features['N_to_K_ratio'] = N / (K + 1)
    features['P_to_K_ratio'] = P / (K + 1)
    features['NPK_sum'] = N + P + K
    features['NPK_product'] = N * P * K
    npk_total = N + P + K + 1
    features['N_dominance'] = N / npk_total
    features['P_dominance'] = P / npk_total
    features['K_dominance'] = K / npk_total
    
    # 3. Climate interaction features (3)
    features['temp_humidity_interaction'] = temperature * humidity
    features['climate_index'] = (temperature * humidity) / 100
    features['heat_stress_index'] = temperature / (humidity + 1)
    
    # 4. pH features (5)
    features['ph_squared'] = ph ** 2
    features['ph_deviation'] = abs(ph - 7)
    features['N_ph_interaction'] = N * ph
    features['P_ph_interaction'] = P * ph
    features['K_ph_interaction'] = K * ph
    
    # 5. Water/moisture features (4)
    features['water_stress_index'] = rainfall / (humidity + 1)
    features['moisture_index'] = humidity * rainfall / 100
    features['rainfall_per_temp'] = rainfall / (temperature + 1)
    features['water_availability'] = rainfall * humidity / 100
    
    # 6. Composite indices (4)
    features['growing_condition_index'] = (temperature * humidity * rainfall) / 10000
    features['resource_availability'] = (N + P + K) * rainfall / 1000
    features['environmental_stress'] = abs(temperature - 25) + abs(humidity - 60) + abs(ph - 7) * 10
    features['nutrient_balance'] = 1 / (1 + abs(N - P) + abs(P - K) + abs(N - K))
    
    # 7. Categorical encoding features (8)
    features['temp_category_Hot'] = 1 if temperature > 30 else 0
    features['temp_category_Moderate'] = 1 if 20 <= temperature <= 30 else 0
    features['humidity_category_Low'] = 1 if humidity < 50 else 0
    features['humidity_category_Medium'] = 1 if 50 <= humidity < 70 else 0
    features['ph_category_Alkaline'] = 1 if ph > 7.5 else 0
    features['ph_category_Neutral'] = 1 if 6.5 <= ph <= 7.5 else 0
    features['rainfall_category_Low'] = 1 if rainfall < 100 else 0
    features['rainfall_category_Medium'] = 1 if 100 <= rainfall < 200 else 0
    
    # Convert to array in correct order
    feature_array = np.array([[features[name] for name in FEATURE_NAMES]])
    
    return feature_array


def get_economic_viability(crop_name):
    """Get economic viability data for a crop."""
    if _economic_df is None:
        return None
    
    try:
        crop_data = _economic_df[_economic_df['crop'].str.lower() == crop_name.lower()]
        if not crop_data.empty:
            row = crop_data.iloc[0]
            return {
                'crop': crop_name,
                'roi': float(row.get('roi', row.get('ROI', 0))),
                'profit': float(row.get('profit', row.get('net_profit', 0))),
                'profit_margin': float(row.get('profit_margin', 0)),
                'risk_category': str(row.get('risk_category', row.get('risk_level', 'Unknown'))),
                'volatility': float(row.get('volatility', row.get('price_cv', 0)))
            }
        return None
    except Exception:
        return None


def get_rotation_suggestion(crop_name, current_season='Kharif'):
    """Get crop rotation suggestion based on rules."""
    crop_categories = {
        'rice': 'Cereal', 'wheat': 'Cereal', 'maize': 'Cereal',
        'chickpea': 'Legume', 'lentil': 'Legume', 'mungbean': 'Legume',
        'blackgram': 'Legume', 'kidneybeans': 'Legume', 'mothbeans': 'Legume',
        'pigeonpeas': 'Legume', 'cotton': 'Fiber', 'jute': 'Fiber',
        'coffee': 'Beverage', 'mango': 'Fruit', 'banana': 'Fruit',
        'pomegranate': 'Fruit', 'grapes': 'Fruit', 'watermelon': 'Fruit',
        'muskmelon': 'Fruit', 'orange': 'Fruit', 'papaya': 'Fruit',
        'apple': 'Fruit', 'coconut': 'Oilseed'
    }
    
    rotation_rules = {
        'Cereal': {'next': ['Legume', 'Oilseed'], 'benefit': 'Nitrogen fixation from legumes'},
        'Legume': {'next': ['Cereal', 'Fiber'], 'benefit': 'Soil nutrient enrichment'},
        'Fiber': {'next': ['Legume', 'Cereal'], 'benefit': 'Break pest cycle'},
        'Oilseed': {'next': ['Cereal', 'Legume'], 'benefit': 'Nutrient cycling'},
        'Fruit': {'next': ['Legume'], 'benefit': 'Perennial management'},
        'Beverage': {'next': ['Legume'], 'benefit': 'Soil restoration'}
    }
    
    current_category = crop_categories.get(crop_name.lower(), 'Unknown')
    rule = rotation_rules.get(current_category, {'next': ['Legume'], 'benefit': 'General rotation'})
    
    next_crops = []
    for crop, cat in crop_categories.items():
        if cat in rule['next'] and crop != crop_name.lower():
            next_crops.append(crop.title())
    
    seasons = ['Kharif', 'Rabi', 'Zaid']
    current_idx = seasons.index(current_season) if current_season in seasons else 0
    
    plan = [
        {'season': current_season, 'crop': crop_name.title(), 'category': current_category},
        {'season': seasons[(current_idx + 1) % 3], 'crop': next_crops[0] if next_crops else 'Rest', 'category': rule['next'][0] if rule['next'] else 'Recovery'},
        {'season': seasons[(current_idx + 2) % 3], 'crop': next_crops[1] if len(next_crops) > 1 else 'Green Manure', 'category': 'Recovery'}
    ]
    
    return {
        'plan': plan,
        'current_category': current_category,
        'benefit': rule['benefit'],
        'sustainability_score': 75,
        'rating': 'Good'
    }
