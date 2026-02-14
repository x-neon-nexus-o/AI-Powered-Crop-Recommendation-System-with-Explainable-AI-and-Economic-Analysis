"""
Economic analysis module for Crop Recommendation System.
Handles ROI calculation, cost-benefit analysis, and risk assessment.
"""

import os
import pandas as pd

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(DATA_PATH, "results")

# Cached data
_economic_df = None
_price_df = None

# Default crop costs (₹/acre)
DEFAULT_COSTS = {
    'seed': 2000,
    'fertilizer': 4000,
    'labor': 5000,
    'misc': 1500
}

# Default yields (quintals/acre)
DEFAULT_YIELDS = {
    'rice': 25, 'wheat': 20, 'maize': 30, 'cotton': 8,
    'chickpea': 10, 'lentil': 8, 'mungbean': 6, 'blackgram': 6,
    'pigeonpeas': 8, 'kidneybeans': 10, 'mothbeans': 5, 'jute': 12,
    'coffee': 5, 'coconut': 100, 'papaya': 200, 'orange': 150,
    'apple': 100, 'mango': 80, 'grapes': 100, 'watermelon': 200,
    'muskmelon': 150, 'banana': 300
}

# Default prices (₹/quintal)
DEFAULT_PRICES = {
    'rice': 2040, 'wheat': 2125, 'maize': 1962, 'cotton': 6080,
    'chickpea': 5230, 'lentil': 5500, 'mungbean': 7755, 'blackgram': 6300,
    'pigeonpeas': 6600, 'kidneybeans': 8000, 'mothbeans': 5500, 'jute': 4750,
    'coffee': 20000, 'coconut': 2500, 'papaya': 1500, 'orange': 3500,
    'apple': 8000, 'mango': 5000, 'grapes': 6000, 'watermelon': 1200,
    'muskmelon': 2000, 'banana': 1500
}


def load_economic_data():
    """Load economic analysis data from CSV."""
    global _economic_df
    
    try:
        _economic_df = pd.read_csv(os.path.join(RESULTS_PATH, "economic_analysis.csv"))
        print("[+] Economic data loaded")
        return _economic_df
    except FileNotFoundError:
        print("[!] Economic CSV not found, using defaults")
        return None


def get_economic_df():
    """Get the economic data DataFrame."""
    return _economic_df


def get_market_price(crop, season='Kharif'):
    """
    Get market price for a crop.
    
    Args:
        crop: Crop name
        season: Agricultural season
    
    Returns:
        float: Price per quintal
    """
    crop_lower = crop.lower()
    
    # Try to get from loaded data
    if _economic_df is not None:
        crop_data = _economic_df[_economic_df['crop'].str.lower() == crop_lower]
        if not crop_data.empty:
            return float(crop_data.iloc[0].get('price', DEFAULT_PRICES.get(crop_lower, 3000)))
    
    return DEFAULT_PRICES.get(crop_lower, 3000)


def calculate_roi(crop, season='Kharif'):
    """
    Calculate Return on Investment for a crop.
    
    Formula:
        revenue = price * yield
        profit = revenue - total_costs
        roi = (profit / total_costs) * 100
    
    Returns:
        dict: ROI calculation breakdown
    """
    crop_lower = crop.lower()
    
    # Get values
    price = get_market_price(crop, season)
    yield_estimate = DEFAULT_YIELDS.get(crop_lower, 20)
    
    # Calculate costs
    costs = get_cost_breakdown(crop)
    total_cost = sum(costs.values())
    
    # Calculate revenue and profit
    revenue = price * yield_estimate
    profit = revenue - total_cost
    roi = (profit / total_cost) * 100 if total_cost > 0 else 0
    profit_margin = (profit / revenue) * 100 if revenue > 0 else 0
    
    return {
        'crop': crop,
        'season': season,
        'price_per_quintal': price,
        'yield_quintals': yield_estimate,
        'revenue': revenue,
        'costs': costs,
        'total_cost': total_cost,
        'profit': profit,
        'roi': roi,
        'profit_margin': profit_margin,
        'risk_category': risk_assessment(crop)['category']
    }


def get_cost_breakdown(crop):
    """
    Get detailed cost breakdown for a crop.
    
    Returns:
        dict: Cost categories and amounts
    """
    crop_lower = crop.lower()
    
    # Adjust costs based on crop type
    multiplier = 1.0
    if crop_lower in ['cotton', 'coffee']:
        multiplier = 1.5
    elif crop_lower in ['rice', 'wheat', 'maize']:
        multiplier = 0.9
    elif crop_lower in ['mango', 'apple', 'grapes']:
        multiplier = 2.0
    
    return {
        'seed': int(DEFAULT_COSTS['seed'] * multiplier),
        'fertilizer': int(DEFAULT_COSTS['fertilizer'] * multiplier),
        'labor': int(DEFAULT_COSTS['labor'] * multiplier),
        'misc': int(DEFAULT_COSTS['misc'] * multiplier)
    }


def cost_benefit_analysis(crops):
    """
    Perform cost-benefit analysis for multiple crops.
    
    Args:
        crops: List of crop names
    
    Returns:
        list: Analysis results for each crop
    """
    results = []
    for crop in crops:
        roi_data = calculate_roi(crop)
        results.append({
            'crop': crop,
            'revenue': roi_data['revenue'],
            'cost': roi_data['total_cost'],
            'profit': roi_data['profit'],
            'roi': roi_data['roi'],
            'benefit_cost_ratio': roi_data['revenue'] / roi_data['total_cost'] if roi_data['total_cost'] > 0 else 0
        })
    
    return results


def rank_by_profitability(crops):
    """
    Rank crops by ROI in descending order.
    
    Args:
        crops: List of crop names
    
    Returns:
        list: Crops sorted by ROI
    """
    analysis = cost_benefit_analysis(crops)
    return sorted(analysis, key=lambda x: x['roi'], reverse=True)


def risk_assessment(crop):
    """
    Assess investment risk for a crop.
    
    Factors considered:
        - Price volatility
        - Weather dependency
        - Market demand
    
    Returns:
        dict: Risk assessment data
    """
    crop_lower = crop.lower()
    
    # Risk categories based on crop characteristics
    high_risk = ['cotton', 'coffee', 'grapes', 'apple']
    medium_risk = ['rice', 'wheat', 'papaya', 'mango', 'orange']
    low_risk = ['chickpea', 'lentil', 'mungbean', 'blackgram', 'maize', 'banana', 'coconut']
    
    if crop_lower in high_risk:
        category = 'High'
        score = 30
        volatility = 25.0
    elif crop_lower in medium_risk:
        category = 'Medium'
        score = 60
        volatility = 15.0
    else:
        category = 'Low'
        score = 85
        volatility = 8.0
    
    return {
        'crop': crop,
        'category': category,
        'risk_score': score,
        'volatility': volatility,
        'factors': {
            'price_stability': 'Stable' if category == 'Low' else 'Variable',
            'weather_dependency': 'Low' if category == 'Low' else 'Moderate' if category == 'Medium' else 'High',
            'market_demand': 'High' if crop_lower in ['rice', 'wheat', 'maize'] else 'Moderate'
        }
    }


def get_economic_summary(crop):
    """
    Get comprehensive economic summary for a crop.
    
    Returns:
        dict: Complete economic data
    """
    roi_data = calculate_roi(crop)
    risk_data = risk_assessment(crop)
    
    return {
        'crop': crop,
        'roi': roi_data['roi'],
        'profit': roi_data['profit'],
        'profit_margin': roi_data['profit_margin'],
        'revenue': roi_data['revenue'],
        'total_cost': roi_data['total_cost'],
        'cost_breakdown': roi_data['costs'],
        'risk_category': risk_data['category'],
        'risk_score': risk_data['risk_score'],
        'volatility': risk_data['volatility']
    }
