
"""
Crop Rotation Planning Module for Flask Web Application
"""

import pickle
import os

# Paths
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')

# Load rotation planner
def load_rotation_planner():
    """Load the saved rotation planner"""
    try:
        planner_path = os.path.join(MODELS_PATH, 'rotation_planner.pkl')
        with open(planner_path, 'rb') as f:
            planner = pickle.load(f)
        return planner
    except FileNotFoundError:
        print("⚠️ Rotation planner not found. Please run Notebook 15 first.")
        return None

# Global planner instance
ROTATION_PLANNER = load_rotation_planner()


def get_rotation_suggestions(current_crop, current_season, num_seasons=3):
    """
    Get rotation suggestions for a given crop and season

    Parameters:
    -----------
    current_crop : str
        Current crop name
    current_season : str
        Current season (Kharif, Rabi, Zaid)
    num_seasons : int
        Number of seasons to plan

    Returns:
    --------
    dict : Rotation plan with sustainability scores
    """
    if ROTATION_PLANNER is None:
        return {'error': 'Rotation planner not loaded'}

    try:
        plan = ROTATION_PLANNER.get_rotation_plan(
            current_crop=current_crop,
            current_season=current_season,
            num_seasons=num_seasons
        )
        return plan
    except Exception as e:
        return {'error': str(e)}


def get_best_successor(current_crop, target_season=None):
    """
    Get the best successor crop

    Parameters:
    -----------
    current_crop : str
        Current crop
    target_season : str, optional
        Target season

    Returns:
    --------
    dict : Best successor details
    """
    if ROTATION_PLANNER is None:
        return {'error': 'Rotation planner not loaded'}

    return ROTATION_PLANNER.get_best_successor(current_crop, target_season)


def get_alternative_plans(current_crop, current_season, num_alternatives=3):
    """
    Get alternative rotation plans

    Parameters:
    -----------
    current_crop : str
        Current crop
    current_season : str
        Current season
    num_alternatives : int
        Number of alternatives

    Returns:
    --------
    list : Alternative rotation plans
    """
    if ROTATION_PLANNER is None:
        return {'error': 'Rotation planner not loaded'}

    return ROTATION_PLANNER.get_alternative_plans(
        current_crop, current_season, num_alternatives
    )


def calculate_sustainability(crop_sequence):
    """
    Calculate sustainability score for a crop sequence

    Parameters:
    -----------
    crop_sequence : list
        List of crops

    Returns:
    --------
    dict : Sustainability scores
    """
    if ROTATION_PLANNER is None:
        return {'error': 'Rotation planner not loaded'}

    return ROTATION_PLANNER.calculate_sustainability_score(crop_sequence)


def track_nutrients(crop_sequence):
    """
    Track soil nutrients for a crop sequence

    Parameters:
    -----------
    crop_sequence : list
        List of crops

    Returns:
    --------
    dict : Nutrient tracking data
    """
    if ROTATION_PLANNER is None:
        return {'error': 'Rotation planner not loaded'}

    return ROTATION_PLANNER.calculate_soil_nutrients(crop_sequence)


def generate_report(current_crop, current_season, num_seasons=3):
    """
    Generate a detailed rotation report

    Parameters:
    -----------
    current_crop : str
        Current crop
    current_season : str
        Current season
    num_seasons : int
        Number of seasons

    Returns:
    --------
    str : Formatted report
    """
    if ROTATION_PLANNER is None:
        return 'Rotation planner not loaded'

    plan = ROTATION_PLANNER.get_rotation_plan(current_crop, current_season, num_seasons)
    return ROTATION_PLANNER.generate_rotation_report(plan)
