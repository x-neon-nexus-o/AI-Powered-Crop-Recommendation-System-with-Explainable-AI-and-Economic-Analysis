"""
Fertilizer Recommendation Module
Recommends fertilizers based on crop type and soil/climate conditions
using KNN matching against the Crop and Fertilizer dataset.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

# Module-level state
_fert_df = None
_feature_means = None
_feature_stds = None

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'raw', 'Crop and fertilizer dataset.csv'
)


def load_fertilizer_data():
    """Load the fertilizer dataset and precompute normalization stats."""
    global _fert_df, _feature_means, _feature_stds

    try:
        _fert_df = pd.read_csv(DATA_PATH)
        # Rename columns to match app conventions
        _fert_df = _fert_df.rename(columns={
            'Nitrogen': 'N', 'Phosphorus': 'P', 'Potassium': 'K'
        })
        # Lowercase crop names for case-insensitive matching
        _fert_df['crop_lower'] = _fert_df['Crop'].str.strip().str.lower()

        # Precompute normalization stats for distance calculation
        features = ['N', 'P', 'K', 'pH', 'Rainfall', 'Temperature']
        _feature_means = _fert_df[features].mean()
        _feature_stds = _fert_df[features].std().replace(0, 1)

        print(f"[+] Fertilizer data loaded ({len(_fert_df)} rows, "
              f"{_fert_df['Fertilizer'].nunique()} fertilizers)")
        return _fert_df
    except FileNotFoundError:
        print("[!] Fertilizer dataset not found at: " + DATA_PATH)
        return None
    except Exception as e:
        print(f"[!] Error loading fertilizer data: {e}")
        return None


def recommend_fertilizer(crop, N, P, K, ph, rainfall, temperature):
    """
    Recommend fertilizer for a given crop and soil/climate conditions.

    Uses KNN (K=5) matching: filters by crop, finds closest soil/climate
    matches, returns majority-vote fertilizer.

    Returns dict with keys: name, link, alternatives
    Returns None if crop not in dataset.
    """
    if _fert_df is None:
        return None

    # Filter by crop (case-insensitive)
    crop_lower = crop.strip().lower()
    crop_rows = _fert_df[_fert_df['crop_lower'] == crop_lower]

    if crop_rows.empty:
        return None

    # Compute normalized euclidean distance
    features = ['N', 'P', 'K', 'pH', 'Rainfall', 'Temperature']
    user_values = np.array([N, P, K, ph, rainfall, temperature], dtype=float)
    normalized_user = (user_values - _feature_means[features].values) / _feature_stds[features].values

    dataset_values = crop_rows[features].values.astype(float)
    normalized_dataset = (dataset_values - _feature_means[features].values) / _feature_stds[features].values

    distances = np.sqrt(np.sum((normalized_dataset - normalized_user) ** 2, axis=1))

    # Take K=5 nearest neighbors (or all if fewer rows)
    k = min(5, len(crop_rows))
    nearest_indices = np.argsort(distances)[:k]
    nearest_rows = crop_rows.iloc[nearest_indices]

    # Majority vote for fertilizer
    fert_counts = Counter(nearest_rows['Fertilizer'].values)
    most_common = fert_counts.most_common()

    primary = most_common[0][0]

    # Get the link from the nearest match with this fertilizer
    primary_rows = nearest_rows[nearest_rows['Fertilizer'] == primary]
    link = primary_rows.iloc[0].get('Link', '')
    if pd.isna(link):
        link = ''

    # Alternatives (other fertilizers in top matches)
    alternatives = [f for f, _ in most_common[1:3] if f != primary]

    return {
        'name': primary,
        'link': link,
        'alternatives': alternatives
    }
