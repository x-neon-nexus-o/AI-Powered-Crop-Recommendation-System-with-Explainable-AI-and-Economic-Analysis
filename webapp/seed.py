"""
Seed recommendation module.
Loads seed variety dataset and ranks varieties using condition-aware feature scoring.
"""

import os
import csv

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "seed_dataset_final_complete.csv")

SEED_FEATURE_COLUMNS = [
    "drought",
    "saline",
    "rainfed",
    "irrigated",
    "disease_resistant",
    "early_maturity",
    "high_yield",
]

_seed_df = None


def _to_int(value):
    """Safely convert values to integer binary flags."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_seed_data():
    """Load seed dataset once and cache it for inference requests."""
    global _seed_df

    if _seed_df is not None:
        return _seed_df

    try:
        with open(SEED_DATA_PATH, mode="r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    except Exception as exc:
        print(f"[!] Seed dataset load failed: {exc}")
        _seed_df = []
        return _seed_df

    normalized_rows = []
    for row in rows:
        clean = dict(row)
        clean["crop_key"] = str(clean.get("Crop", "")).strip().lower()
        clean["state_key"] = str(clean.get("State", "")).strip().lower()
        for col in SEED_FEATURE_COLUMNS:
            clean[col] = _to_int(clean.get(col, 0))
        normalized_rows.append(clean)

    _seed_df = normalized_rows
    print(f"[+] Seed dataset loaded: {len(_seed_df)} rows")
    return _seed_df


def recommend_seeds_for_crop(crop, state=None, conditions=None, top_k=3):
    """
    Recommend top seed varieties for a predicted crop.

    Args:
        crop: Predicted crop name
        state: User state/region (optional)
        conditions: Dict with binary flags for supported seed features
        top_k: Number of varieties to return

    Returns:
        list: Ranked seed variety dictionaries
    """
    seed_df = load_seed_data()
    if not seed_df or not crop:
        return []

    conditions = conditions or {}
    crop_key = str(crop).strip().lower()
    state_key = str(state).strip().lower() if state else None

    filtered = [row for row in seed_df if row.get("crop_key") == crop_key]
    if not filtered:
        return []

    state_used = False
    if state_key:
        state_filtered = [row for row in filtered if row.get("state_key") == state_key]
        if state_filtered:
            filtered = state_filtered
            state_used = True

    # Base scoring: weighted sum of feature alignment + slight preference for high_yield.
    scored_rows = []
    for row in filtered:
        score = 0.0
        for feature in SEED_FEATURE_COLUMNS:
            user_flag = int(bool(conditions.get(feature, 0)))
            score += _to_int(row.get(feature, 0)) * user_flag

        # Always keep a soft yield preference even if user did not explicitly request it.
        if not conditions.get("high_yield", 0):
            score += _to_int(row.get("high_yield", 0)) * 0.25

        scored = dict(row)
        scored["score"] = float(score)
        scored_rows.append(scored)

    scored_rows.sort(key=lambda row: (row.get("score", 0), _to_int(row.get("high_yield", 0))), reverse=True)

    recommendations = []
    for row in scored_rows[:top_k]:
        recommendations.append({
            "category": row.get("Category", ""),
            "crop": row.get("Crop", ""),
            "variety_name": row.get("Name of Variety", ""),
            "variety_type": row.get("Variety/Hybrid", ""),
            "organization": row.get("Sponsoring organization", ""),
            "state": row.get("State", ""),
            "description": row.get("Description", ""),
            "score": float(row.get("score", 0)),
            "feature_flags": {
                feature: _to_int(row.get(feature, 0)) for feature in SEED_FEATURE_COLUMNS
            },
            "state_match": state_used,
        })

    return recommendations
