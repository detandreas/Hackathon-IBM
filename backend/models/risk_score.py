"""
Insurance Risk Scoring System
Calculates 0-100 risk score based on climate and satellite data
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Risk metrics configuration (golden standards)
RISK_METRICS = {
    'NDVI': {
        'name': 'Vegetation Health Index',
        'ideal_min': 0.5, 'ideal_max': 1.0,
        'risk_min': -0.5, 'risk_max': 0.3,
        'direction': 'inverse',
        'weight': 0.20
    },
    'NDMI': {
        'name': 'Soil Moisture Index',
        'ideal_min': 0.1, 'ideal_max': 0.3,
        'risk_min': -0.5, 'risk_max': 0.5,
        'direction': 'moderate',
        'weight': 0.20
    },
    'precip_mm_day': {
        'name': 'Precipitation (Daily)',
        'ideal_min': 10, 'ideal_max': 30,
        'risk_min': 0.1, 'risk_max': 100,
        'direction': 'moderate',
        'weight': 0.20
    },
    'temp_2m_C': {
        'name': '2-meter Temperature',
        'ideal_min': 15, 'ideal_max': 25,
        'risk_min': -10, 'risk_max': 45,
        'direction': 'moderate',
        'weight': 0.20
    },
    'BSI': {
        'name': 'Bare Soil Index',
        'ideal_min': -0.5, 'ideal_max': 0.1,
        'risk_min': 0.2, 'risk_max': 0.5,
        'direction': 'inverse',
        'weight': 0.10
    },
    'NDBI': {
        'name': 'Built-up/Urban Index',
        'ideal_min': -0.5, 'ideal_max': 0,
        'risk_min': 0.1, 'risk_max': 0.4,
        'direction': 'inverse',
        'weight': 0.10
    }
}


def normalize_to_risk_score(value: float, metric_config: dict) -> float:
    """
    Convert metric value to 0-100 risk score
    0 = Safe, 100 = Risky
    """
    ideal_min = metric_config['ideal_min']
    ideal_max = metric_config['ideal_max']
    risk_min = metric_config['risk_min']
    risk_max = metric_config['risk_max']
    direction = metric_config['direction']

    if direction == 'inverse':
        # High values = low risk
        if value >= ideal_max:
            return 0.0
        elif value <= risk_min:
            return 100.0
        else:
            risk_range = ideal_max - risk_min
            current_diff = ideal_max - value
            return (current_diff / risk_range) * 100

    elif direction == 'moderate':
        # Middle range = low risk, extremes = high risk
        if ideal_min <= value <= ideal_max:
            return 0.0
        else:
            if value < ideal_min:
                range_low = ideal_min - risk_min
                diff = ideal_min - value
                return min(100.0, (diff / range_low) * 100)
            else:
                range_high = risk_max - ideal_max
                diff = value - ideal_max
                return min(100.0, (diff / range_high) * 100)

    return 50.0


def calculate_risk_score(metrics: dict) -> tuple:
    """
    Calculate overall area risk score (0-100) from individual metrics

    Args:
        metrics: Dictionary with metric values
                 e.g., {'NDVI': 0.35, 'temp_2m_C': 24.2, ...}

    Returns:
        (overall_score, individual_scores)
    """
    individual_scores = {}
    weighted_sum = 0
    total_weight = 0

    for metric, value in metrics.items():
        if metric in RISK_METRICS:
            config = RISK_METRICS[metric]
            risk_score = normalize_to_risk_score(value, config)
            weight = config['weight']

            individual_scores[metric] = risk_score
            weighted_sum += risk_score * weight
            total_weight += weight

    overall_score = weighted_sum / total_weight if total_weight > 0 else 50.0

    return overall_score, individual_scores


def get_risk_tier(risk_score: float) -> tuple:
    """
    Convert risk score to insurance tier and pricing

    Args:
        risk_score: Risk score 0-100

    Returns:
        (tier_name, pricing_recommendation)
    """
    if risk_score < 20:
        return "TIER 1 - Premium Safe", "Charge 25% less than baseline"
    elif risk_score < 40:
        return "TIER 2 - Safe", "Charge 10% less than baseline"
    elif risk_score < 60:
        return "TIER 3 - Moderate", "Charge baseline rates"
    elif risk_score < 80:
        return "TIER 4 - Elevated Risk", "Charge 20% more than baseline"
    else:
        return "TIER 5 - High Risk", "Charge 40% more than baseline"


def get_current_risk_score() -> dict:
    """
    Calculate current risk score from latest data

    Returns:
        Dictionary with risk assessment
    """
    # Load latest data
    era5 = pd.read_csv(os.path.join(DATA_DIR, 'era5_land_features.csv'))
    sentinel2 = pd.read_csv(os.path.join(DATA_DIR, 'sentinel2_features.csv'))

    latest_era5 = era5.iloc[-1]
    latest_sentinel = sentinel2.iloc[-1]

    # Prepare metrics
    metrics = {
        'NDVI': latest_sentinel['NDVI'],
        'NDBI': latest_sentinel['NDBI'],
        'NDMI': latest_sentinel['NDMI'],
        'BSI': latest_sentinel['BSI'],
        'precip_mm_day': latest_era5['precip_mm_day'],
        'temp_2m_C': latest_era5['temp_2m_C'],
    }

    # Calculate
    overall_score, individual_scores = calculate_risk_score(metrics)
    tier, pricing = get_risk_tier(overall_score)

    return {
        'overall_score': float(overall_score),
        'tier': tier,
        'pricing_recommendation': pricing,
        'individual_scores': {k: float(v) for k, v in individual_scores.items()},
        'metrics': {k: float(v) for k, v in metrics.items()}
    }


if __name__ == '__main__':
    risk = get_current_risk_score()
    print(f"Risk Score: {risk['overall_score']:.1f}/100")
    print(f"Tier: {risk['tier']}")
    print(f"Pricing: {risk['pricing_recommendation']}")
