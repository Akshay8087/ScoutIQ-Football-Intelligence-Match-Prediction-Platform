"""
ScoutIQ — Feature Engineering Module
Advanced football-domain feature construction for match prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

# Numeric features used in modelling (after engineering)
NUMERIC_FEATURES = [
    'fifa_rank', 'fifa_points', 'wins_last_10_matches', 'losses_last_10_matches',
    'draws_last_10_matches', 'win_rate_last_year', 'goals_scored_avg',
    'goals_conceded_avg', 'clean_sheets_last_10', 'shots_per_game',
    'shots_on_target_ratio', 'avg_player_rating', 'star_players_count',
    'market_value_million_eur', 'experience_avg_caps', 'coach_experience_years',
    'recent_form_score', 'possession_avg', 'passing_accuracy', 'host_advantage',
    'travel_distance_avg', 'climate_similarity_score',
    # Engineered
    'points_per_match_last_10', 'goal_difference_avg', 'performance_index',
    'value_per_cap', 'dominance_score', 'shot_quality', 'defensive_solidity',
    'log_market_value',
]

DUMMY_PREFIXES = ['conf_', 'rank_']


def get_all_feature_cols(df):
    """Collect all numeric + dummy columns present in df, excluding string/object columns."""
    dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in DUMMY_PREFIXES)]
    numeric_present = [c for c in NUMERIC_FEATURES if c in df.columns]
    all_cols = numeric_present + dummy_cols
    # Only keep columns that are actually numeric
    numeric_only = [c for c in all_cols if df[c].dtype in ['int64', 'float64', 'bool', 'int32', 'uint8']]
    return numeric_only


def build_sklearn_pipeline():
    """
    Returns a sklearn Pipeline:
      1. SimpleImputer (median) — handles any remaining NaNs
      2. RobustScaler — robust to outliers (better than StandardScaler for football data)
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])
    return pipeline


def prepare_X_y(df, feature_cols):
    """Extract X (features) and y (target) arrays."""
    X = df[feature_cols].copy()
    y = df['winner'].values if 'winner' in df.columns else None
    return X, y


def get_feature_importance_names(feature_cols):
    """Human-readable mapping for feature column names."""
    mapping = {
        'fifa_rank': 'FIFA Rank',
        'fifa_points': 'FIFA Points',
        'wins_last_10_matches': 'Wins (Last 10)',
        'losses_last_10_matches': 'Losses (Last 10)',
        'draws_last_10_matches': 'Draws (Last 10)',
        'win_rate_last_year': 'Win Rate (Last Year)',
        'goals_scored_avg': 'Avg Goals Scored',
        'goals_conceded_avg': 'Avg Goals Conceded',
        'clean_sheets_last_10': 'Clean Sheets (Last 10)',
        'shots_per_game': 'Shots Per Game',
        'shots_on_target_ratio': 'Shots on Target %',
        'avg_player_rating': 'Avg Player Rating',
        'star_players_count': 'Star Players',
        'market_value_million_eur': 'Squad Market Value (€M)',
        'experience_avg_caps': 'Avg International Caps',
        'coach_experience_years': 'Coach Experience (Yrs)',
        'recent_form_score': 'Recent Form Score',
        'possession_avg': 'Avg Possession %',
        'passing_accuracy': 'Passing Accuracy %',
        'host_advantage': 'Home Advantage',
        'travel_distance_avg': 'Avg Travel Distance',
        'climate_similarity_score': 'Climate Similarity',
        'points_per_match_last_10': 'Points/Match (Last 10)',
        'goal_difference_avg': 'Goal Difference Avg',
        'performance_index': 'Performance Index',
        'value_per_cap': 'Value per Cap (€M)',
        'dominance_score': 'Technical Dominance',
        'shot_quality': 'Shot Quality Index',
        'defensive_solidity': 'Defensive Solidity',
        'log_market_value': 'Log Market Value',
    }
    return [mapping.get(c, c) for c in feature_cols]
