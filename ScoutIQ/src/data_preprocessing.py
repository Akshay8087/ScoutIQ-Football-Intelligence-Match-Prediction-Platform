"""
ScoutIQ — Data Preprocessing Module
Handles data loading, cleaning, and preparation for the football match prediction pipeline.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def load_raw_data(split='train'):
    """Load raw CSV data for train or test split."""
    filename = 'fifa_matches_train.csv' if split == 'train' else 'fifa_matches_test.csv'
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    logger.info(f"Loaded {split} data: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def standardise_columns(df):
    """Rename columns to consistent snake_case."""
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def add_derived_features(df):
    """
    Football-logic feature engineering:
    - points_per_match_last_10: weighted recent form
    - goal_difference_avg: attacking minus defensive balance
    - performance_index: composite rating × form
    - value_per_cap: squad value efficiency
    - dominance_score: possession + pass accuracy composite
    - experience_score: caps weighted by rating
    - age_category: youth / prime / veteran squad
    - rank_tier: elite / strong / mid / developing
    """
    df = df.copy()

    # Recent 10-match point rate (win=3, draw=1, loss=0)
    total_games = df['wins_last_10_matches'] + df['losses_last_10_matches'] + df['draws_last_10_matches']
    df['points_per_match_last_10'] = (
        (df['wins_last_10_matches'] * 3 + df['draws_last_10_matches']) /
        total_games.replace(0, np.nan)
    ).fillna(0)

    # Attacking vs defensive balance
    df['goal_difference_avg'] = df['goals_scored_avg'] - df['goals_conceded_avg']

    # Composite performance index
    df['performance_index'] = (
        df['avg_player_rating'] * 0.35 +
        df['recent_form_score'] * 3.5 +
        df['win_rate_last_year'] * 10
    )

    # Squad value efficiency (value per experienced cap)
    df['value_per_cap'] = (
        df['market_value_million_eur'] /
        df['experience_avg_caps'].replace(0, np.nan)
    ).fillna(0)

    # Technical dominance (possession + passing)
    df['dominance_score'] = (
        df['possession_avg'] * 0.5 +
        df['passing_accuracy'] * 0.5
    )

    # Shots quality ratio
    df['shot_quality'] = df['shots_on_target_ratio'] * df['shots_per_game']

    # Defensive solidity: clean sheets normalised over 10
    df['defensive_solidity'] = df['clean_sheets_last_10'] / 10.0

    # FIFA rank tier (lower rank = better)
    df['rank_tier'] = pd.cut(
        df['fifa_rank'],
        bins=[0, 10, 20, 35, 50],
        labels=['Elite', 'Strong', 'Mid', 'Developing'],
        right=True
    ).astype(str)

    # Log-transformed market value (reduces skewness)
    df['log_market_value'] = np.log1p(df['market_value_million_eur'])

    logger.info("Feature engineering complete — added 10 derived features")
    return df


def encode_categoricals(df):
    """One-hot encode confederation and rank_tier."""
    df = df.copy()
    confederation_dummies = pd.get_dummies(df['confederation'], prefix='conf', drop_first=False)
    rank_tier_dummies = pd.get_dummies(df['rank_tier'], prefix='rank', drop_first=False)
    df = pd.concat([df, confederation_dummies, rank_tier_dummies], axis=1)
    return df


def get_feature_columns(df):
    """Return the exact list of feature columns used for modelling."""
    drop_cols = [
        'team_name', 'country_code', 'confederation', 'rank_tier', 'winner'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def clean_and_process(df, is_train=True):
    """Full preprocessing pipeline for a dataframe."""
    df = standardise_columns(df)
    df = add_derived_features(df)
    df = encode_categoricals(df)

    if is_train:
        df.dropna(subset=['winner'], inplace=True)
        df['winner'] = df['winner'].astype(int)

    return df


def save_processed(df, filename='cleaned_fifa_matches.csv'):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data → {path}")


if __name__ == '__main__':
    train_df = load_raw_data('train')
    test_df = load_raw_data('test')

    train_clean = clean_and_process(train_df, is_train=True)
    test_clean = clean_and_process(test_df, is_train=False)

    save_processed(train_clean, 'cleaned_train.csv')
    save_processed(test_clean, 'cleaned_test.csv')

    print(f"Train processed: {train_clean.shape}")
    print(f"Test processed:  {test_clean.shape}")
    print("Feature columns:", get_feature_columns(train_clean))
