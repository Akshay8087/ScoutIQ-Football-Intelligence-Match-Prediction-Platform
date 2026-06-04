"""
ScoutIQ — Prediction Pipeline
Loads trained artefacts and provides inference for Flask routes.
"""

import os, json, logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_pipeline():
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    return joblib.load(path)


def load_feature_columns():
    path = os.path.join(MODELS_DIR, 'feature_columns.json')
    with open(path) as f:
        return json.load(f)


def load_metadata():
    path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(path) as f:
        return json.load(f)


class PredictionPipeline:
    """Singleton prediction interface for Flask."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if not self._loaded:
            try:
                self.model = load_pipeline()
                self.feature_cols = load_feature_columns()
                self.metadata = load_metadata()
                self._loaded = True
                logger.info(f"Model loaded: {self.metadata['model_name']}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._loaded = False
        return self._loaded

    def predict(self, input_dict: dict) -> dict:
        """
        Predict win probability from a dict of team attributes.
        Returns {'probability': float, 'verdict': str, 'confidence': str}
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded — run model training first.")

        row = self._dict_to_features(input_dict)
        prob = self.model.predict_proba(row)[0][1]

        if prob >= 0.70:
            verdict = "High Win Probability"
            confidence = "Strong"
            color = "#22c55e"
        elif prob >= 0.55:
            verdict = "Likely Win"
            confidence = "Moderate"
            color = "#84cc16"
        elif prob >= 0.45:
            verdict = "Competitive Match"
            confidence = "Uncertain"
            color = "#f59e0b"
        elif prob >= 0.30:
            verdict = "Likely Loss"
            confidence = "Moderate"
            color = "#f97316"
        else:
            verdict = "High Loss Probability"
            confidence = "Strong"
            color = "#ef4444"

        return {
            'probability': round(float(prob), 4),
            'probability_pct': round(float(prob) * 100, 1),
            'verdict': verdict,
            'confidence': confidence,
            'color': color,
            'loss_probability_pct': round((1 - float(prob)) * 100, 1),
        }

    def _dict_to_features(self, d: dict) -> pd.DataFrame:
        """Convert form dict → feature dataframe aligned with training columns."""
        from src.data_preprocessing import add_derived_features, encode_categoricals
        df = pd.DataFrame([d])

        # Derived features
        total_games = (df['wins_last_10_matches'] + df['losses_last_10_matches'] +
                       df['draws_last_10_matches'])
        df['points_per_match_last_10'] = (
            (df['wins_last_10_matches'] * 3 + df['draws_last_10_matches']) /
            total_games.replace(0, np.nan)
        ).fillna(0)
        df['goal_difference_avg'] = df['goals_scored_avg'] - df['goals_conceded_avg']
        df['performance_index'] = (
            df['avg_player_rating'] * 0.35 +
            df['recent_form_score'] * 3.5 +
            df['win_rate_last_year'] * 10
        )
        df['value_per_cap'] = (
            df['market_value_million_eur'] /
            df['experience_avg_caps'].replace(0, np.nan)
        ).fillna(0)
        df['dominance_score'] = (
            df['possession_avg'] * 0.5 + df['passing_accuracy'] * 0.5
        )
        df['shot_quality'] = df['shots_on_target_ratio'] * df['shots_per_game']
        df['defensive_solidity'] = df['clean_sheets_last_10'] / 10.0
        df['log_market_value'] = np.log1p(df['market_value_million_eur'])

        # Rank tier
        rank = df['fifa_rank'].iloc[0]
        if rank <= 10:
            tier = 'Elite'
        elif rank <= 20:
            tier = 'Strong'
        elif rank <= 35:
            tier = 'Mid'
        else:
            tier = 'Developing'
        df['rank_tier'] = tier

        # Confederation dummies
        for conf in ['AFC', 'CAF', 'CONCACAF', 'CONMEBOL', 'OFC', 'UEFA']:
            df[f'conf_{conf}'] = 1 if d.get('confederation') == conf else 0

        # Rank tier dummies
        for t in ['Developing', 'Elite', 'Mid', 'Strong']:
            df[f'rank_{t}'] = 1 if tier == t else 0

        # Align to training feature columns
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_cols]
