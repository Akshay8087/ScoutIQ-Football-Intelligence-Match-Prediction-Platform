"""
ScoutIQ — Utility Functions
Shared helpers for formatting, EDA calculations, and chart generation.
"""

import os, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_currency(value_million) -> str:
    """Format a value in millions EUR to a readable string."""
    try:
        v = float(value_million)
    except (TypeError, ValueError):
        return "N/A"
    if np.isnan(v):
        return "N/A"
    if v >= 1000:
        return f"€{v/1000:.2f}B"
    elif v >= 1:
        return f"€{v:.1f}M"
    else:
        return f"€{v*1000:.0f}K"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value) -> str:
    try:
        v = float(value)
        if v >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        elif v >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:.0f}"
    except Exception:
        return str(value)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_processed_train():
    """Load cleaned training data for Flask dashboard use."""
    path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_train.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback: load and process raw on-the-fly
    from src.data_preprocessing import load_raw_data, clean_and_process
    df = clean_and_process(load_raw_data('train'), is_train=True)
    return df


def load_raw_train():
    path = os.path.join(BASE_DIR, 'data', 'raw', 'fifa_matches_train.csv')
    return pd.read_csv(path)


def load_metadata():
    path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── EDA KPI Calculations ──────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute executive KPI cards from the training dataframe."""
    kpis = {
        'total_matches': len(df),
        'total_teams': df['team_name'].nunique() if 'team_name' in df.columns else 'N/A',
        'total_confederations': df['confederation'].nunique() if 'confederation' in df.columns else 'N/A',
        'avg_player_rating': round(df['avg_player_rating'].mean(), 1),
        'avg_fifa_rank': round(df['fifa_rank'].mean(), 1),
        'avg_market_value': fmt_currency(df['market_value_million_eur'].mean()),
        'total_market_value': fmt_currency(df['market_value_million_eur'].sum()),
        'highest_market_value_team': df.loc[df['market_value_million_eur'].idxmax(), 'team_name'],
        'highest_market_value': fmt_currency(df['market_value_million_eur'].max()),
        'win_rate_avg': round(df['win_rate_last_year'].mean() * 100, 1),
        'avg_goals_scored': round(df['goals_scored_avg'].mean(), 2),
        'avg_goals_conceded': round(df['goals_conceded_avg'].mean(), 2),
        'avg_possession': round(df['possession_avg'].mean(), 1),
        'avg_passing_accuracy': round(df['passing_accuracy'].mean(), 1),
        'elite_teams': int((df['avg_player_rating'] >= 85).sum()),
        'high_form_teams': int((df['recent_form_score'] >= 8).sum()),
        'home_advantage_matches': int(df['host_advantage'].sum()) if 'host_advantage' in df.columns else 0,
        'winners_count': int(df['winner'].sum()) if 'winner' in df.columns else 'N/A',
        'class_balance': round(df['winner'].mean() * 100, 1) if 'winner' in df.columns else 'N/A',
    }
    return kpis


def confederation_summary(df: pd.DataFrame) -> list:
    """Win rate, avg rating, avg market value by confederation."""
    if 'confederation' not in df.columns:
        return []
    g = df.groupby('confederation').agg(
        matches=('team_name', 'count'),
        win_rate=('winner', 'mean'),
        avg_rating=('avg_player_rating', 'mean'),
        avg_market_value=('market_value_million_eur', 'mean'),
        avg_fifa_rank=('fifa_rank', 'mean'),
    ).reset_index()
    g['win_rate_pct'] = (g['win_rate'] * 100).round(1)
    g['avg_rating'] = g['avg_rating'].round(1)
    g['avg_market_value_fmt'] = g['avg_market_value'].apply(lambda x: fmt_currency(float(x)))
    g['avg_fifa_rank'] = g['avg_fifa_rank'].round(1)
    return g.sort_values('win_rate', ascending=False).to_dict(orient='records')


def top_teams_by_metric(df: pd.DataFrame, metric: str, n: int = 10) -> list:
    """Return top N teams by a given metric."""
    if metric not in df.columns or 'team_name' not in df.columns:
        return []
    base_cols = ['team_name', 'confederation', 'avg_player_rating',
                 'fifa_rank', 'market_value_million_eur', 'winner']
    if metric not in base_cols:
        base_cols.insert(2, metric)
    # deduplicate while preserving order
    seen, cols = set(), []
    for c in base_cols:
        if c not in seen and c in df.columns:
            seen.add(c); cols.append(c)
    top = df.nlargest(n, metric)[cols].copy()
    top['market_value_fmt'] = top['market_value_million_eur'].apply(lambda x: fmt_currency(float(x)))
    return top.to_dict(orient='records')


def undervalued_teams(df: pd.DataFrame, n: int = 10) -> list:
    """Teams with high win rate but below-median market value."""
    median_mv = df['market_value_million_eur'].median()
    mask = (df['market_value_million_eur'] < median_mv) & (df['winner'] == 1)
    candidates = df[mask].copy()
    candidates['value_efficiency'] = candidates['win_rate_last_year'] / (
        candidates['market_value_million_eur'] / 1000 + 0.001
    )
    top = candidates.nlargest(n, 'win_rate_last_year')
    top['market_value_fmt'] = top['market_value_million_eur'].apply(lambda x: fmt_currency(float(x)))
    return top[['team_name', 'confederation', 'avg_player_rating', 'fifa_rank',
                'market_value_million_eur', 'market_value_fmt', 'win_rate_last_year',
                'recent_form_score']].to_dict(orient='records')
