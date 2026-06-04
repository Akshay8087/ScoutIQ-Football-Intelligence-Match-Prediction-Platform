"""
ScoutIQ — FIFA Football Intelligence & Match Prediction Platform
Flask Application — Production-Grade Backend
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, abort

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_raw_train, load_metadata, compute_kpis,
    confederation_summary, top_teams_by_metric,
    undervalued_teams, fmt_currency
)
from src.data_preprocessing import clean_and_process
from src.prediction_pipeline import PredictionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'scoutiq-dev-secret')

# ── Singleton prediction pipeline ─────────────────────────────────────────────
predictor = PredictionPipeline()
MODEL_LOADED = predictor.load()

# ── Cached data ────────────────────────────────────────────────────────────────
def get_train_df():
    raw = load_raw_train()
    return clean_and_process(raw, is_train=True)

_df_cache = None
def df():
    global _df_cache
    if _df_cache is None:
        _df_cache = get_train_df()
    return _df_cache

_raw_cache = None
def raw_df():
    global _raw_cache
    if _raw_cache is None:
        _raw_cache = load_raw_train()
    return _raw_cache


# ── Page Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    kpis = compute_kpis(df())
    meta = load_metadata()
    return render_template('index.html', kpis=kpis, meta=meta, model_loaded=MODEL_LOADED)


@app.route('/dashboard')
def dashboard():
    kpis = compute_kpis(df())
    conf_summary = confederation_summary(df())
    top_win = top_teams_by_metric(raw_df(), 'win_rate_last_year', 10)
    top_mv  = top_teams_by_metric(raw_df(), 'market_value_million_eur', 10)
    under   = undervalued_teams(raw_df(), 8)
    return render_template(
        'dashboard.html',
        kpis=kpis,
        conf_summary=conf_summary,
        top_win=top_win,
        top_mv=top_mv,
        undervalued=under,
    )


@app.route('/explorer')
def explorer():
    raw = raw_df()
    confederations = sorted(raw['confederation'].dropna().unique().tolist())
    teams = sorted(raw['team_name'].dropna().unique().tolist())
    return render_template('explorer.html', confederations=confederations, teams=teams)


@app.route('/predict')
def predict_page():
    confederations = ['AFC', 'CAF', 'CONCACAF', 'CONMEBOL', 'OFC', 'UEFA']
    return render_template('predictor.html', confederations=confederations,
                           model_loaded=MODEL_LOADED)


@app.route('/compare')
def compare():
    raw = raw_df()
    teams = sorted(raw['team_name'].dropna().unique().tolist())
    return render_template('compare.html', teams=teams)


@app.route('/model-insights')
def model_insights():
    meta = load_metadata()
    # Feature importance from model
    fi = []
    if MODEL_LOADED and hasattr(predictor.model.named_steps['clf'], 'coef_'):
        import json as _json
        feat_cols = predictor.feature_cols
        coefs = np.abs(predictor.model.named_steps['clf'].coef_[0])
        fi = sorted(
            [{'feature': f, 'importance': round(float(c), 4)} for f, c in zip(feat_cols, coefs)],
            key=lambda x: x['importance'], reverse=True
        )[:20]
    return render_template('model_insights.html', meta=meta, feature_importance=fi)


@app.route('/about')
def about():
    return render_template('about.html')


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.route('/api/dashboard-data')
def api_dashboard_data():
    try:
        kpis = compute_kpis(df())
        conf = confederation_summary(df())
        return jsonify({'kpis': kpis, 'confederation_summary': conf, 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/players')
def api_players():
    """Paginated / filtered player (team) data for the Explorer."""
    try:
        raw = raw_df().copy()
        # Filters
        conf = request.args.get('confederation', '')
        search = request.args.get('search', '').lower()
        min_rating = float(request.args.get('min_rating', 0))
        max_rank = int(request.args.get('max_rank', 9999))
        sort_by = request.args.get('sort_by', 'avg_player_rating')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        if conf:
            raw = raw[raw['confederation'] == conf]
        if search:
            raw = raw[raw['team_name'].str.lower().str.contains(search, na=False)]
        if min_rating:
            raw = raw[raw['avg_player_rating'] >= min_rating]
        if max_rank < 9999:
            raw = raw[raw['fifa_rank'] <= max_rank]

        if sort_by in raw.columns:
            raw = raw.sort_values(sort_by, ascending=(sort_by == 'fifa_rank'))

        total = len(raw)
        start = (page - 1) * per_page
        end = start + per_page
        page_data = raw.iloc[start:end].copy()
        page_data['market_value_fmt'] = page_data['market_value_million_eur'].apply(fmt_currency)

        cols = ['team_name', 'country_code', 'confederation', 'fifa_rank', 'fifa_points',
                'avg_player_rating', 'star_players_count', 'market_value_million_eur',
                'market_value_fmt', 'win_rate_last_year', 'recent_form_score',
                'goals_scored_avg', 'goals_conceded_avg', 'possession_avg',
                'passing_accuracy', 'winner']
        available = [c for c in cols if c in page_data.columns]
        records = page_data[available].fillna(0).to_dict(orient='records')

        return jsonify({
            'data': records,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page,
            'status': 'ok'
        })
    except Exception as e:
        logger.error(f"API /players error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict win probability from JSON payload."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Run python src/model_training.py first.',
                        'status': 'error'}), 503
    try:
        data = request.get_json(force=True)
        required = [
            'fifa_rank', 'fifa_points', 'wins_last_10_matches', 'losses_last_10_matches',
            'draws_last_10_matches', 'win_rate_last_year', 'goals_scored_avg',
            'goals_conceded_avg', 'clean_sheets_last_10', 'shots_per_game',
            'shots_on_target_ratio', 'avg_player_rating', 'star_players_count',
            'market_value_million_eur', 'experience_avg_caps', 'coach_experience_years',
            'recent_form_score', 'possession_avg', 'passing_accuracy', 'host_advantage',
            'travel_distance_avg', 'climate_similarity_score', 'confederation',
        ]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({'error': f"Missing fields: {missing}", 'status': 'error'}), 400

        # Type-cast numeric fields
        numeric_fields = [f for f in required if f != 'confederation']
        for field in numeric_fields:
            data[field] = float(data[field])

        result = predictor.predict(data)
        return jsonify({'result': result, 'status': 'ok'})
    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/compare')
def api_compare():
    """Compare two teams by name."""
    try:
        team_a = request.args.get('team_a', '')
        team_b = request.args.get('team_b', '')
        raw = raw_df()

        def get_team(name):
            rows = raw[raw['team_name'] == name]
            if rows.empty:
                return None
            row = rows.iloc[-1].copy()
            row['market_value_fmt'] = fmt_currency(row['market_value_million_eur'])
            return row.to_dict()

        ta = get_team(team_a)
        tb = get_team(team_b)
        if not ta or not tb:
            return jsonify({'error': 'One or both teams not found.', 'status': 'error'}), 404

        metrics = [
            'avg_player_rating', 'fifa_rank', 'fifa_points', 'win_rate_last_year',
            'goals_scored_avg', 'goals_conceded_avg', 'market_value_million_eur',
            'recent_form_score', 'possession_avg', 'passing_accuracy',
            'shots_per_game', 'clean_sheets_last_10', 'star_players_count',
            'experience_avg_caps', 'coach_experience_years',
        ]
        comparison = {
            'team_a': {k: ta.get(k, 0) for k in ['team_name', 'confederation', 'country_code'] + metrics},
            'team_b': {k: tb.get(k, 0) for k in ['team_name', 'confederation', 'country_code'] + metrics},
            'metrics': metrics,
        }
        comparison['team_a']['market_value_fmt'] = fmt_currency(float(ta.get('market_value_million_eur', 0)))
        comparison['team_b']['market_value_fmt'] = fmt_currency(float(tb.get('market_value_million_eur', 0)))
        return jsonify({'data': comparison, 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', error="Page not found",
                           kpis=compute_kpis(df()), meta=load_metadata(), model_loaded=MODEL_LOADED), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', error="Internal server error",
                           kpis={}, meta={}, model_loaded=MODEL_LOADED), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
