"""
ScoutIQ — Model Training Module
Trains multiple classification models, tunes the best one, and persists artefacts.
"""

import os, sys, json, time, logging
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_raw_data, clean_and_process, get_feature_columns
from feature_engineering import get_all_feature_cols, prepare_X_y

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def build_pipeline(classifier):
    """Wrap classifier in imputer + scaler pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('clf', classifier),
    ])


def get_candidate_models():
    """Return dict of named candidate models."""
    models = {
        'Logistic Regression': build_pipeline(
            LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        ),
        'Random Forest': build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        ),
        'Extra Trees': build_pipeline(
            ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        ),
        'Gradient Boosting': build_pipeline(
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = build_pipeline(
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          random_state=42, eval_metric='logloss', use_label_encoder=False)
        )
    if HAS_LGB:
        models['LightGBM'] = build_pipeline(
            LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
        )
    return models


def evaluate_models(X, y, models, cv=5):
    """Cross-validate all models and return sorted results."""
    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, pipeline in models.items():
        t0 = time.time()
        roc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        acc_scores  = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        f1_scores   = cross_val_score(pipeline, X, y, cv=skf, scoring='f1', n_jobs=-1)
        elapsed = time.time() - t0

        results.append({
            'Model': name,
            'ROC-AUC (CV)': roc_scores.mean(),
            'ROC-AUC Std': roc_scores.std(),
            'Accuracy (CV)': acc_scores.mean(),
            'F1-Score (CV)': f1_scores.mean(),
            'Train Time (s)': round(elapsed, 2),
        })
        logger.info(f"{name:25s} | AUC={roc_scores.mean():.4f} ± {roc_scores.std():.4f} | Acc={acc_scores.mean():.4f}")

    results_df = pd.DataFrame(results).sort_values('ROC-AUC (CV)', ascending=False).reset_index(drop=True)
    return results_df


def tune_best_model(X, y, best_model_name):
    """Hyperparameter tune the best-performing model."""
    logger.info(f"Tuning: {best_model_name}")

    if best_model_name == 'XGBoost' and HAS_XGB:
        base = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        param_dist = {
            'clf__n_estimators': [200, 300, 400],
            'clf__max_depth': [3, 4, 5, 6],
            'clf__learning_rate': [0.02, 0.05, 0.1],
            'clf__subsample': [0.7, 0.8, 0.9],
            'clf__colsample_bytree': [0.7, 0.8, 1.0],
            'clf__min_child_weight': [1, 3, 5],
        }
    elif best_model_name == 'LightGBM' and HAS_LGB:
        base = LGBMClassifier(random_state=42, verbose=-1)
        param_dist = {
            'clf__n_estimators': [200, 300, 400],
            'clf__max_depth': [3, 5, 7, -1],
            'clf__learning_rate': [0.02, 0.05, 0.1],
            'clf__num_leaves': [31, 63, 127],
            'clf__subsample': [0.7, 0.8, 0.9],
        }
    elif best_model_name == 'Random Forest':
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            'clf__n_estimators': [200, 300, 500],
            'clf__max_depth': [None, 10, 15, 20],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features': ['sqrt', 'log2'],
        }
    elif best_model_name == 'Gradient Boosting':
        base = GradientBoostingClassifier(random_state=42)
        param_dist = {
            'clf__n_estimators': [200, 300],
            'clf__max_depth': [3, 4, 5],
            'clf__learning_rate': [0.02, 0.05, 0.1],
            'clf__subsample': [0.7, 0.8, 0.9],
        }
    else:
        base = LogisticRegression(max_iter=1000, random_state=42)
        param_dist = {
            'clf__C': [0.01, 0.1, 1.0, 5.0, 10.0],
            'clf__solver': ['lbfgs', 'saga'],
            'clf__penalty': ['l2'],
        }

    pipeline = build_pipeline(base)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=30, cv=skf,
        scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X, y)
    logger.info(f"Best tuned AUC: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_, search.best_params_


def get_feature_importances(model_pipeline, feature_names):
    """Extract feature importances from the fitted pipeline."""
    clf = model_pipeline.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return fi_df


def save_model_artefacts(best_pipeline, feature_cols, results_df, best_params,
                         best_model_name, cv_auc, X_train, y_train):
    """Persist model, pipeline, metadata, and feature list."""
    # Final fit on full training data
    best_pipeline.fit(X_train, y_train)
    y_pred_prob = best_pipeline.predict_proba(X_train)[:, 1]
    y_pred = best_pipeline.predict(X_train)

    train_auc  = roc_auc_score(y_train, y_pred_prob)
    train_acc  = accuracy_score(y_train, y_pred)
    train_f1   = f1_score(y_train, y_pred)
    train_loss = log_loss(y_train, y_pred_prob)

    # Save pipeline
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'best_model.pkl'))

    # Save feature list
    with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save model comparison table
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)

    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'problem_type': 'Binary Classification',
        'target_variable': 'winner',
        'target_description': 'Predicts probability of a football team winning a match',
        'dataset_size': len(X_train),
        'n_features': len(feature_cols),
        'cv_roc_auc': round(cv_auc, 4),
        'train_roc_auc': round(train_auc, 4),
        'train_accuracy': round(train_acc, 4),
        'train_f1': round(train_f1, 4),
        'train_log_loss': round(train_loss, 4),
        'best_params': best_params,
        'model_comparison': results_df.to_dict(orient='records'),
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Artefacts saved → {MODELS_DIR}")
    logger.info(f"Train AUC: {train_auc:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    return metadata


def run_training():
    """End-to-end training pipeline."""
    # 1. Load & process
    logger.info("=== ScoutIQ Model Training Pipeline ===")
    train_raw = load_raw_data('train')
    train_df  = clean_and_process(train_raw, is_train=True)

    feature_cols = get_all_feature_cols(train_df)
    X, y = prepare_X_y(train_df, feature_cols)

    logger.info(f"Training set: {X.shape[0]} rows × {X.shape[1]} features")
    logger.info(f"Class balance: {np.bincount(y)} (0=Loss/Draw, 1=Win)")

    # 2. Baseline comparison
    logger.info("\n--- Model Comparison (5-Fold CV) ---")
    models = get_candidate_models()
    results_df = evaluate_models(X, y, models, cv=5)
    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))

    # 3. Tune best model
    best_model_name = results_df.iloc[0]['Model']
    cv_auc = results_df.iloc[0]['ROC-AUC (CV)']
    logger.info(f"\nBest model: {best_model_name} (AUC={cv_auc:.4f})")

    tuned_pipeline, tuned_auc, best_params = tune_best_model(X, y, best_model_name)
    logger.info(f"After tuning AUC: {tuned_auc:.4f}")

    # Update comparison table with tuned result
    results_df.loc[results_df['Model'] == best_model_name, 'ROC-AUC (Tuned)'] = tuned_auc

    # 4. Save everything
    metadata = save_model_artefacts(
        tuned_pipeline, feature_cols, results_df,
        best_params, best_model_name, tuned_auc, X, y
    )

    # 5. Generate test predictions
    test_raw = load_raw_data('test')
    test_df  = clean_and_process(test_raw, is_train=False)

    # Align columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    X_test = test_df[feature_cols]

    probs = tuned_pipeline.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({'id': range(len(probs)), 'winner_probability': probs})
    sub_path = os.path.join(BASE_DIR, 'data', 'processed', 'submission.csv')
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved → {sub_path}")

    return metadata


if __name__ == '__main__':
    metadata = run_training()
    print("\n=== Training Complete ===")
    print(f"Model: {metadata['model_name']}")
    print(f"CV AUC: {metadata['cv_roc_auc']}")
    print(f"Train AUC: {metadata['train_roc_auc']}")
