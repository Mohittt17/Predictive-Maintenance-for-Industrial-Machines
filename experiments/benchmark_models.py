"""
Empirical Model Benchmark Script for Predictive Maintenance Platform.

Generates realistic sensor dataset with imbalanced degradation features (15:1 ratio),
fits all candidate models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting),
evaluates metrics (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC), and outputs a verified json/markdown summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np

from src.failure_prediction.logistic_regression import LogisticRegressionPredictor
from src.failure_prediction.random_forest import RandomForestPredictor
from src.failure_prediction.xgboost_model import XGBoostPredictor
from src.failure_prediction.gradient_boosting import GradientBoostingPredictor
from src.failure_prediction.evaluator import compare_models, EvaluationReport


def generate_benchmark_dataset(n_samples: int = 2500, anomaly_ratio: float = 0.08, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_pos = int(n_samples * anomaly_ratio)
    n_neg = n_samples - n_pos

    # Normal operation sensors (15 features: time, frequency, wavelet, environmental)
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, 15))
    
    # Degraded operation with realistic overlapping noise
    X_pos = rng.normal(loc=1.1, scale=1.4, size=(n_pos, 15))
    X_pos[:, 0] += rng.normal(0.9, 0.5, size=n_pos)  # vib_rms
    X_pos[:, 1] += rng.normal(1.2, 0.8, size=n_pos)  # vib_kurtosis
    X_pos[:, 5] += rng.normal(0.8, 0.4, size=n_pos)  # temp_max
    X_pos[:, 7] += rng.normal(0.9, 0.6, size=n_pos)  # acou_rms

    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * n_neg + [1] * n_pos)

    # Add 2% label noise to simulate real-world sensor tagging ambiguity
    noisy_indices = rng.choice(len(y), size=int(0.02 * len(y)), replace=False)
    y[noisy_indices] = 1 - y[noisy_indices]

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def run_benchmark():
    print("=" * 70)
    print("RUNNING EMPIRICAL MODEL BENCHMARK FOR PREDICTIVE MAINTENANCE PLATFORM")
    print("=" * 70)

    X, y = generate_benchmark_dataset(n_samples=2500, anomaly_ratio=0.08, seed=42)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Dataset split: Train={len(X_train)} ({sum(y_train)} failures), Test={len(X_test)} ({sum(y_test)} failures)")

    feature_names = [
        "vib_rms", "vib_kurtosis", "vib_crest_factor", "vib_skewness", "vib_peak_to_peak",
        "temp_max", "temp_mean", "acou_rms", "acou_kurtosis", "curr_rms",
        "pres_mean", "bearing_band1_energy", "bearing_band2_energy", "imbalance_1x", "imbalance_2x"
    ]

    models = [
        ("Logistic Regression", LogisticRegressionPredictor(C=1.0, class_weight="balanced")),
        ("Random Forest", RandomForestPredictor(n_estimators=200, class_weight="balanced_subsample")),
        ("XGBoost", XGBoostPredictor(n_estimators=200, max_depth=5, learning_rate=0.05)),
        ("Gradient Boosting", GradientBoostingPredictor(n_estimators=200, max_depth=5, learning_rate=0.05)),
    ]

    reports: list[EvaluationReport] = []
    results_dict = {}

    for name, model in models:
        model.fit(X_train, y_train, feature_names=feature_names)
        report = model.evaluate(X_test, y_test)
        reports.append(report)
        results_dict[name] = {
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "f1": round(report.f1, 4),
            "pr_auc": round(report.pr_auc, 4),
            "roc_auc": round(report.roc_auc, 4),
            "false_alarm_rate": round(report.false_alarm_rate, 4),
            "missed_failure_rate": round(report.missed_failure_rate, 4),
            "confusion_matrix": {"tp": report.tp, "fp": report.fp, "tn": report.tn, "fn": report.fn}
        }

    print("\nBENCHMARK RESULTS SUMMARY:")
    print(compare_models(reports))

    out_dir = Path(__file__).resolve().parents[1] / "logs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nSaved empirical benchmark results to {out_dir / 'benchmark_results.json'}")
    return results_dict


if __name__ == "__main__":
    run_benchmark()
