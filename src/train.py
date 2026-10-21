"""Students Performance - matematik skoru tahmini (regresyon).

Kullanım:
    python src/train.py                 # reading/writing skorlarıyla
    python src/train.py --no-scores     # yalnızca sosyo-demografik özelliklerle
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "StudentsPerformance.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42

TARGET = "math_score"
ORDINAL_COLS = ["parental_level_of_education"]
NOMINAL_COLS = ["gender", "race_ethnicity", "lunch", "test_preparation_course"]
EDUCATION_ORDER = [[
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree",
]]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """CSV'yi okur ve sütun adlarını kod dostu hale getirir."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("/", "_").str.replace(" ", "_")
    return df


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Ordinal + one-hot + ölçekleme adımlarını tek yapıda toplar."""
    return ColumnTransformer([
        ("ord", OrdinalEncoder(categories=EDUCATION_ORDER), ORDINAL_COLS),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_COLS),
        ("num", StandardScaler(), numeric_cols),
    ])


def compare_models(X_train, y_train, preprocess, cv) -> pd.DataFrame:
    """Birkaç modeli çapraz doğrulama ile karşılaştırır."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        rmse = -cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error")
        rows.append({"model": name, "cv_r2_mean": r2.mean(),
                     "cv_r2_std": r2.std(), "cv_rmse_mean": rmse.mean()})
    return pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)


def tune_ridge(X_train, y_train, preprocess, cv) -> GridSearchCV:
    pipe = Pipeline([("prep", preprocess), ("model", Ridge())])
    grid = GridSearchCV(pipe, {"model__alpha": [0.01, 0.1, 1, 10, 100]},
                        cv=cv, scoring="r2", n_jobs=-1)
    return grid.fit(X_train, y_train)


def save_diagnostic_plots(y_test, y_pred, tag: str) -> None:
    resid = y_test - y_pred
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].scatter(y_test, y_pred, alpha=0.6)
    ax[0].plot([0, 100], [0, 100], "r--")
    ax[0].set(xlabel="Gerçek", ylabel="Tahmin", title="Gerçek vs Tahmin")
    ax[1].scatter(y_pred, resid, alpha=0.6)
    ax[1].axhline(0, color="r", ls="--")
    ax[1].set(xlabel="Tahmin", ylabel="Artık", title="Artık Grafiği")
    sns.histplot(resid, kde=True, ax=ax[2])
    ax[2].set(title="Artık Dağılımı")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"diagnostics_{tag}.png", dpi=120)
    plt.close(fig)


def save_importance_plot(model, X_test, y_test, tag: str) -> None:
    pi = permutation_importance(model, X_test, y_test, n_repeats=20,
                                random_state=RANDOM_STATE, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
    ax = imp.plot(kind="barh", figsize=(7, 4), title="Permütasyon Önemi (R² düşüşü)")
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / f"importance_{tag}.png", dpi=120)
    plt.close(ax.figure)


def main(use_scores: bool) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = "with_scores" if use_scores else "no_scores"

    df = load_data()
    numeric_cols = ["reading_score", "writing_score"] if use_scores else []
    X = df[ORDINAL_COLS + NOMINAL_COLS + numeric_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocess = build_preprocessor(numeric_cols)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison = compare_models(X_train, y_train, preprocess, cv)
    print("\n=== Model karşılaştırması (5-fold CV) ===")
    print(comparison.round(3).to_string(index=False))

    grid = tune_ridge(X_train, y_train, preprocess, cv)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    metrics = {
        "scenario": tag,
        "best_params": grid.best_params_,
        "cv_r2": round(float(grid.best_score_), 4),
        "test_mae": round(mean_absolute_error(y_test, y_pred), 3),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
        "test_r2": round(r2_score(y_test, y_pred), 4),
        "comparison": comparison.round(4).to_dict(orient="records"),
    }
    print("\n=== Test sonuçları ===")
    print({k: v for k, v in metrics.items() if k != "comparison"})

    save_diagnostic_plots(y_test, y_pred, tag)
    save_importance_plot(best, X_test, y_test, tag)
    joblib.dump(best, MODEL_DIR / f"math_score_model_{tag}.pkl")
    (ROOT / "reports" / f"metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-scores", action="store_true",
                        help="reading/writing skorlarını modelden çıkar")
    args = parser.parse_args()
    main(use_scores=not args.no_scores)
