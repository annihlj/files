"""
train_models.py
===============

Training mehrerer Klassifikationsmodelle zur Vorhersage von TARGET.

MODELLAUSWAHL (mit Begründung, warum für tabellarische Kreditrisikodaten geeignet):
    1. DummyClassifier (Baseline):
       WARUM: liefert die Referenz, die jedes echte Modell schlagen muss. Eine
       "most_frequent"-Strategie zeigt drastisch, dass hohe Accuracy bei Imbalance
       nichts wert ist (sie erkennt keinen einzigen Ausfall).
    2. LogisticRegression:
       WARUM: starkes, INTERPRETIERBARES lineares Baseline-Modell. Koeffizienten sind
       als Log-Odds deutbar -> in regulierten Kreditkontexten geschätzt. Profitiert von
       Skalierung.
    3. RandomForest:
       WARUM: erfasst Nichtlinearitäten und Interaktionen, robust gegen Ausreißer und
       Skalierung, wenig Tuning-bedürftig. Guter, stabiler Allrounder für Tabellen.
    4. HistGradientBoosting:
       WARUM: Gradient Boosting ist auf tabellarischen Daten oft state of the art
       (vgl. Grinsztajn et al., 2022). Die Histogramm-Variante ist schnell, behandelt
       fehlende Werte nativ und skaliert gut.
    (Optional LightGBM/XGBoost analog, falls installiert.)

UMGANG MIT IMBALANCE:
    Wo verfügbar nutzen wir class_weight="balanced": die Verlustfunktion gewichtet die
    seltene Positivklasse stärker, sodass das Modell nicht einfach die Mehrheit vorhersagt.
    WARUM nicht primär SMOTE: synthetisches Oversampling (SMOTE) kann unrealistische
    Datenpunkte erzeugen, Leakage verursachen, wenn falsch in die CV eingebettet, und die
    Wahrscheinlichkeitskalibrierung verzerren. class_weight ist einfacher, transparenter
    und ohne Datenverfälschung. SMOTE bleibt eine optionale, kritisch zu diskutierende
    Erweiterung (siehe Diskussion).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    from . import config, preprocessing as prep, utils
except ImportError:  # pragma: no cover
    import config, preprocessing as prep, utils  # type: ignore

log = utils.get_logger()


@dataclass
class ModelSpec:
    """Bündelt einen Modellnamen, den Schätzer und ob er Skalierung braucht."""
    name: str
    estimator: object
    needs_scaling: bool = True
    notes: str = ""


def get_model_specs(random_state: int = config.RANDOM_STATE) -> list[ModelSpec]:
    """
    Definiert die zu trainierenden Modelle.

    needs_scaling steuert, ob der Preprocessor numerische Features skaliert:
    True für lineare/distanzbasierte Modelle, False für Tree-Ensembles.
    """
    specs = [
        ModelSpec(
            "dummy",
            DummyClassifier(strategy="most_frequent"),
            needs_scaling=False,
            notes="Baseline: sagt immer die Mehrheitsklasse vorher.",
        ),
        ModelSpec(
            "logreg",
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=random_state),
            needs_scaling=True,
            notes="Interpretierbares lineares Modell, balanciert.",
        ),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  n_jobs=-1, random_state=random_state),
            needs_scaling=False,
            notes="Nichtlinear, robust, skalierungsunabhängig.",
        ),
        ModelSpec(
            "hist_gbdt",
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          early_stopping=True, n_iter_no_change=15,
                                          random_state=random_state),
            needs_scaling=False,
            notes="Gradient Boosting, oft SOTA auf Tabellen; native NaN-Behandlung.",
        ),
    ]
    # Optional: LightGBM, falls installiert.
    try:
        from lightgbm import LGBMClassifier
        specs.append(ModelSpec(
            "lightgbm",
            LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight="balanced",
                          random_state=random_state, n_jobs=-1, verbose=-1),
            needs_scaling=False,
            notes="LightGBM (optional).",
        ))
    except ImportError:
        log.info("LightGBM nicht installiert -> wird übersprungen (optional).")
    return specs


def build_full_pipeline(spec: ModelSpec, numeric, categorical) -> Pipeline:
    """
    Verbindet Preprocessor und Schätzer zu EINER Pipeline.

    WARUM EIN Objekt: garantiert, dass bei fit/predict/CV der Preprocessor immer nur
    auf den jeweiligen Trainingsdaten ge-fittet wird (kein Leakage), und macht das
    Speichern für die App trivial (Preprocessing + Modell in einem Artefakt).
    """
    pre = prep.build_preprocessor(numeric, categorical, scale_numeric=spec.needs_scaling)
    return Pipeline([("preprocessor", pre), ("classifier", spec.estimator)])


def _stratified_sample(X, y, max_samples: int, random_state: int = config.RANDOM_STATE):
    """
    Gibt einen stratifizierten Subsample zurück, falls len(X) > max_samples.

    WARUM stratifiziert: erhält den Positivanteil (~8 %) auch im Sample, damit
    Imbalance-Effekte korrekt sichtbar bleiben.
    WARUM Subsampling für CV/Tuning: Bei >300k Zeilen dauert jeder Fold mehrere
    Minuten. 30k-50k Zeilen liefern statistisch stabile CV-Schätzer in einem
    Bruchteil der Zeit – die Methodik bleibt identisch.
    """
    if len(X) <= max_samples:
        return X, y
    from sklearn.model_selection import train_test_split as _tts
    _, X_s, _, y_s = _tts(
        X, y,
        test_size=max_samples,
        stratify=y,
        random_state=random_state,
    )
    log.info("Demo-Subsampling: %d → %d Zeilen (stratifiziert)", len(X), len(X_s))
    return X_s, y_s


def cross_validate_models(X_train, y_train, scoring: str = "average_precision",
                          cv_splits: int = 5,
                          max_train_samples: int = 40_000):
    """
    Vergleicht Modelle per stratifizierter Cross-Validation.

    WARUM average_precision (PR-AUC) als Leitmetrik: bei starker Imbalance ist die
    Fläche unter der Precision-Recall-Kurve aussagekräftiger als Accuracy oder selbst
    ROC-AUC, weil sie sich auf die seltene, relevante Positivklasse konzentriert
    (Saito & Rehmsmeier, 2015).
    WARUM StratifiedKFold: erhält den Positivanteil in jedem Fold.
    WARUM max_train_samples: Bei 300k+ Zeilen dauert CV sehr lange; ein stratifizierter
    Subsample (default 40k) liefert statistisch stabile Schätzer in einem Bruchteil der Zeit.
    Für Produktion: max_train_samples=None setzen.

    RÜCKGABE: dict {modellname: (mean_score, std_score)}.
    """
    if max_train_samples:
        X_cv, y_cv = _stratified_sample(X_train, y_train, max_train_samples)
    else:
        X_cv, y_cv = X_train, y_train
    numeric, categorical = prep.identify_feature_types(X_cv)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
    results = {}
    for spec in get_model_specs():
        pipe = build_full_pipeline(spec, numeric, categorical)
        scores = cross_val_score(pipe, X_cv, y_cv, scoring=scoring, cv=cv, n_jobs=-1)
        results[spec.name] = (float(scores.mean()), float(scores.std()))
        log.info("CV %-14s %s = %.4f +/- %.4f", spec.name, scoring,
                 scores.mean(), scores.std())
    return results


def fit_all_models(X_train, y_train) -> dict[str, Pipeline]:
    """Fittet alle Modell-Pipelines auf den vollständigen Trainingsdaten."""
    numeric, categorical = prep.identify_feature_types(X_train)
    fitted = {}
    for spec in get_model_specs():
        pipe = build_full_pipeline(spec, numeric, categorical)
        pipe.fit(X_train, y_train)
        fitted[spec.name] = pipe
        log.info("Gefittet: %s", spec.name)
    return fitted


def tune_best_model(X_train, y_train, scoring: str = "average_precision",
                    max_train_samples: int = 40_000):
    """
    Begrenztes Hyperparameter-Tuning für das Gradient-Boosting-Modell.

    WARUM begrenzter Suchraum: vollständige Suchen sind teuer und überanpassungsgefährdet.
    Ein kleiner, sinnvoll gewählter Raum mit Cross-Validation reicht für ein Portfolio
    und hält die Laufzeit beherrschbar.
    WARUM RandomizedSearch-Alternative erwähnt: bei größeren Räumen wäre RandomizedSearchCV
    effizienter; hier genügt GridSearch über wenige Werte.
    WARUM max_train_samples: Tuning auf dem vollen Datensatz würde mit 24 Fits sehr lange
    dauern; ein stratifizierter Subsample liefert stabile Parameterschätzungen.
    """
    from sklearn.model_selection import GridSearchCV

    if max_train_samples:
        X_tune, y_tune = _stratified_sample(X_train, y_train, max_train_samples)
    else:
        X_tune, y_tune = X_train, y_train
    numeric, categorical = prep.identify_feature_types(X_tune)
    spec = ModelSpec("hist_gbdt",
                     HistGradientBoostingClassifier(early_stopping=True,
                                                    n_iter_no_change=15,
                                                    random_state=config.RANDOM_STATE),
                     needs_scaling=False)
    pipe = build_full_pipeline(spec, numeric, categorical)
    param_grid = {
        "classifier__max_iter": [100, 200],
        "classifier__learning_rate": [0.03, 0.1],
        "classifier__max_leaf_nodes": [31, 63],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    search.fit(X_tune, y_tune)
    log.info("Bestes %s = %.4f bei %s", scoring, search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


def save_model(model, name: str) -> "Path":
    """Speichert ein gefittetes Modell/Pipeline mit joblib."""
    import joblib
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    log.info("Modell gespeichert: %s", path)
    return path


if __name__ == "__main__":
    feat = utils.load_processed("feature_matrix")
    X_train, X_test, y_train, y_test = prep.make_train_test_split(feat)
    print("\n== Cross-Validation (PR-AUC) ==")
    cv_results = cross_validate_models(X_train, y_train)
    print("\n== Finales Tuning (hist_gbdt) ==")
    best, params, score = tune_best_model(X_train, y_train)
    save_model(best, "best_model")
