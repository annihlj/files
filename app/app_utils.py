"""
app/app_utils.py
================

Hilfsfunktionen für die Streamlit-App. Kapselt das Laden der Artefakte und die
Vorhersage-Logik, damit streamlit_app.py schlank und lesbar bleibt.

WARUM getrennt von src/: Die App ist die "Deployment"-Schicht. Eine dünne Adapterschicht
zwischen gespeicherten Modellen und UI erleichtert Tests und Wiederverwendung.

WICHTIG: Diese App lädt ausschließlich BEREITS GESPEICHERTE Artefakte (Modell,
Clustering, Feature-Matrix). Sie trainiert nichts neu. Voraussetzung ist daher, dass
zuvor Notebook 04 (Modell) und 05 (Clustering) gelaufen sind bzw. die entsprechenden
Trainingsskripte ausgeführt wurden.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Projektwurzel zum Pfad hinzufügen, damit 'src' importierbar ist (App liegt in app/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, utils  # noqa: E402

# Die wenigen Features, die ein Nutzer sinnvoll selbst anpassen kann.
# WARUM nur diese: Das Modell nutzt 60+ Features; ein Formular mit allen wäre unbrauchbar.
# Diese hier sind die fachlich greifbarsten und (laut Interpretation) einflussreichsten.
EDITABLE_FEATURES = {
    "AMT_INCOME_TOTAL": ("Jahreseinkommen (€)", 0.0, 1_000_000.0, 5000.0),
    "AMT_CREDIT": ("Kreditbetrag (€)", 0.0, 2_000_000.0, 5000.0),
    "AMT_ANNUITY": ("Jährliche Rate (€)", 0.0, 200_000.0, 1000.0),
    "age_years": ("Alter (Jahre)", 18.0, 90.0, 1.0),
    "employment_years": ("Beschäftigungsdauer (Jahre)", 0.0, 50.0, 0.5),
    "external_score_mean": ("Externer Bonitätsscore (0-1)", 0.0, 1.0, 0.01),
}


def artifacts_exist() -> tuple[bool, list[str]]:
    """Prüft, ob die nötigen Artefakte vorhanden sind, und meldet fehlende."""
    needed = {
        "Modell (models/best_model.joblib)": config.MODELS_DIR / "best_model.joblib",
        "Clustering (models/clustering.joblib)": config.MODELS_DIR / "clustering.joblib",
    }
    missing = [name for name, path in needed.items() if not path.exists()]
    # Feature-Matrix (Parquet ODER CSV-Fallback) prüfen.
    fm_ok = ((config.PROCESSED_DIR / "feature_matrix.parquet").exists()
             or (config.PROCESSED_DIR / "feature_matrix.csv").exists())
    if not fm_ok:
        missing.append("Feature-Matrix (data/processed/feature_matrix.*)")
    return (len(missing) == 0), missing


def load_artifacts():
    """Lädt Modell, Clustering-Bundle und Feature-Matrix (gecacht durch Streamlit)."""
    import joblib
    model = joblib.load(config.MODELS_DIR / "best_model.joblib")
    clustering = joblib.load(config.MODELS_DIR / "clustering.joblib")
    feature_matrix = utils.load_processed("feature_matrix")
    return model, clustering, feature_matrix


def get_example_people(feature_matrix: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """
    Wählt einige Beispielpersonen aus der Feature-Matrix.

    WARUM: Ein Nutzer kann nicht 60+ Felder ausfüllen. Er wählt eine reale (bzw. im
    Demo-Modus synthetische) Person; alle Features sind dann konsistent vorausgefüllt
    und können punktuell angepasst werden.
    """
    sample = feature_matrix.sample(n=min(n, len(feature_matrix)),
                                   random_state=config.RANDOM_STATE).copy()
    return sample.reset_index(drop=True)


def predict_default_probability(model, person_row: pd.DataFrame) -> float:
    """
    Gibt die vorhergesagte Ausfallwahrscheinlichkeit (Klasse 1) zurück.

    person_row: ein DataFrame mit EINER Zeile, das alle Modell-Eingabespalten enthält
    (ID und TARGET werden ignoriert/entfernt).
    """
    from src import preprocessing as prep
    X = person_row.drop(columns=[c for c in prep.NON_FEATURE_COLUMNS
                                 if c in person_row.columns], errors="ignore")
    proba = model.predict_proba(X)[:, 1]
    return float(proba[0])


def risk_class(probability: float) -> tuple[str, str]:
    """
    Übersetzt eine Wahrscheinlichkeit in eine Risikoklasse + Farbe.

    WARUM Schwellen 0.10/0.25: Diese Grenzen sind BEISPIELHAFT und orientieren sich grob
    an der Basis-Ausfallrate (~8-10 %). Sie sind KEINE geprüften, geschäftlich/regulatorisch
    validierten Schwellen. In der App ist das transparent zu machen.
    """
    if probability < 0.10:
        return "niedrig", "green"
    if probability < 0.25:
        return "mittel", "orange"
    return "hoch", "red"


def assign_cluster(clustering: dict, person_row: pd.DataFrame) -> int:
    """Ordnet die Person einem Cluster zu (mit demselben Preprocessor wie im Training)."""
    features = clustering["features"]
    cols = [c for c in features if c in person_row.columns]
    X = person_row[cols]
    X_scaled = clustering["preprocessor"].transform(X)
    cluster = int(clustering["kmeans"].predict(X_scaled)[0])
    return cluster


def top_influencing_factors(model, feature_matrix: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """
    Liefert die global wichtigsten Einflussfaktoren (Permutation Importance auf einem
    Subsample), für die Anzeige in der App.

    WARUM global statt lokal: Lokale (SHAP-)Erklärungen wären schöner, sind aber
    rechenintensiv und nicht immer verfügbar. Global ist robust und schnell. Die App
    kennzeichnet klar, dass es sich um GLOBALE, ASSOZIATIVE (nicht kausale) Wichtigkeit
    handelt.
    """
    from sklearn.inspection import permutation_importance
    from src import preprocessing as prep

    sample = feature_matrix.sample(n=min(500, len(feature_matrix)),
                                   random_state=config.RANDOM_STATE)
    X, y = prep.split_features_target(sample)
    result = permutation_importance(model, X, y, n_repeats=5,
                                    scoring="average_precision",
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
    df = pd.DataFrame({"Merkmal": X.columns,
                       "Einfluss": result.importances_mean}
                      ).sort_values("Einfluss", ascending=False).head(top)
    return df.reset_index(drop=True)


DISCLAIMER = (
    "Diese Anwendung dient nur zu Demonstrations- und Lernzwecken und darf nicht "
    "für reale Kreditentscheidungen verwendet werden."
)
