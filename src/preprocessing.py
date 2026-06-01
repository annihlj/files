"""
preprocessing.py
================

Robuste, leakage-sichere Vorverarbeitung mit scikit-learn.

LEITPRINZIP (Leakage-Vermeidung):
    Jede Transformation, die PARAMETER aus den Daten schätzt (Imputations-Mediane,
    Skalierungsstatistiken, One-Hot-Kategorien), wird ausschließlich auf den
    TRAININGSDATEN ge-fittet und dann auf Test-/Neudaten nur noch ANGEWENDET
    (transform). Würde man auf dem Gesamtdatensatz fitten, flösse Information aus dem
    Testset in die Vorverarbeitung -> zu optimistische, nicht generalisierende Ergebnisse.
    Deshalb: erst Split, DANN fit auf X_train.

WARUM eine sklearn-Pipeline / ColumnTransformer:
    - Kapselt alle Schritte in EIN Objekt -> beim Speichern (joblib) ist die komplette
      Vorverarbeitung reproduzierbar mitgesichert (wichtig für die Streamlit-App).
    - Verhindert Leakage automatisch bei Cross-Validation (fit nur auf jeweiligem Fold).
    - Trennt numerische und kategoriale Behandlung sauber.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()

# Spalten, die KEINE Modellfeatures sind (IDs, Ziel, reine Hilfsspalten).
NON_FEATURE_COLUMNS = [config.ID_COLUMN, config.TARGET]


def split_features_target(df: pd.DataFrame, target: str = config.TARGET):
    """
    Trennt Features (X) und Ziel (y) und entfernt Nicht-Feature-Spalten.

    WARUM die ID entfernen: SK_ID_CURR ist ein willkürlicher Identifikator ohne
    prädiktive Bedeutung. Bliebe er im Feature-Satz, könnte ein Modell ihn
    fälschlich "auswendig lernen" (Overfitting/Pseudo-Leakage).
    """
    y = df[target].astype(int)
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Teilt Spalten in numerisch und kategorial auf.

    WARUM: numerische und kategoriale Merkmale brauchen unterschiedliche Behandlung
    (Skalierung vs. Encoding). Die Aufteilung erfolgt datentyp-basiert.
    """
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()
    log.info("Feature-Typen: %d numerisch, %d kategorial.", len(numeric), len(categorical))
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str],
                       scale_numeric: bool = True) -> ColumnTransformer:
    """
    Baut den ColumnTransformer.

    NUMERISCHE Features:
        - SimpleImputer(strategy="median"): fehlende Werte durch den Median ersetzen.
          WARUM Median statt Mittelwert: robuster gegen die starke Rechtsschiefe und
          Ausreißer der Geldgrößen (Einkommen/Kredit).
        - StandardScaler (optional): zentriert/skaliert auf Mittelwert 0, Varianz 1.
          WARUM nötig für lineare Modelle (Logistic Regression mit Regularisierung)
          und distanzbasierte Verfahren (K-Means, PCA): diese reagieren empfindlich
          auf unterschiedliche Wertebereiche; ohne Skalierung dominieren großskalige
          Merkmale (z. B. Einkommen in Zehntausenden) gegenüber kleinskaligen (Ratios).
          WARUM bei TREE-basierten Modellen NICHT nötig: Entscheidungsbäume splitten
          anhand von Schwellenwerten je Merkmal; monotone Transformationen ändern die
          Split-Reihenfolge nicht -> Skalierung ist für RF/Gradient Boosting irrelevant.
          Deshalb ist scale_numeric für Tree-Modelle abschaltbar (spart unnötige Schritte).

    KATEGORIALE Features:
        - SimpleImputer(strategy="most_frequent"): fehlende Kategorien durch Modus.
        - OneHotEncoder(handle_unknown="ignore"): wandelt Kategorien in 0/1-Spalten.
          WARUM handle_unknown="ignore": zur Inferenzzeit unbekannte Kategorien führen
          nicht zum Absturz, sondern zu einem Null-Vektor (wichtig für die App).
          ALTERNATIVE: Ordinal-/Target-Encoding (kompakter, aber Target-Encoding birgt
          Leakage-Risiko und braucht sorgfältige CV-Einbettung). One-Hot ist transparent
          und für moderate Kardinalität die sichere Standardwahl.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",  # alles Nicht-Spezifizierte verwerfen (Sicherheit)
    )
    return preprocessor


def make_train_test_split(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                          random_state: int = config.RANDOM_STATE):
    """
    Stratifizierter Train-Test-Split.

    WARUM stratify=y: erhält den Positivanteil (~10 %) in beiden Mengen. Ohne
    Stratifizierung könnte der seltene Ausfall im Testset zufällig über-/
    unterrepräsentiert sein -> verzerrte, instabile Evaluation.
    WARUM fester random_state: Reproduzierbarkeit des Splits.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info("Split: Train=%d (Positivrate %.3f), Test=%d (Positivrate %.3f)",
             len(X_train), y_train.mean(), len(X_test), y_test.mean())
    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor, name: str = "preprocessor") -> "Path":
    """Speichert einen (gefitteten) Preprocessor mit joblib nach models/."""
    import joblib
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(preprocessor, path)
    log.info("Preprocessor gespeichert: %s", path)
    return path


if __name__ == "__main__":
    feat = utils.load_processed("feature_matrix")
    X_train, X_test, y_train, y_test = make_train_test_split(feat)
    num, cat = identify_feature_types(X_train)
    pre = build_preprocessor(num, cat, scale_numeric=True)
    # WICHTIG: fit NUR auf Trainingsdaten.
    Xt = pre.fit_transform(X_train)
    print("Transformierte Trainingsmatrix:", Xt.shape)
    print("Erste numerische Spalten:", num[:5])
    print("Erste kategoriale Spalten:", cat[:5])
