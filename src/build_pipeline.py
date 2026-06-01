"""
build_pipeline.py
=================

Komfort-Skript: führt die komplette Pipeline in EINEM Befehl aus und speichert ALLE
Artefakte, die die Streamlit-App benötigt.

WARUM: Damit die App startklar ist, ohne dass man zwingend alle Notebooks manuell
durchklicken muss. Praktisch für den ersten Start und für eine reproduzierbare Demo.

AUSFÜHRUNG (im Projektwurzelverzeichnis, venv aktiviert):
    python -m src.build_pipeline

Danach:
    streamlit run app/streamlit_app.py

Erzeugt:
    data/processed/feature_matrix.(parquet|csv)   – Modellierungsdatensatz
    models/best_model.joblib                      – getuntes Klassifikationsmodell
    models/clustering.joblib                      – KMeans + Preprocessor + Profile
"""
from __future__ import annotations

import joblib

from . import (clustering as cl, config, data_cleaning, feature_engineering as fe,
               load_data, preprocessing as prep, train_models as tm, utils)

log = utils.get_logger()


def main(use_demo_if_missing: bool = True) -> None:
    # 1) Daten sicherstellen.
    from . import data_download
    if not data_download.verify_raw_data():
        if use_demo_if_missing:
            log.warning("Keine echten Daten gefunden -> erzeuge synthetische Demo-Daten.")
            from . import make_demo_data
            make_demo_data.main()
        else:
            raise FileNotFoundError(
                "Keine Rohdaten in data/raw/. Bitte 'python -m src.data_download' "
                "ausführen oder Daten manuell ablegen."
            )

    # 2) Laden + Cleaning + Feature-Matrix.
    log.info("Schritt 1/4: Daten laden & Feature-Matrix bauen ...")
    tables = load_data.load_all_available()
    tables["application_train"] = data_cleaning.clean_application(tables["application_train"])
    feat = fe.build_feature_matrix(tables)
    utils.save_parquet(feat, "feature_matrix", stage="processed")

    # 3) Klassifikationsmodell trainieren + tunen + speichern.
    log.info("Schritt 2/4: Train-Test-Split & Modelltraining ...")
    X_train, X_test, y_train, y_test = prep.make_train_test_split(feat)
    log.info("Schritt 3/4: Hyperparameter-Tuning (kann etwas dauern) ...")
    best_model, best_params, best_score = tm.tune_best_model(X_train, y_train)
    tm.save_model(best_model, "best_model")
    log.info("Bestes CV-PR-AUC=%.4f | Parameter=%s", best_score, best_params)

    # 4) Clustering trainieren + speichern (inkl. Preprocessor & Profile für die App).
    log.info("Schritt 4/4: Clustering ...")
    labels, profiles, kmeans, cpre = cl.run_clustering(feat, k=4)
    joblib.dump(
        {"kmeans": kmeans, "preprocessor": cpre,
         "features": cl.DEFAULT_CLUSTER_FEATURES, "profiles": profiles},
        config.MODELS_DIR / "clustering.joblib",
    )
    log.info("clustering.joblib gespeichert.")

    log.info("FERTIG. Jetzt App starten mit:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
