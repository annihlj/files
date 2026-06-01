"""
interpretation.py
=================

Interpretation des besten Klassifikationsmodells (Teilfrage 1, Explainable AI).

WARUM Interpretierbarkeit bei Kreditwürdigkeit BESONDERS wichtig:
    - Rechtlich: Betroffene haben (z. B. nach DSGVO Art. 22 / Erwägungsgrund 71) ein
      berechtigtes Interesse an einer nachvollziehbaren Begründung automatisierter
      Entscheidungen mit erheblichen Auswirkungen.
    - Ethisch: Eine Ablehnung muss begründbar sein; "Blackbox lehnt ab" ist nicht
      akzeptabel.
    - Praktisch: Interpretation deckt Fehler/Leakage/Bias auf (z. B. wenn ein Proxy für
      ein sensibles Merkmal dominiert).

VERFAHREN & IHRE GRENZEN:
    1. Impurity-based Feature Importance (nur Tree-Modelle):
       schnell, aber VERZERRT zugunsten hochkardinaler/kontinuierlicher Merkmale und
       instabil bei korrelierten Features. Daher nur ergänzend.
    2. Permutation Importance (modellagnostisch):
       misst den Leistungsabfall, wenn ein Merkmal zufällig permutiert wird. Robuster
       und auf dem TESTSET interpretierbar. GRENZE: bei stark korrelierten Merkmalen
       kann Wichtigkeit "geteilt"/unterschätzt werden.
    3. SHAP (optional): lokale, additive Erklärungen mit soliden spieltheoretischen
       Eigenschaften; rechenintensiv.

KAUSALITÄT: Alle diese Maße sind ASSOZIATIV, nicht kausal. "Wichtig fürs Modell" heißt
NICHT "Ursache des Ausfalls". Ein Merkmal kann wichtig sein, weil es mit der wahren
Ursache korreliert. Diese Unterscheidung ist im Bericht klar zu benennen.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()


def get_feature_names(pipeline) -> list[str]:
    """
    Holt die Feature-Namen NACH der Vorverarbeitung (inkl. One-Hot-Spalten).

    WARUM: Nach dem ColumnTransformer heißen die Spalten anders (z. B. cat__CODE_GENDER_M).
    Für interpretierbare Plots brauchen wir diese Namen.
    """
    pre = pipeline.named_steps["preprocessor"]
    try:
        return pre.get_feature_names_out().tolist()
    except Exception:  # pragma: no cover
        return [f"f{i}" for i in range(pre.transform(pre._validate_data).shape[1])]


def permutation_importances(pipeline, X_test, y_test, n_repeats: int = 10,
                            scoring: str = "average_precision", top: int = 20,
                            save: bool = True) -> pd.DataFrame:
    """
    Berechnet Permutation Importance auf den ORIGINAL-Features (vor Encoding).

    WARUM auf Originalspalten: Wir permutieren die Eingabespalten der GESAMTEN Pipeline,
    sodass die Wichtigkeit pro fachlichem Merkmal (nicht pro One-Hot-Spalte) interpretierbar
    bleibt. Das ist intuitiver für den Bericht.
    """
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, scoring=scoring,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    if save:
        utils.save_table(df.round(5), "permutation_importance", subdir="interpretation")

    top_df = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(top_df))))
    ax.barh(top_df["feature"], top_df["importance_mean"],
            xerr=top_df["importance_std"], color="#4f81bd")
    ax.set_xlabel(f"Permutation Importance ({scoring}-Abfall)")
    ax.set_title(f"Top {top} Merkmale (Permutation Importance)")
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "permutation_importance", subdir="interpretation")
    log.info("Permutation Importance berechnet. Top-3: %s",
             df.head(3)["feature"].tolist())
    return df


def tree_feature_importance(pipeline, top: int = 20, save: bool = True) -> pd.DataFrame | None:
    """
    Impurity-based Feature Importance für Tree-Modelle (ergänzend, mit Vorbehalt).

    Gibt None zurück, wenn das Modell keine feature_importances_ besitzt.
    """
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        log.info("Modell hat keine impurity-based Importances (z. B. LogReg).")
        return None
    names = get_feature_names(pipeline)
    imp = clf.feature_importances_
    df = (pd.DataFrame({"feature": names, "importance": imp})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    if save:
        utils.save_table(df.round(5), "tree_feature_importance", subdir="interpretation")
    return df


def try_shap_summary(pipeline, X_sample, max_display: int = 15, save: bool = True):
    """
    Optionaler SHAP-Summary-Plot. Scheitert leise, wenn SHAP nicht installiert ist.

    WARUM optional/try: SHAP ist eine zusätzliche Abhängigkeit und je nach Modell/Version
    nicht immer verfügbar. Das Projekt soll auch ohne SHAP vollständig laufen.
    """
    try:
        import shap
    except ImportError:
        log.info("SHAP nicht installiert -> übersprungen (optional).")
        return None
    try:
        pre = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        X_trans = pre.transform(X_sample)
        names = get_feature_names(pipeline)
        explainer = shap.Explainer(clf, X_trans, feature_names=names)
        shap_values = explainer(X_trans)
        fig = plt.figure()
        shap.summary_plot(shap_values, X_trans, feature_names=names,
                          max_display=max_display, show=False)
        if save:
            utils.save_figure(fig, "shap_summary", subdir="interpretation")
        return shap_values
    except Exception as exc:  # pragma: no cover
        log.warning("SHAP-Berechnung fehlgeschlagen: %s", exc)
        return None


def plausibility_check(importance_df: pd.DataFrame) -> str:
    """
    Erzeugt eine kurze, textuelle Plausibilitätseinschätzung der Top-Features.

    WARUM: Fachliche Plausibilität ist Teil der wissenschaftlichen Bewertung – sind die
    wichtigen Merkmale inhaltlich sinnvoll (z. B. externe Scores, Verschuldungsquoten)
    oder verdächtig (z. B. eine ID, ein offensichtlicher Proxy)?
    """
    top = importance_df.head(5)["feature"].tolist()
    sensible = [f for f in top if f in config.SENSITIVE_OR_PROXY_FEATURES]
    msg = f"Top-5 Merkmale: {top}. "
    if sensible:
        msg += (f"ACHTUNG: sensible/Proxy-Merkmale unter den wichtigsten: {sensible}. "
                "Im Ethikteil kritisch diskutieren (mögliche indirekte Diskriminierung).")
    else:
        msg += "Keine als sensibel markierten Merkmale unter den Top-5."
    return msg


if __name__ == "__main__":
    from src import preprocessing as prep, train_models as tm
    feat = utils.load_processed("feature_matrix")
    X_train, X_test, y_train, y_test = prep.make_train_test_split(feat)
    fitted = tm.fit_all_models(X_train, y_train)
    # Bestes Tree-Modell für Interpretation (Random Forest in der Demo).
    model = fitted["random_forest"]
    pi = permutation_importances(model, X_test, y_test, n_repeats=5)
    print(pi.head(10).round(4).to_string(index=False))
    print("\nPlausibilität:", plausibility_check(pi))
    try_shap_summary(model, X_test.iloc[:200])
