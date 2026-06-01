"""
evaluate_models.py
==================

Evaluation der Klassifikationsmodelle mit für UNAUSGEWOGENE Daten geeigneten Metriken.

WARUM Accuracy allein ungeeignet:
    Bei ~10 % Positiven erreicht ein Modell, das immer "kein Ausfall" sagt, ~90 %
    Accuracy – ohne einen einzigen Ausfall zu erkennen. Accuracy misst hier faktisch
    nur die Klassenverteilung.

WARUM Recall (Sensitivität) wichtig:
    Recall = Anteil der TATSÄCHLICHEN Ausfälle, die erkannt werden. Ein nicht erkannter
    Ausfall (False Negative) bedeutet einen vergebenen, riskanten Kredit -> potenzieller
    finanzieller Verlust.

WARUM Precision ebenfalls relevant:
    Precision = Anteil der als "Ausfall" markierten, die wirklich ausfallen. Niedrige
    Precision bedeutet viele False Positives -> kreditwürdige Personen werden fälschlich
    abgelehnt (Ertragsverlust UND ethisches Problem: ungerechtfertigte Benachteiligung).

WARUM ROC-AUC und PR-AUC:
    - ROC-AUC: Rangordnungsgüte über alle Schwellen, klassenverteilungsunabhängig.
    - PR-AUC / Average Precision: fokussiert die seltene Positivklasse; bei starker
      Imbalance aussagekräftiger als ROC-AUC (Saito & Rehmsmeier, 2015).

DER ZENTRALE TRADE-OFF (technisch UND ethisch):
    - False Negative (riskanter Kredit fälschlich bewilligt): Verlust für den Kreditgeber.
    - False Positive (kreditwürdige Person fälschlich abgelehnt): entgangener Ertrag und
      ggf. unfaire Benachteiligung der Person (Zugang zu Kredit verwehrt).
    Welcher Fehler "schlimmer" ist, ist KEINE rein technische Frage, sondern hängt von
    Geschäftsmodell, Regulierung und gesellschaftlichen Werten ab. Die Schwelle (threshold)
    ist daher eine bewusste, begründungspflichtige Entscheidung – nicht der Default 0.5.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()


def _proba(model, X) -> np.ndarray:
    """Holt Positivklassen-Wahrscheinlichkeiten (robust gegenüber Modelltypen)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Fallback für Schätzer ohne predict_proba (z. B. einige SVMs): Decision-Score.
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def evaluate_model(model, X_test, y_test, name: str, threshold: float = 0.5) -> dict:
    """
    Berechnet das volle Metrik-Set für ein Modell.

    WARUM threshold-Parameter: erlaubt es, die Entscheidungsschwelle bewusst zu
    verschieben (z. B. Recall erhöhen, indem man früher "Ausfall" vorhersagt).
    """
    y_proba = _proba(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    log.info("%-14s | Acc=%.3f P=%.3f R=%.3f F1=%.3f ROC=%.3f PR=%.3f",
             name, metrics["accuracy"], metrics["precision"], metrics["recall"],
             metrics["f1"], metrics["roc_auc"], metrics["pr_auc"])
    return metrics


def compare_models(fitted: dict, X_test, y_test, save: bool = True) -> pd.DataFrame:
    """Erstellt eine Vergleichstabelle aller Modelle, sortiert nach PR-AUC."""
    rows = [evaluate_model(m, X_test, y_test, name) for name, m in fitted.items()]
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    if save:
        utils.save_table(df.round(4), "model_comparison", subdir="modeling")
    return df


def plot_confusion(model, X_test, y_test, name: str, threshold: float = 0.5, save: bool = True):
    """
    Confusion Matrix als Heatmap.

    INTERPRETATION: Die Zellen zeigen TN/FP/FN/TP. Bei Kreditrisiko ist besonders die
    FN-Zelle (übersehene Ausfälle) und die FP-Zelle (fälschlich Abgelehnte) zu beachten.
    """
    y_proba = _proba(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (kein Ausfall)", "1 (Ausfall)"])
    ax.set_yticklabels(["0 (kein Ausfall)", "1 (Ausfall)"])
    ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("Tatsächlich")
    ax.set_title(f"Confusion Matrix – {name}")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save:
        utils.save_figure(fig, f"confusion_{name}", subdir="modeling")
    return fig, cm


def plot_roc_pr_curves(fitted: dict, X_test, y_test, save: bool = True):
    """Zeichnet ROC- und PR-Kurven aller Modelle nebeneinander."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    baseline = y_test.mean()
    for name, model in fitted.items():
        if name == "dummy":
            continue  # Dummy hat keine sinnvolle Kurve
        y_proba = _proba(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ax_pr.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, y_proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("False-Positive-Rate"); ax_roc.set_ylabel("True-Positive-Rate")
    ax_roc.set_title("ROC-Kurven"); ax_roc.legend(fontsize=8)
    ax_pr.axhline(baseline, ls="--", color="k", alpha=0.4,
                  label=f"Zufall (Positivrate={baseline:.2f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall-Kurven"); ax_pr.legend(fontsize=8)
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "roc_pr_curves", subdir="modeling")
    return fig


def plot_calibration(model, X_test, y_test, name: str, n_bins: int = 10, save: bool = True):
    """
    Kalibrierungskurve (Reliability Diagram).

    WARUM relevant: Bei Kreditrisiko interessiert oft die WAHRSCHEINLICHKEIT selbst
    (z. B. erwarteter Verlust), nicht nur die 0/1-Entscheidung. Ein gut kalibriertes
    Modell sagt bei "20 % Ausfallwahrscheinlichkeit" auch tatsächlich in ~20 % der
    Fälle einen Ausfall vorher. INTERPRETATION: Punkte nahe der Diagonale = gut
    kalibriert. GRENZEN: Kalibrierung sagt nichts über Trennschärfe (Ranking) aus.
    """
    from sklearn.calibration import calibration_curve
    y_proba = _proba(model, X_test)
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfekt kalibriert")
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set_xlabel("Mittlere vorhergesagte Wahrscheinlichkeit")
    ax.set_ylabel("Beobachtete Ausfallrate")
    ax.set_title(f"Kalibrierungskurve – {name}")
    ax.legend()
    fig.tight_layout()
    if save:
        utils.save_figure(fig, f"calibration_{name}", subdir="modeling")
    return fig


def threshold_analysis(model, X_test, y_test, name: str,
                       thresholds=(0.5, 0.3, 0.2, 0.15, 0.1), save: bool = True) -> pd.DataFrame:
    """
    Zeigt, wie sich Precision/Recall/F1 mit der Entscheidungsschwelle verändern.

    WARUM zentral: Der Default-Threshold 0.5 ist bei unausgewogenen Daten oft
    UNGEEIGNET – ein konservatives Modell sagt dann fast nie "Ausfall" (Recall ~0).
    Senkt man die Schwelle, erkennt das Modell mehr Ausfälle (Recall steigt), erzeugt
    aber mehr Fehlalarme (Precision sinkt). Die "richtige" Schwelle ist eine
    GESCHÄFTLICHE und ETHISCHE Entscheidung, keine technische Konstante.

    INTERPRETATION für Kreditrisiko: Will man möglichst wenige riskante Kredite
    übersehen (FN minimieren), wählt man eine niedrigere Schwelle und akzeptiert mehr
    fälschliche Ablehnungen (FP) – mit den entsprechenden Fairness-Implikationen.
    """
    y_proba = _proba(model, X_test)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "predicted_positive": int(y_pred.sum()),
        })
    df = pd.DataFrame(rows)
    log.info("Threshold-Analyse (%s):\n%s", name, df.round(3).to_string(index=False))
    if save:
        utils.save_table(df.round(4), f"threshold_analysis_{name}", subdir="modeling")
    return df



def full_classification_report(model, X_test, y_test, name: str) -> str:
    """Gibt den sklearn-Classification-Report als Text zurück (und loggt ihn)."""
    y_pred = (_proba(model, X_test) >= 0.5).astype(int)
    report = classification_report(y_test, y_pred,
                                   target_names=["kein Ausfall (0)", "Ausfall (1)"],
                                   zero_division=0)
    return report


if __name__ == "__main__":
    from src import preprocessing as prep, train_models as tm
    feat = utils.load_processed("feature_matrix")
    X_train, X_test, y_train, y_test = prep.make_train_test_split(feat)
    fitted = tm.fit_all_models(X_train, y_train)
    print("\n== Modellvergleich ==")
    comp = compare_models(fitted, X_test, y_test)
    print(comp.round(4).to_string(index=False))
    # Bestes Nicht-Dummy-Modell für Detailplots.
    best_name = comp[comp.model != "dummy"].iloc[0]["model"]
    plot_confusion(fitted[best_name], X_test, y_test, best_name)
    plot_roc_pr_curves(fitted, X_test, y_test)
    plot_calibration(fitted[best_name], X_test, y_test, best_name)
    print(f"\nClassification Report ({best_name}):")
    print(full_classification_report(fitted[best_name], X_test, y_test, best_name))
