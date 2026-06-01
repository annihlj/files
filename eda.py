"""
eda.py
======

Wiederverwendbare Funktionen für die explorative Datenanalyse (EDA).

WARUM als Modul (statt nur im Notebook): Reproduzierbarkeit und Testbarkeit. Die
Funktionen speichern Abbildungen automatisch nach reports/figures/, sodass die
Grafiken für den Portfolio-Bericht reproduzierbar erzeugt werden.

METHODISCHER HINWEIS: EDA ist HYPOTHESENGENERIEREND, nicht -bestätigend. Korrelationen
und Gruppenunterschiede in der EDA sind erste Hinweise, KEINE kausalen Belege und keine
gesicherten Modellergebnisse. Alle Interpretationen stehen unter diesem Vorbehalt.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # nicht-interaktives Backend: funktioniert ohne Display (Server/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()
sns.set_theme(style="whitegrid", context="notebook")


def plot_target_distribution(df: pd.DataFrame, target: str = config.TARGET, save: bool = True):
    """
    Balkendiagramm der Zielvariable.

    WAS ZEIGT ES: absolute Häufigkeit der Klassen 0/1.
    WARUM RELEVANT: visualisiert die Klassenungleichheit, die Metrik- und
    Trainingswahl bestimmt.
    INTERPRETATION: Eine kleine Positivklasse (~8-10 %) bedeutet, dass Accuracy als
    Leitmetrik ungeeignet ist.
    GRENZEN: zeigt nur die Randverteilung, nichts über Zusammenhänge mit Merkmalen.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                hue=counts.index.astype(str), palette="muted", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.18)  # Headroom für Beschriftung über dem Balken
    ax.set_xlabel("TARGET (0 = kein Ausfall, 1 = Ausfall)")
    ax.set_ylabel("Anzahl")
    ax.set_title("Verteilung der Zielvariable")
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "target_distribution", subdir="eda")
    return fig


def plot_missingness(df: pd.DataFrame, top: int = 20, save: bool = True):
    """
    Horizontale Balken der Spalten mit dem höchsten Fehlwert-Anteil.

    WARUM RELEVANT: Fehlwerte beeinflussen Imputation und Feature-Auswahl.
    INTERPRETATION: stark fehlende Spalten erfordern Entscheidung (imputieren,
    Indikator, oder ausschließen). GRENZEN: zeigt nicht den Fehl-Mechanismus (MCAR/MAR/MNAR).
    """
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(top)
    if miss.empty:
        log.info("Keine fehlenden Werte zum Plotten.")
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(miss))))
    sns.barplot(x=miss.values * 100, y=miss.index, ax=ax, color="#c0504d")
    ax.set_xlabel("Anteil fehlender Werte (%)")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(miss)} Spalten mit fehlenden Werten")
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "missingness", subdir="eda")
    return fig


def plot_numeric_distributions(df: pd.DataFrame, cols: list[str], save: bool = True):
    """
    Histogramme zentraler numerischer Features.

    WARUM RELEVANT: zeigt Schiefe, Mehrgipfligkeit, Ausreißer.
    INTERPRETATION: stark rechtsschiefe Geldgrößen (Einkommen, Kredit) sprechen für
    Log-/Robust-Transformationen. GRENZEN: univariat, ohne Bezug zum Target.
    """
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        sns.histplot(data, bins=40, ax=ax, color="#4f81bd")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Verteilungen zentraler numerischer Features", y=1.02)
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "numeric_distributions", subdir="eda")
    return fig


def plot_target_comparison(df: pd.DataFrame, cols: list[str],
                           target: str = config.TARGET, save: bool = True):
    """
    Boxplots eines Features getrennt nach Ausfall-/Nicht-Ausfall-Gruppe.

    WARUM RELEVANT: zeigt, ob sich die Gruppen in einem Merkmal unterscheiden
    (erste Hinweise auf prädiktive Kraft).
    INTERPRETATION: deutliche Median-/Streuungsunterschiede deuten auf Relevanz.
    GRENZEN: bivariat, ignoriert Wechselwirkungen; Unterschiede sind korrelativ,
    NICHT kausal.
    """
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=target, y=col, ax=ax, hue=target,
                    palette="muted", legend=False, showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("TARGET")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Ausfall (1) vs. Nicht-Ausfall (0) – zentrale Features", y=1.02)
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "target_comparison_boxplots", subdir="eda")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None,
                             target: str = config.TARGET, save: bool = True):
    """
    Korrelations-Heatmap ausgewählter numerischer Features (inkl. TARGET).

    WARUM RELEVANT: zeigt lineare Zusammenhänge und potenzielle Multikollinearität.
    INTERPRETATION: hohe |Korrelation| mit TARGET = Hinweis auf prädiktive Merkmale;
    hohe Korrelation ZWISCHEN Features = Redundanz/Multikollinearität (für lineare
    Modelle relevant). GRENZEN: erfasst nur LINEARE, monotone Zusammenhänge; nicht-
    lineare Effekte bleiben unsichtbar (Pearson). Korrelation ≠ Kausalität.
    """
    if cols is None:
        num = df.select_dtypes(include=[np.number])
        # Begrenzen, damit die Heatmap lesbar bleibt: stärkste Target-Korrelationen.
        if target in num.columns:
            corr_t = num.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
            cols = corr_t.head(15).index.tolist()
        else:
            cols = num.columns[:15].tolist()
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols), 0.9 * len(cols)))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Korrelations-Heatmap (Pearson)")
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "correlation_heatmap", subdir="eda")
    return fig, corr


def target_correlation_table(df: pd.DataFrame, target: str = config.TARGET, top: int = 20):
    """
    Tabelle der stärksten (Pearson-)Korrelationen mit TARGET.

    WARUM: liefert eine erste, quantitative Rangliste potenziell prädiktiver Merkmale.
    GRENZEN: nur lineare Zusammenhänge; reine EDA-Heuristik, kein Modellergebnis.
    """
    num = df.select_dtypes(include=[np.number])
    if target not in num.columns:
        return pd.DataFrame()
    corr = num.corr(numeric_only=True)[target].drop(labels=[target])
    out = (corr.abs().sort_values(ascending=False).head(top)
           .rename("abs_corr").to_frame())
    out["corr"] = corr.loc[out.index]
    return out.reset_index().rename(columns={"index": "feature"})


if __name__ == "__main__":
    from src import utils as u
    feat = u.load_processed("feature_matrix")
    plot_target_distribution(feat)
    plot_missingness(feat)
    plot_numeric_distributions(feat, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "age_years",
                                      "credit_income_ratio", "external_score_mean",
                                      "INST_LATE_PAYMENT_RATIO"])
    print(target_correlation_table(feat).head(10).to_string(index=False))
    print("EDA-Figuren in reports/figures/eda/ gespeichert.")
