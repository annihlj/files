"""
clustering.py
=============

Unüberwachte Segmentierung der Antragsteller (Teilfrage 4).

ABGRENZUNG ZUR KLASSIFIKATION (sehr wichtig):
    Clustering ist ein UNÜBERWACHTES Verfahren: Es erhält KEINE Zielvariable, sondern
    sucht Struktur allein in den Merkmalen. Wir verwenden TARGET hier bewusst NICHT als
    Eingabe. WARUM: Würden wir TARGET ins Clustering geben, würden wir die Antwort
    "hineinfüttern" – die entstehenden Gruppen wären dann keine eigenständige Segmentierung
    mehr, sondern eine triviale Nachbildung der Zielvariable (zirkulär). Stattdessen
    bilden wir Cluster nur aus Merkmalen und vergleichen die Ausfallrate je Cluster ERST
    NACHTRÄGLICH – das ist eine valide, aussagekräftige Auswertung.

WARUM Skalierung zwingend:
    K-Means und PCA sind DISTANZ-/VARIANZ-basiert. Unskalierte großskalige Merkmale
    (Einkommen in Zehntausenden) würden die Distanz dominieren und kleinskalige (Ratios)
    faktisch ignorieren. Standardisierung stellt alle Merkmale gleich.

WARUM PCA für Visualisierung:
    Cluster leben im hochdimensionalen Raum; wir können sie nicht direkt sehen. PCA
    projiziert auf 2 Dimensionen maximaler Varianz, um die Trennung visuell zu prüfen.
    GRENZE: 2 Komponenten erklären nur einen Teil der Varianz – Überlappung in 2D heißt
    nicht zwingend Überlappung im vollen Raum.
"""
from __future__ import annotations

import os
# Unterdrückt den WinError 2 / wmic-Fehler: joblib kann auf neuem Windows 11 die
# physischen Kerne nicht per wmic ermitteln. Wir geben die logische Kernzahl explizit
# vor, damit kein subprocess gestartet wird.
import multiprocessing
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(multiprocessing.cpu_count()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()

# Bewusst ausgewählte, fachlich interpretierbare Merkmale für die Segmentierung.
# WARUM diese Auswahl: Sie decken Einkommen, Verschuldung, Alter, Bonität, externe
# Historie und Zahlungsverhalten ab – die zentralen Dimensionen, in denen sich
# Antragsteller fachlich sinnvoll unterscheiden. KEINE TARGET-abhängigen Spalten.
DEFAULT_CLUSTER_FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "credit_income_ratio", "annuity_income_ratio",
    "age_years", "employment_years", "external_score_mean",
    "BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "PREV_COUNT",
    "INST_LATE_PAYMENT_RATIO", "missing_values_count",
]


def select_cluster_features(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Wählt die Clustering-Merkmale aus (nur vorhandene)."""
    features = features or DEFAULT_CLUSTER_FEATURES
    cols = [c for c in features if c in df.columns]
    assert config.TARGET not in cols, "TARGET darf nicht ins Clustering!"
    log.info("Clustering nutzt %d Merkmale: %s", len(cols), cols)
    return df[cols].copy()


def build_clustering_preprocessor() -> Pipeline:
    """
    Imputation (Median) + Standardisierung für das Clustering.

    WARUM auch hier Imputation: K-Means kann nicht mit NaN umgehen. Median ist robust.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9), save: bool = True,
                   silhouette_max_samples: int = 8_000):
    """
    Bestimmt eine geeignete Clusterzahl per Elbow-Methode (Inertia) und Silhouette.

    WARUM zwei Kriterien:
        - Inertia (within-cluster sum of squares) fällt monoton mit k; der "Ellenbogen"
          (Knick) deutet auf ein gutes k. Subjektiv ablesbar.
        - Silhouette Score misst, wie gut Punkte zum eigenen vs. nächsten Cluster passen
          (-1..1, höher = besser). Objektiver, aber rechenintensiver.
    GRENZE: Beide sind Heuristiken; "die" wahre Clusterzahl existiert i. d. R. nicht.
    Fachliche Interpretierbarkeit schlägt im Zweifel die Kennzahl.
    WARUM silhouette_max_samples: silhouette_score ist O(n²) – bei 300k Zeilen würde
    es Stunden dauern. Ein zufälliger Subsample von 8k Punkten liefert statistisch
    stabile Scores in Sekunden. KMeans wird aber auf dem VOLLEN Datensatz gefittet,
    damit die Inertia und die Zentroide korrekt sind.
    """
    inertias, silhouettes = [], []
    ks = list(k_range)
    rng = np.random.default_rng(config.RANDOM_STATE)
    # Subsample-Indizes für Silhouette (einmal bestimmen, für alle k gleich)
    n = X_scaled.shape[0]
    sil_idx = rng.choice(n, size=min(silhouette_max_samples, n), replace=False)
    if n > silhouette_max_samples:
        log.info("Silhouette-Subsampling: %d → %d Zeilen (KMeans auf vollen Daten)", n, len(sil_idx))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        # Silhouette nur auf Subsample berechnen
        sil = silhouette_score(X_scaled[sil_idx], labels[sil_idx])
        silhouettes.append(sil)
        log.info("k=%d | Inertia=%.0f | Silhouette=%.4f", k, km.inertia_, sil)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ks, inertias, "o-"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow-Methode")
    ax2.plot(ks, silhouettes, "o-", color="green")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette-Analyse")
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "cluster_k_selection", subdir="clustering")
    best_k = ks[int(np.argmax(silhouettes))]
    log.info("Bester k nach Silhouette: %d", best_k)
    return best_k, dict(zip(ks, inertias)), dict(zip(ks, silhouettes))


def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """Fittet K-Means mit fester Clusterzahl."""
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    return km


def visualize_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray, save: bool = True,
                           max_plot_samples: int = 15_000):
    """
    Projiziert die Cluster mit PCA auf 2D.

    INTERPRETATION: gut getrennte Punktwolken sprechen für sinnvolle Cluster.
    GRENZE: nur 2 Komponenten; erklärte Varianz wird im Titel angegeben.
    WARUM max_plot_samples: PCA auf 307k Zeilen + Scatter mit 307k Punkten ist
    langsam und visuell überladen. 15k Punkte ergeben dieselbe Aussage schneller.
    Die PCA-Achsen werden auf dem VOLLEN Datensatz gefittet (korrekte Varianz),
    nur der Scatter-Plot zeigt einen repräsentativen Subsample.
    """
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)       # fit auf vollen Daten
    var = pca.explained_variance_ratio_

    # Subsampling nur für den Plot
    n = len(labels)
    if n > max_plot_samples:
        rng = np.random.default_rng(config.RANDOM_STATE)
        idx = rng.choice(n, size=max_plot_samples, replace=False)
        coords_plot, labels_plot = coords[idx], labels[idx]
        log.info("PCA-Plot: %d → %d Punkte (Subsample)", n, max_plot_samples)
    else:
        coords_plot, labels_plot = coords, labels

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(coords_plot[:, 0], coords_plot[:, 1],
                    c=labels_plot, cmap="tab10", s=8, alpha=0.5)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% Varianz)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% Varianz)")
    ax.set_title("Cluster in 2D (PCA-Projektion)")
    legend = ax.legend(*sc.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    if save:
        utils.save_figure(fig, "clusters_pca_2d", subdir="clustering")
    return fig, var.sum()


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, features: list[str],
                     target: str = config.TARGET, save: bool = True) -> pd.DataFrame:
    """
    Erstellt ein Cluster-Profil: Mittelwerte der Merkmale + Größe + Ausfallrate.

    WARUM Ausfallrate NACHTRÄGLICH: TARGET war nicht Teil des Clusterings. Der Vergleich
    der Ausfallrate je Cluster zeigt, ob die (rein merkmalsbasierte) Segmentierung auch
    risikorelevant ist – eine eigenständige, nicht-zirkuläre Erkenntnis.
    GRENZE: Cluster sind KEINE kausalen Gruppen; Unterschiede in der Ausfallrate sind
    deskriptiv, nicht ursächlich.
    """
    work = df.copy()
    work["cluster"] = labels
    agg = work.groupby("cluster")[features].mean()
    agg["cluster_size"] = work.groupby("cluster").size()
    if target in work.columns:
        agg["default_rate"] = work.groupby("cluster")[target].mean()
    agg = agg.reset_index()
    if save:
        utils.save_table(agg.round(3), "cluster_profiles", subdir="clustering")
    return agg


def run_clustering(df: pd.DataFrame, k: int | None = None):
    """
    Führt die komplette Clustering-Analyse aus und gibt Labels + Artefakte zurück.

    Ablauf: Feature-Auswahl -> Skalierung -> (optional k bestimmen) -> K-Means ->
    PCA-Visualisierung -> Cluster-Profile inkl. Ausfallrate.
    """
    X = select_cluster_features(df)
    feature_names = X.columns.tolist()
    pre = build_clustering_preprocessor()
    X_scaled = pre.fit_transform(X)

    if k is None:
        k, _, _ = find_optimal_k(X_scaled)
    km = fit_kmeans(X_scaled, k)
    labels = km.labels_

    _, explained = visualize_clusters_pca(X_scaled, labels)
    profiles = profile_clusters(df, labels, feature_names)
    log.info("Clustering fertig: k=%d, erklärte Varianz (2 PCs)=%.1f%%", k, explained * 100)
    return labels, profiles, km, pre


if __name__ == "__main__":
    feat = utils.load_processed("feature_matrix")
    labels, profiles, km, pre = run_clustering(feat)
    print("\n== Cluster-Profile (Auszug) ==")
    cols = ["cluster", "cluster_size", "default_rate", "AMT_INCOME_TOTAL",
            "credit_income_ratio", "external_score_mean", "age_years"]
    cols = [c for c in cols if c in profiles.columns]
    print(profiles[cols].round(3).to_string(index=False))
