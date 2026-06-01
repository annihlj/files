"""
utils.py
========

Gemeinsame Hilfsfunktionen für das gesamte Projekt.

WAS:    Logging-Setup, deterministisches Seeding, einheitliches Speichern von
        Abbildungen/Tabellen/Modellen mit Zeitstempel.
WARUM:  DRY-Prinzip (Don't Repeat Yourself). Wenn jedes Notebook seine eigene
        Speicherlogik hätte, wären Pfade und Formate inkonsistent. Einheitliche
        Helfer sichern reproduzierbare, nachvollziehbare Artefakte.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "credit_risk") -> logging.Logger:
    """
    Liefert einen konfigurierten Logger.

    WARUM Logging statt print(): Logs haben Zeitstempel und Level (INFO/WARNING/
    ERROR), lassen sich filtern und in Dateien umleiten. In einem wissenschaftlichen
    Projekt ist nachvollziehbar, WANN was passierte – wichtig für Reproduzierbarkeit.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # doppelte Handler bei erneutem Import vermeiden
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Reproduzierbarkeit
# ---------------------------------------------------------------------------
def set_seed(seed: int = config.RANDOM_STATE) -> None:
    """
    Setzt globale Zufalls-Seeds (Python, NumPy).

    WARUM: Einheitliches Seeding an einer Stelle. scikit-learn-Schätzer erhalten
    random_state zusätzlich explizit als Parameter (lokale Reproduzierbarkeit),
    aber globale Operationen (z. B. Sampling) profitieren von gesetzten Seeds.
    LIMITATION: Garantiert Determinismus nur bei single-threaded Verfahren und
    identischen Versionen; manche parallelisierten Routinen bleiben minimal
    nicht-deterministisch.
    """
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Speicher-Helfer
# ---------------------------------------------------------------------------
def save_figure(fig: Any, name: str, subdir: str | None = None, dpi: int = 150) -> Path:
    """
    Speichert eine Matplotlib-Figur reproduzierbar nach reports/figures/.

    Parameter
    ---------
    fig : matplotlib.figure.Figure
    name : str   Dateiname ohne Endung.
    subdir : str | None   Optionales Unterverzeichnis (z. B. "eda").
    dpi : int    Auflösung. 150 ist ein guter Kompromiss aus Schärfe und Dateigröße.

    WARUM bbox_inches='tight': verhindert abgeschnittene Achsenbeschriftungen.
    """
    target_dir = config.FIGURES_DIR / subdir if subdir else config.FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    get_logger().info("Abbildung gespeichert: %s", path)
    return path


def save_table(df, name: str, subdir: str | None = None, index: bool = False) -> Path:
    """Speichert ein DataFrame als CSV nach reports/tables/."""
    target_dir = config.TABLES_DIR / subdir if subdir else config.TABLES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.csv"
    df.to_csv(path, index=index)
    get_logger().info("Tabelle gespeichert: %s", path)
    return path


def timestamp() -> str:
    """Liefert einen kompakten Zeitstempel YYYYmmdd_HHMMSS für Artefaktnamen."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_parquet(df, name: str, stage: str = "interim") -> Path:
    """
    Speichert ein DataFrame als Parquet in interim/ oder processed/.

    WARUM Parquet: spaltenorientiert, komprimiert, typsicher (behält dtypes),
    deutlich schneller zu lesen als CSV. Ideal für Zwischenartefakte der Pipeline.
    FALLBACK: Ist kein Parquet-Engine (pyarrow/fastparquet) installiert, wird
    automatisch CSV genutzt, damit die Pipeline trotzdem läuft. In der über
    requirements.txt installierten Umgebung ist pyarrow vorhanden -> echtes Parquet.
    """
    stage_dir = {"interim": config.INTERIM_DIR, "processed": config.PROCESSED_DIR}[stage]
    path = stage_dir / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
        get_logger().info("Parquet gespeichert: %s (%s)", path, df.shape)
    except (ImportError, ValueError) as exc:
        path = stage_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        get_logger().warning("Kein Parquet-Engine (%s) -> CSV-Fallback: %s (%s)",
                             type(exc).__name__, path, df.shape)
    return path


def load_processed(name: str, stage: str = "processed"):
    """
    Lädt ein zuvor gespeichertes Artefakt (Parquet bevorzugt, CSV-Fallback).

    WARUM: Spiegelt save_parquet – nachfolgende Notebooks laden das fertige Artefakt
    unabhängig vom Format.
    """
    import pandas as pd
    stage_dir = {"interim": config.INTERIM_DIR, "processed": config.PROCESSED_DIR}[stage]
    parquet_path = stage_dir / f"{name}.parquet"
    csv_path = stage_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Weder {parquet_path} noch {csv_path} gefunden.")
