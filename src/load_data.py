"""
load_data.py
============

Laden der Rohdaten in pandas-DataFrames.

WAS:    Stellt Funktionen bereit, um einzelne Tabellen oder alle verfügbaren Tabellen
        zu laden. Optional Speicheroptimierung durch Downcasting der Datentypen.
WARUM:  Eine zentrale Ladeschicht entkoppelt "wo/wie liegen die Daten" von "was machen
        wir damit". Notebooks und Skripte rufen einheitlich dieselben Funktionen auf.
ALTERNATIVEN:
        - Polars oder DuckDB statt pandas (deutlich speicher-/zeiteffizienter bei den
          großen Nebentabellen; hier bewusst pandas wegen Verbreitung im Lehrkontext).
        - Parquet statt CSV als Zwischenformat (schneller, typsicher) — wird in
          späteren Schritten für interim/processed genutzt.
RISIKEN/LIMITATIONEN:
        - Einige Tabellen (installments_payments ~13 Mio. Zeilen) sind groß; auf
          schwacher Hardware kann das RAM-kritisch werden. Daher Downcasting-Option.
"""
from __future__ import annotations

import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduziert den Speicherverbrauch durch Downcasting numerischer Spalten.

    WARUM: pandas liest Ganzzahlen standardmäßig als int64 und Gleitkomma als float64.
    Viele Spalten passen problemlos in int32/float32, was den RAM-Bedarf grob halbiert.
    LIMITATION: float32 hat geringere Präzision; für reine Modellfeatures unkritisch,
    aber man sollte es nicht blind auf z. B. Geldbeträge mit Centgenauigkeit anwenden,
    wenn exakte Arithmetik nötig ist. Hier akzeptabel, da Features ohnehin skaliert werden.
    """
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_table(name: str, optimize_memory: bool = True, nrows: int | None = None) -> pd.DataFrame:
    """
    Lädt eine einzelne Tabelle anhand ihres logischen Namens (Schlüssel in config.TABLES).

    Parameter
    ---------
    name : str
        Logischer Tabellenname, z. B. "application_train".
    optimize_memory : bool
        Wenn True, werden Datentypen per Downcasting reduziert.
    nrows : int | None
        Optionales Zeilenlimit – nützlich zum schnellen Prototyping/Debuggen,
        damit man nicht jedes Mal Millionen Zeilen laden muss.

    Rückgabe
    --------
    pd.DataFrame
    """
    if name not in config.TABLES:
        raise KeyError(f"Unbekannte Tabelle '{name}'. Bekannt: {list(config.TABLES)}")

    fpath = config.RAW_DIR / config.TABLES[name]["filename"]
    if not fpath.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {fpath}\n"
            f"Bitte zuerst 'python -m src.data_download' ausführen."
        )

    df = pd.read_csv(fpath, nrows=nrows)
    if optimize_memory:
        df = _downcast(df)
    return df


def load_all_available(required_only: bool = False, **kwargs) -> dict[str, pd.DataFrame]:
    """
    Lädt alle vorhandenen Tabellen in ein Dictionary {name: DataFrame}.

    WARUM "available": Optionale Tabellen fehlen evtl. Statt hart zu scheitern,
    überspringen wir nicht vorhandene Dateien und protokollieren das – so läuft die
    Pipeline auch im priorisierten (Pflicht-)Modus durch.
    """
    names = config.REQUIRED_TABLES if required_only else list(config.TABLES)
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        if name == "HomeCredit_columns_description":
            continue  # reine Metadaten, kein Modellierungsdatensatz
        fpath = config.RAW_DIR / config.TABLES[name]["filename"]
        if not fpath.exists():
            print(f"[load_data] uebersprungen (nicht vorhanden): {name}")
            continue
        print(f"[load_data] lade {name} ...")
        tables[name] = load_table(name, **kwargs)
        print(f"           shape={tables[name].shape}")
    return tables


def memory_report(df: pd.DataFrame) -> str:
    """Gibt eine kompakte Speichernutzung-Zusammenfassung als String zurück."""
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    return f"{df.shape[0]:,} Zeilen x {df.shape[1]} Spalten | {mb:,.1f} MB"


if __name__ == "__main__":
    # Selbsttest: nur ausführbar, wenn Daten vorhanden sind.
    try:
        app = load_table("application_train", nrows=1000)
        print("application_train (1000 Zeilen):", memory_report(app))
        print("Spalten (Auszug):", list(app.columns[:8]))
    except FileNotFoundError as exc:
        print(exc)
