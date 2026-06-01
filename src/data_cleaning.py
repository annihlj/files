"""
data_cleaning.py
================

Datenbereinigung des Home-Credit-Datensatzes.

LEITPRINZIP (sehr wichtig für Wissenschaftlichkeit & Leakage-Vermeidung):
    Wir unterscheiden strikt zwischen zwei Arten von "Cleaning":

    (A) DETERMINISTISCHE Korrekturen, die KEINE Statistik aus den Daten lernen.
        Beispiele: kodierte Fehlwerte (Sentinels) durch NaN ersetzen, exakte
        Duplikate entfernen, offensichtlich unmögliche Werte korrigieren,
        Datentypen vereinheitlichen. Diese sind leakage-frei und dürfen auf dem
        GESAMTEN Datensatz vor dem Train-Test-Split erfolgen, weil sie nicht von
        der Verteilung abhängen.

    (B) DATENGETRIEBENE Schritte, die Parameter SCHÄTZEN (z. B. Median für Imputation,
        Quantilgrenzen für Ausreißer-Clipping, Skalierungsstatistiken). Diese gehören
        in die scikit-learn-Pipeline und werden NUR auf den Trainingsdaten gefittet
        (siehe preprocessing.py). Andernfalls "sieht" das Modell indirekt die Testdaten
        -> Data Leakage -> zu optimistische, nicht generalisierende Ergebnisse.

    Dieses Modul enthält daher bewusst NUR Typ-(A)-Operationen plus Diagnosefunktionen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()

# Bekannter Sentinel im Home-Credit-Datensatz: DAYS_EMPLOYED == 365243 kodiert
# faktisch "nicht erwerbstätig" (Rentner/arbeitslos), nicht eine echte Beschäftigungsdauer.
DAYS_EMPLOYED_SENTINEL = 365243


# ===========================================================================
# DIAGNOSE (verändert nichts, dient der Dokumentation)
# ===========================================================================
def diagnose(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """
    Erstellt einen Datenqualitätsbericht je Spalte.

    WAS:    Pro Spalte Datentyp, Anteil fehlender Werte, Anzahl eindeutiger Werte,
            Anteil des häufigsten Werts (Hinweis auf Konstanten/Quasi-Konstanten).
    WARUM:  Ein systematischer Qualitätsbericht ersetzt punktuelles "Anschauen" und
            ist im Portfolio reproduzierbar dokumentierbar.
    RÜCKGABE: DataFrame mit einer Zeile pro Spalte.
    """
    n = len(df)
    records = []
    for col in df.columns:
        s = df[col]
        vc = s.value_counts(dropna=True)
        top_share = (vc.iloc[0] / n) if len(vc) else np.nan
        records.append({
            "Spalte": col,
            "Dtype": str(s.dtype),
            "Fehlend_%": round(s.isna().mean() * 100, 2),
            "Eindeutige_Werte": int(s.nunique(dropna=True)),
            "Top_Wert_Anteil_%": round(top_share * 100, 2) if not np.isnan(top_share) else np.nan,
        })
    report = pd.DataFrame(records).sort_values("Fehlend_%", ascending=False)
    log.info("Diagnose '%s': %d Zeilen x %d Spalten", name, n, df.shape[1])
    return report.reset_index(drop=True)


def find_constant_columns(df: pd.DataFrame, threshold: float = 0.999) -> list[str]:
    """
    Findet (quasi-)konstante Spalten.

    WARUM: Konstante Merkmale tragen keine Information, blähen aber die Feature-Matrix
    auf und können numerische Probleme verursachen (z. B. Division durch Null bei
    Skalierung). threshold=0.999 fängt auch Quasi-Konstanten ab.
    """
    constants = []
    for col in df.columns:
        top_share = df[col].value_counts(normalize=True, dropna=False)
        if len(top_share) and top_share.iloc[0] >= threshold:
            constants.append(col)
    return constants


# ===========================================================================
# DETERMINISTISCHE KORREKTUREN (Typ A – leakage-frei)
# ===========================================================================
def remove_exact_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """
    Entfernt exakte Duplikatzeilen.

    WARUM: Duplikate verzerren Statistiken und können bei Train-Test-Split dazu führen,
    dass identische Zeilen in beiden Mengen landen (subtiles Leakage). Wir prüfen
    zusätzlich auf doppelte IDs, da SK_ID_CURR eindeutig sein MUSS.
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    removed = before - len(df)
    if removed:
        log.warning("%d exakte Duplikatzeilen entfernt.", removed)
    return df


def assert_unique_id(df: pd.DataFrame, id_col: str = config.ID_COLUMN) -> None:
    """
    Stellt sicher, dass die ID eindeutig ist (Integritätsprüfung der Zielebene).

    WARUM: Wenn SK_ID_CURR nicht eindeutig ist, sind alle nachfolgenden Joins/
    Aggregationen fehlerhaft. Lieber früh und laut scheitern.
    """
    if id_col in df.columns:
        dup = df[id_col].duplicated().sum()
        if dup:
            raise ValueError(f"ID '{id_col}' nicht eindeutig: {dup} Duplikate.")
        log.info("ID '%s' ist eindeutig (%d Zeilen).", id_col, len(df))


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ersetzt den Sentinel DAYS_EMPLOYED == 365243 durch NaN und legt eine
    Indikatorvariable an.

    WAS:    365243 Tage ≈ 1000 Jahre – ein klar unmöglicher Wert, der "nicht
            erwerbstätig" kodiert.
    WARUM:  Würde man den Wert belassen, verzerrte er Mittelwerte/Skalierung massiv.
            Das bloße Löschen würde aber Information vernichten – die Tatsache "nicht
            erwerbstätig" ist pot.  prädiktiv. Daher: Wert -> NaN (später imputiert),
            zusätzlich Flag DAYS_EMPLOYED_ANOMALY behalten. Das ist gängige Praxis im
            Umgang mit "informativem Fehlen" (missing-not-at-random).
    ALTERNATIVE: Zeilen verwerfen (Datenverlust) oder Wert auf 0 setzen (verfälscht).
    """
    if "DAYS_EMPLOYED" not in df.columns:
        return df
    df = df.copy()
    mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    df["DAYS_EMPLOYED_ANOMALY"] = mask.astype(int)
    df.loc[mask, "DAYS_EMPLOYED"] = np.nan
    if mask.sum():
        log.info("DAYS_EMPLOYED-Sentinel in %d Zeilen -> NaN + Indikator.", int(mask.sum()))
    return df


def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Korrigiert offensichtlich unmögliche/inkonsistente Werte (deterministisch).

    WARUM: Manche Datenfehler sind ohne Statistik als unmöglich erkennbar
    (z. B. negatives Einkommen, Kreditbetrag <= 0). Solche Werte werden zu NaN,
    damit die Imputation sie sauber behandelt, statt das Modell zu verfälschen.
    LIMITATION: Wir greifen NUR bei logisch unmöglichen Werten ein, nicht bei
    "nur ungewöhnlichen" – Letzteres ist eine datengetriebene Entscheidung (Pipeline).
    """
    df = df.copy()
    # Geldbeträge müssen positiv sein.
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df.columns:
            bad = df[col] <= 0
            if bad.sum():
                log.info("%s: %d nicht-positive Werte -> NaN.", col, int(bad.sum()))
                df.loc[bad, col] = np.nan
    # CODE_GENDER hat im Originaldatensatz seltene "XNA"-Einträge.
    if "CODE_GENDER" in df.columns:
        xna = df["CODE_GENDER"] == "XNA"
        if xna.sum():
            log.info("CODE_GENDER 'XNA' in %d Zeilen -> NaN.", int(xna.sum()))
            df.loc[xna, "CODE_GENDER"] = np.nan
    return df


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wendet alle deterministischen Korrekturen auf die Haupttabelle an.

    Reihenfolge ist bewusst gewählt:
    1) ID-Eindeutigkeit prüfen, 2) exakte Duplikate, 3) Sentinel-Fix,
    4) unmögliche Werte. Alle Schritte sind leakage-frei.
    """
    assert_unique_id(df)
    df = remove_exact_duplicates(df, subset=[config.ID_COLUMN] if config.ID_COLUMN in df else None)
    df = fix_days_employed(df)
    df = fix_invalid_values(df)
    return df


def class_balance_summary(df: pd.DataFrame, target: str = config.TARGET) -> dict:
    """
    Fasst die Klassenbalance zusammen (für Dokumentation/EDA).

    WARUM kritisch (Kreditrisiko): Eine stark unausgewogene Zielvariable bedeutet,
    dass ein Modell durch bloßes Vorhersagen der Mehrheitsklasse ("kein Ausfall")
    eine hohe Accuracy erzielt, OHNE Ausfälle zu erkennen. Genau die Minderheitsklasse
    (Ausfall) ist aber geschäftlich/ethisch die wichtige. Daraus folgt: geeignete
    Metriken (PR-AUC, Recall) und ggf. Umgang mit Imbalance (class_weight) statt
    Accuracy als Leitgröße. (He & Garcia, 2009)
    """
    if target not in df.columns:
        return {}
    counts = df[target].value_counts().sort_index()
    rate = float(df[target].mean())
    summary = {
        "n_total": int(len(df)),
        "n_negativ_0": int(counts.get(0, 0)),
        "n_positiv_1": int(counts.get(1, 0)),
        "positivrate": rate,
        "imbalance_ratio_0_zu_1": float(counts.get(0, 0) / max(counts.get(1, 1), 1)),
    }
    return summary


if __name__ == "__main__":
    from src import load_data
    app = load_data.load_table("application_train")
    rep = diagnose(app, "application_train")
    print(rep.head(10).to_string(index=False))
    cleaned = clean_application(app)
    print("\nKlassenbalance:", class_balance_summary(cleaned))
    print("Konstante Spalten:", find_constant_columns(cleaned))
