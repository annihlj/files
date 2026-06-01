"""
feature_engineering.py
======================

Aggregation der relationalen Nebentabellen auf Antragsteller-Ebene (SK_ID_CURR)
und Erzeugung fachlich begründeter Features.

ZENTRALE METHODIK (Aggregation von 1:n-Tabellen):
    Die Nebentabellen (bureau, previous_application, installments_payments) enthalten
    MEHRERE Zeilen pro Antragsteller. Ein Modell auf SK_ID_CURR-Ebene braucht aber
    GENAU EINE Zeile pro Person. Deshalb verdichten wir jede Nebentabelle per
    Gruppierung (groupby SK_ID_CURR) zu Kennzahlen (count, sum, mean, max, ...).

    WARUM Aggregation statt naivem Join:
        Ein direkter Join von application (1 Zeile) mit bureau (n Zeilen) würde die
        Antragszeile n-mal vervielfachen ("row explosion") -> Duplikate, verzerrte
        Statistiken, kaputter Train-Test-Split. Aggregation verhindert das.

    PROBLEME, die bei Joins entstehen können (und wie wir sie adressieren):
        - Duplikate / row explosion  -> wir aggregieren VOR dem Join (1 Zeile je ID).
        - Datenverlust                -> Left-Join auf application: jede Antragszeile
                                         bleibt erhalten; fehlende Aggregate = "keine
                                         Historie" und werden später als 0/NaN behandelt.
        - Datenleckage (Leakage)      -> wir nutzen ausschließlich VERGANGENHEITSdaten
                                         (alle DAYS_*-Felder sind negativ = vor Antrag).
                                         Wir erzeugen KEINE Features, die Information aus
                                         der Zukunft oder aus TARGET selbst enthalten.

NAMENSKONVENTION: <tabelle>_<aggregat>_<spalte>, z. B. BUREAU_MEAN_AMT_CREDIT_SUM.
    -> nachvollziehbar, kollisionsfrei beim Zusammenführen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config, utils
except ImportError:  # pragma: no cover
    import config, utils  # type: ignore

log = utils.get_logger()


# ===========================================================================
# 1) FEATURES AUS DER HAUPTTABELLE (application)
# ===========================================================================
def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt fachlich motivierte Verhältnis- und Transformationsfeatures aus
    application.

    Jedes Feature mit fachlicher Begründung:
    - credit_income_ratio:   Kredit / Einkommen. Hohe Werte = hohe Verschuldung
      relativ zum Einkommen -> klassischer Risikoindikator (Schuldentragfähigkeit).
    - annuity_income_ratio:  Jahresrate / Einkommen. Anteil des Einkommens, der für
      die Rate gebunden ist -> Liquiditätsbelastung.
    - goods_credit_ratio:    Warenwert / Kredit. Werte <1 deuten auf Kredit über den
      reinen Warenwert hinaus (z. B. Gebühren) -> potenziell riskanter.
    - age_years / employment_years / registration_years: aus DAYS_* (negativ) in
      interpretierbare positive Jahre. Alter und Beschäftigungsdauer sind etablierte
      Bonitätsmerkmale (Achtung: Alter ist ethisch sensibel, siehe Ethik-Kapitel).
    - external_score_mean / _std: Aggregation der drei externen Scores. Diese sind im
      Home-Credit-Datensatz erfahrungsgemäß sehr prädiktiv; der Mittelwert glättet
      Fehlwerte, die Streuung misst Uneinigkeit der Quellen.
    - missing_values_count: Anzahl fehlender Felder pro Person. "Informatives Fehlen":
      unvollständige Anträge können mit Risiko korrelieren (MNAR).
    """
    df = df.copy()
    eps = 1e-5  # verhindert Division durch 0

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["credit_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["annuity_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["goods_credit_ratio"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + eps)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        # Annuität/Kredit ~ inverse Laufzeit; kürzere Laufzeit = höhere Ratenlast.
        df["annuity_credit_ratio"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)

    if "DAYS_BIRTH" in df.columns:
        df["age_years"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    if "DAYS_EMPLOYED" in df.columns:
        # DAYS_EMPLOYED wurde im Cleaning für Sentinel auf NaN gesetzt -> bleibt NaN.
        df["employment_years"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    if "DAYS_REGISTRATION" in df.columns:
        df["registration_years"] = (-df["DAYS_REGISTRATION"] / 365.25).round(1)

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["external_score_mean"] = df[ext_cols].mean(axis=1)
        df["external_score_std"] = df[ext_cols].std(axis=1)
        df["external_score_min"] = df[ext_cols].min(axis=1)
        df["external_score_max"] = df[ext_cols].max(axis=1)

    # Fehlwert-Zähler über die ursprünglichen Spalten (vor weiteren Joins).
    df["missing_values_count"] = df.isna().sum(axis=1)

    log.info("application-Features ergänzt -> %d Spalten.", df.shape[1])
    return df


# ===========================================================================
# 2) AGGREGATION: bureau (1:n) + bureau_balance (nested, optional)
# ===========================================================================
def aggregate_bureau(bureau: pd.DataFrame,
                     bureau_balance: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Verdichtet die externe Kredithistorie auf SK_ID_CURR-Ebene.

    WARUM relevant: bureau enthält Kredite bei ANDEREN Instituten. Anzahl, Status
    (aktiv/geschlossen), Höhe und insbesondere ÜBERFÄLLIGKEITEN sind starke
    Risikosignale (jemand mit vielen überfälligen Fremdkrediten ist riskanter).

    Erzeugte Features (Auswahl, je fachlich begründet):
    - BUREAU_COUNT:            Anzahl früherer Fremdkredite (Kreditaktivität).
    - BUREAU_ACTIVE_COUNT:     aktuell aktive Kredite (laufende Belastung).
    - BUREAU_CLOSED_COUNT:     abgeschlossene Kredite (Erfahrung/Historie).
    - BUREAU_OVERDUE_COUNT:    Kredite mit Überfälligkeit (Verzugsneigung).
    - BUREAU_MEAN/SUM_AMT_CREDIT_SUM: durchschnittliche/gesamte Kredithöhe.
    - BUREAU_MEAN_DAYS_CREDIT: durchschnittliches Alter der Kredite.
    - BUREAU_CREDIT_TYPE_NUNIQUE: Diversität der Kreditarten.
    """
    if bureau is None or bureau.empty:
        return pd.DataFrame(columns=[config.ID_COLUMN])

    df = bureau.copy()

    # Optional: bureau_balance zuerst auf SK_ID_BUREAU verdichten, dann anhängen.
    if bureau_balance is not None and not bureau_balance.empty:
        bb = bureau_balance.copy()
        # Status "X" = unbekannt, "C" = abgeschlossen; Ziffern = Monate Verzug.
        bb["status_is_overdue"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_OVERDUE_MONTHS=("status_is_overdue", "sum"),
        ).reset_index()
        df = df.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Hilfsindikatoren auf Kreditebene
    df["is_active"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)
    df["is_closed"] = (df["CREDIT_ACTIVE"] == "Closed").astype(int)
    df["has_overdue"] = (df.get("AMT_CREDIT_SUM_OVERDUE", 0) > 0).astype(int)

    agg_spec = {
        "SK_ID_BUREAU": ("SK_ID_BUREAU", "count"),
        "is_active": ("is_active", "sum"),
        "is_closed": ("is_closed", "sum"),
        "has_overdue": ("has_overdue", "sum"),
        "AMT_CREDIT_SUM_mean": ("AMT_CREDIT_SUM", "mean"),
        "AMT_CREDIT_SUM_sum": ("AMT_CREDIT_SUM", "sum"),
        "AMT_CREDIT_SUM_max": ("AMT_CREDIT_SUM", "max"),
        "DAYS_CREDIT_mean": ("DAYS_CREDIT", "mean"),
        "DAYS_CREDIT_min": ("DAYS_CREDIT", "min"),
        "CREDIT_TYPE_nunique": ("CREDIT_TYPE", "nunique"),
    }
    # Nur vorhandene Quellspalten verwenden (robust gegen fehlende optionale Spalten).
    agg_spec = {k: v for k, v in agg_spec.items() if v[0] in df.columns}

    grouped = df.groupby(config.ID_COLUMN).agg(**agg_spec)
    grouped.columns = [
        "BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_CLOSED_COUNT",
        "BUREAU_OVERDUE_COUNT", "BUREAU_MEAN_AMT_CREDIT_SUM",
        "BUREAU_SUM_AMT_CREDIT_SUM", "BUREAU_MAX_AMT_CREDIT_SUM",
        "BUREAU_MEAN_DAYS_CREDIT", "BUREAU_MIN_DAYS_CREDIT",
        "BUREAU_CREDIT_TYPE_NUNIQUE",
    ][:len(grouped.columns)]

    # Optional aus bureau_balance angehängte Aggregate mitnehmen.
    if "BB_OVERDUE_MONTHS" in df.columns:
        bb_extra = df.groupby(config.ID_COLUMN).agg(
            BUREAU_BB_OVERDUE_MONTHS_SUM=("BB_OVERDUE_MONTHS", "sum"),
            BUREAU_BB_MONTHS_COUNT_SUM=("BB_MONTHS_COUNT", "sum"),
        )
        grouped = grouped.join(bb_extra)

    log.info("bureau aggregiert -> %d Antragsteller, %d Features.",
             grouped.shape[0], grouped.shape[1])
    return grouped.reset_index()


# ===========================================================================
# 3) AGGREGATION: previous_application (1:n)
# ===========================================================================
def aggregate_previous(previous: pd.DataFrame) -> pd.DataFrame:
    """
    Verdichtet frühere Anträge bei Home Credit auf SK_ID_CURR-Ebene.

    WARUM relevant: Das bisherige Antragsverhalten (viele Anträge? oft abgelehnt?)
    ist informativ. Eine niedrige Bewilligungsquote kann auf bekannte Bonitätsprobleme
    hindeuten.

    Erzeugte Features:
    - PREV_COUNT:            Anzahl früherer Anträge.
    - PREV_APPROVED_COUNT / PREV_REFUSED_COUNT.
    - PREV_APPROVAL_RATE:    bewilligte / alle Anträge.
    - PREV_MEAN_AMT_CREDIT / PREV_MEAN_AMT_ANNUITY.
    - PREV_CONTRACT_TYPE_NUNIQUE.
    """
    if previous is None or previous.empty:
        return pd.DataFrame(columns=[config.ID_COLUMN])

    df = previous.copy()
    df["is_approved"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    df["is_refused"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    agg_spec = {
        "SK_ID_PREV": ("SK_ID_PREV", "count"),
        "is_approved": ("is_approved", "sum"),
        "is_refused": ("is_refused", "sum"),
        "AMT_CREDIT_mean": ("AMT_CREDIT", "mean"),
        "AMT_ANNUITY_mean": ("AMT_ANNUITY", "mean"),
        "CONTRACT_TYPE_nunique": ("NAME_CONTRACT_TYPE", "nunique"),
    }
    agg_spec = {k: v for k, v in agg_spec.items() if v[0] in df.columns}
    grouped = df.groupby(config.ID_COLUMN).agg(**agg_spec)
    grouped.columns = [
        "PREV_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT",
        "PREV_MEAN_AMT_CREDIT", "PREV_MEAN_AMT_ANNUITY", "PREV_CONTRACT_TYPE_NUNIQUE",
    ][:len(grouped.columns)]

    # Bewilligungsquote als abgeleitetes Verhältnis.
    if {"PREV_APPROVED_COUNT", "PREV_COUNT"}.issubset(grouped.columns):
        grouped["PREV_APPROVAL_RATE"] = (
            grouped["PREV_APPROVED_COUNT"] / grouped["PREV_COUNT"].clip(lower=1)
        )

    log.info("previous_application aggregiert -> %d Antragsteller, %d Features.",
             grouped.shape[0], grouped.shape[1])
    return grouped.reset_index()


# ===========================================================================
# 4) AGGREGATION: installments_payments (nested)
# ===========================================================================
def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Verdichtet das Ratenzahlungsverhalten auf SK_ID_CURR-Ebene.

    WARUM relevant: Tatsächliches Zahlungsverhalten ist eines der direktesten Signale
    für künftiges Ausfallrisiko. Wir messen Verspätung und Unterzahlung.

    Berechnete Größen (auf Ratenebene, dann aggregiert):
    - payment_delay = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
        > 0 bedeutet: tatsächliche Zahlung NACH Fälligkeit (Verzug).
    - payment_difference = AMT_INSTALMENT - AMT_PAYMENT
        > 0 bedeutet: weniger gezahlt als fällig (Unterzahlung).

    Features:
    - INST_PAYMENT_DELAY_MEAN / _MAX
    - INST_LATE_PAYMENT_COUNT / INST_LATE_PAYMENT_RATIO
    - INST_PAYMENT_DIFF_MEAN / _SUM
    - INST_COUNT (Anzahl erfasster Raten)
    """
    if installments is None or installments.empty:
        return pd.DataFrame(columns=[config.ID_COLUMN])

    df = installments.copy()
    df["payment_delay"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["is_late"] = (df["payment_delay"] > 0).astype(int)
    df["payment_difference"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    grouped = df.groupby(config.ID_COLUMN).agg(
        INST_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INST_PAYMENT_DELAY_MEAN=("payment_delay", "mean"),
        INST_PAYMENT_DELAY_MAX=("payment_delay", "max"),
        INST_LATE_PAYMENT_COUNT=("is_late", "sum"),
        INST_PAYMENT_DIFF_MEAN=("payment_difference", "mean"),
        INST_PAYMENT_DIFF_SUM=("payment_difference", "sum"),
    )
    grouped["INST_LATE_PAYMENT_RATIO"] = (
        grouped["INST_LATE_PAYMENT_COUNT"] / grouped["INST_COUNT"].clip(lower=1)
    )
    log.info("installments_payments aggregiert -> %d Antragsteller, %d Features.",
             grouped.shape[0], grouped.shape[1])
    return grouped.reset_index()


# ===========================================================================
# 5) ZUSAMMENFÜHREN ZUM FINALEN MODELLIERUNGSDATENSATZ
# ===========================================================================
def build_feature_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Baut den finalen Datensatz auf SK_ID_CURR-Ebene.

    Vorgehen:
    1) Haupttabelle + application-Features.
    2) Jede Nebentabelle aggregieren.
    3) Per LEFT JOIN auf SK_ID_CURR anhängen (keine Antragszeile geht verloren).
    4) Zeilenanzahl-Invariante prüfen: nach allen Joins muss die Zeilenzahl
       UNVERÄNDERT der Anzahl der Antragsteller entsprechen (Kontrolle gegen
       row explosion / Duplikate).

    WARUM Left-Join: Personen ohne Bureau-/Vorhistorie sollen erhalten bleiben;
    fehlende Aggregate bedeuten "keine Historie" und werden später behandelt.
    """
    app = tables["application_train"].copy()
    app = add_application_features(app)
    n_applicants = app[config.ID_COLUMN].nunique()
    log.info("Start build_feature_matrix mit %d Antragstellern.", n_applicants)

    if "bureau" in tables:
        bureau_agg = aggregate_bureau(tables["bureau"], tables.get("bureau_balance"))
        app = app.merge(bureau_agg, on=config.ID_COLUMN, how="left")

    if "previous_application" in tables:
        prev_agg = aggregate_previous(tables["previous_application"])
        app = app.merge(prev_agg, on=config.ID_COLUMN, how="left")

    if "installments_payments" in tables:
        inst_agg = aggregate_installments(tables["installments_payments"])
        app = app.merge(inst_agg, on=config.ID_COLUMN, how="left")

    # Invariante: keine Zeilenvervielfachung.
    assert len(app) == n_applicants, (
        f"Row explosion! {len(app)} Zeilen statt {n_applicants}. "
        "Aggregation/Join fehlerhaft."
    )

    # Count-Features, die für Personen ohne Historie NaN sind, sinnvoll mit 0 füllen.
    # WARUM hier (und nicht in der Pipeline): Dies ist eine deterministische, fachlich
    # eindeutige Setzung ("keine früheren Kredite" = 0), keine geschätzte Statistik
    # -> leakage-frei. Verteilungsabhängige Imputation (Median etc.) bleibt der Pipeline.
    count_cols = [c for c in app.columns if c.endswith("_COUNT")]
    app[count_cols] = app[count_cols].fillna(0)

    log.info("Finale Feature-Matrix: %d Zeilen x %d Spalten.", *app.shape)
    return app


def potential_leakage_and_ethics_review(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Markiert kritische Spalten – KEINE automatische Löschung, sondern Dokumentation
    zur bewussten Entscheidung im Modellierungs-/Ethikteil.

    KATEGORIEN:
    - sensible/Proxy-Merkmale: Geschlecht, Alter etc. (rechtlich/ethisch heikel; in
      DE/EU sind direkte Merkmale wie Geschlecht in der Kreditvergabe i. d. R. unzulässig,
      und Proxys können indirekte Diskriminierung erzeugen).
    - mögliche Leakage-Kandidaten: hier bewusst LEER, weil wir ausschließlich
      Vergangenheitsdaten und keine TARGET-abhängigen Features gebildet haben. Die
      Funktion bleibt als Prüfhaken bestehen, falls später Features ergänzt werden.
    """
    sensitive = [c for c in config.SENSITIVE_OR_PROXY_FEATURES if c in df.columns]
    # Heuristik: Spalten, die TARGET im Namen tragen (außer der Zielspalte selbst).
    leakage = [c for c in df.columns if "TARGET" in c.upper() and c != config.TARGET]
    review = {"sensitive_or_proxy": sensitive, "leakage_candidates": leakage}
    log.info("Ethik/Leakage-Review: %d sensible, %d Leakage-Kandidaten.",
             len(sensitive), len(leakage))
    return review


if __name__ == "__main__":
    from src import load_data, data_cleaning
    tabs = load_data.load_all_available()
    tabs["application_train"] = data_cleaning.clean_application(tabs["application_train"])
    fm = build_feature_matrix(tabs)
    print("Feature-Matrix:", fm.shape)
    print("Neue Features (Auszug):",
          [c for c in fm.columns if c.startswith(("BUREAU_", "PREV_", "INST_"))][:12])
    print("Review:", potential_leakage_and_ethics_review(fm))
