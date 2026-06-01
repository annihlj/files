"""
config.py
=========

Zentrale Konfiguration für das Projekt "Kreditwürdigkeitsprüfung mit Machine Learning".

WAS:    Bündelt alle Pfade, Konstanten, Spaltennamen und Tabellen-Metadaten an einer Stelle.
WARUM:  Reproduzierbarkeit und Wartbarkeit. Wenn Pfade oder Parameter (z. B. random_state)
        an genau einer Stelle definiert sind, lassen sich Experimente nachvollziehbar
        wiederholen und es entstehen keine widersprüchlichen "magic numbers" über mehrere
        Notebooks/Skripte verteilt.
ALTERNATIVEN: YAML/TOML-Konfigurationsdateien (z. B. mit Hydra oder OmegaConf) oder
        Umgebungsvariablen (.env). Für ein Portfolio-Projekt dieser Größe ist eine
        Python-Datei am transparentesten, weil sie ohne Zusatzabhängigkeiten auskommt
        und IDE-Autovervollständigung erlaubt.
RISIKEN/LIMITATIONEN: Hartcodierte absolute Pfade wären nicht portabel – deshalb werden
        hier ausschließlich relative Pfade vom Projektwurzelverzeichnis abgeleitet.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
# WARUM Path(__file__): Das Projektwurzelverzeichnis wird relativ zur Lage dieser
# Datei bestimmt (src/config.py -> Projektwurzel ist eine Ebene höher). So funktioniert
# der Code unabhängig vom aktuellen Arbeitsverzeichnis (Notebook, CLI, Streamlit).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"          # unveränderte Originaldaten (read-only Mentalität)
INTERIM_DIR: Path = DATA_DIR / "interim"  # Zwischenergebnisse (z. B. aggregierte Nebentabellen)
PROCESSED_DIR: Path = DATA_DIR / "processed"  # finaler Modellierungsdatensatz

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"

# Verzeichnisse bei Import sicherstellen (idempotent, schadet nicht bei Mehrfachaufruf).
for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproduzierbarkeit
# ---------------------------------------------------------------------------
# WARUM: Ein fixer Seed macht Train-Test-Splits, Modellinitialisierung und
# stochastische Verfahren (Random Forest, K-Means) deterministisch. Ohne fixen
# Seed wären Ergebnisse zwischen Läufen nicht vergleichbar -> nicht reproduzierbar.
# LIMITATION: Reproduzierbarkeit gilt nur bei identischen Bibliotheksversionen und
# (bei manchen GPU-/Multithread-Verfahren) identischer Hardware. Deshalb requirements.txt
# mit gepinnten Versionen.
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Zielvariable
# ---------------------------------------------------------------------------
TARGET: str = "TARGET"
# Bedeutung laut Home-Credit-Dokumentation:
#   1 = Klient mit Zahlungsschwierigkeiten (Verzug über definierte Schwelle) -> "Ausfall"
#   0 = alle übrigen Fälle -> "kein dokumentiertes Ausfallereignis"
TARGET_POSITIVE_LABEL: int = 1

# Haupt-Identifikator (Antragsteller-/Antragsebene).
ID_COLUMN: str = "SK_ID_CURR"

# ---------------------------------------------------------------------------
# Train-Test-Split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
# WARUM stratify: Bei stark unausgewogener Zielvariable (~8 % Positive) stellt
# Stratifizierung sicher, dass Trainings- und Testset denselben Positivanteil haben.
# Ohne Stratifizierung könnte der Testanteil der Minderheitsklasse zufällig
# über-/unterrepräsentiert sein und die Evaluation verzerren.

# ---------------------------------------------------------------------------
# Tabellen-Metadaten des Home-Credit-Datensatzes
# ---------------------------------------------------------------------------
# Diese Struktur dokumentiert maschinenlesbar, welche Tabelle welche Granularität
# hat und über welche Schlüssel sie mit der Haupttabelle verbunden ist.
# WARUM als Datenstruktur: ermöglicht generische Lade-/Aggregationslogik und
# dient gleichzeitig als Dokumentation (Single Source of Truth).
#
# Granularität (entscheidend für Aggregation!):
#   - "current_application": eine Zeile pro Antrag = Zielebene (SK_ID_CURR eindeutig)
#   - "one_to_many":         mehrere Zeilen pro SK_ID_CURR -> muss aggregiert werden
#   - "nested":              hängt über SK_ID_BUREAU bzw. SK_ID_PREV an einer
#                            Zwischentabelle, nicht direkt an SK_ID_CURR

TABLES: dict[str, dict] = {
    "application_train": {
        "filename": "application_train.csv",
        "granularity": "current_application",
        "primary_key": ["SK_ID_CURR"],
        "join_key_to_main": "SK_ID_CURR",
        "priority": "required",
        "description": (
            "Haupttabelle: ein Antrag pro Zeile. Enthält Zielvariable TARGET, "
            "demografische Daten, Einkommen, Kreditbetrag, externe Scores."
        ),
    },
    "application_test": {
        "filename": "application_test.csv",
        "granularity": "current_application",
        "primary_key": ["SK_ID_CURR"],
        "join_key_to_main": "SK_ID_CURR",
        "priority": "optional",
        "description": (
            "Wie application_train, aber OHNE TARGET. Nur für spätere Inferenz/"
            "Submission. Wird beim Training NICHT verwendet (kein Leakage)."
        ),
    },
    "bureau": {
        "filename": "bureau.csv",
        "granularity": "one_to_many",
        "primary_key": ["SK_ID_BUREAU"],
        "join_key_to_main": "SK_ID_CURR",
        "priority": "required",
        "description": (
            "Frühere Kredite des Antragstellers bei ANDEREN Instituten, gemeldet an "
            "die Kreditauskunftei. Mehrere Zeilen je SK_ID_CURR -> Aggregation nötig."
        ),
    },
    "bureau_balance": {
        "filename": "bureau_balance.csv",
        "granularity": "nested",
        "primary_key": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
        "join_key_to_main": "SK_ID_BUREAU",  # indirekt über bureau!
        "priority": "optional",
        "description": (
            "Monatliche Salden/Status je Bureau-Kredit. Hängt an SK_ID_BUREAU. "
            "Muss zuerst auf SK_ID_BUREAU, dann über bureau auf SK_ID_CURR aggregiert werden."
        ),
    },
    "previous_application": {
        "filename": "previous_application.csv",
        "granularity": "one_to_many",
        "primary_key": ["SK_ID_PREV"],
        "join_key_to_main": "SK_ID_CURR",
        "priority": "required",
        "description": (
            "Frühere Kreditanträge bei Home Credit selbst. Mehrere Zeilen je "
            "SK_ID_CURR -> Aggregation nötig (z. B. Bewilligungsquote)."
        ),
    },
    "installments_payments": {
        "filename": "installments_payments.csv",
        "granularity": "nested",
        "primary_key": ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"],
        "join_key_to_main": "SK_ID_PREV",  # indirekt über previous_application
        "priority": "required",
        "description": (
            "Tatsächliche vs. geplante Ratenzahlungen früherer Home-Credit-Kredite. "
            "Sehr informativ für Zahlungsverhalten (Verzug, Unterzahlung)."
        ),
    },
    "POS_CASH_balance": {
        "filename": "POS_CASH_balance.csv",
        "granularity": "nested",
        "primary_key": ["SK_ID_PREV", "MONTHS_BALANCE"],
        "join_key_to_main": "SK_ID_PREV",
        "priority": "optional",
        "description": (
            "Monatliche Salden von POS- (Point of Sale) und Cash-Krediten bei Home Credit."
        ),
    },
    "credit_card_balance": {
        "filename": "credit_card_balance.csv",
        "granularity": "nested",
        "primary_key": ["SK_ID_PREV", "MONTHS_BALANCE"],
        "join_key_to_main": "SK_ID_PREV",
        "priority": "optional",
        "description": (
            "Monatliche Kreditkartensalden früherer Home-Credit-Kreditkarten."
        ),
    },
    "HomeCredit_columns_description": {
        "filename": "HomeCredit_columns_description.csv",
        "granularity": "metadata",
        "primary_key": [],
        "join_key_to_main": None,
        "priority": "optional",
        "description": "Spaltenbeschreibungen (Data Dictionary). Keine Modellierungsdaten.",
    },
}

# Bequemer Zugriff auf Pflicht-/Optional-Listen.
REQUIRED_TABLES: list[str] = [k for k, v in TABLES.items() if v["priority"] == "required"]
OPTIONAL_TABLES: list[str] = [k for k, v in TABLES.items() if v["priority"] == "optional"]

# ---------------------------------------------------------------------------
# Features, die aus ethischen/rechtlichen Gründen kritisch zu behandeln sind
# ---------------------------------------------------------------------------
# WARUM: Diese Spalten sind entweder sensible Merkmale oder potenzielle Proxys dafür.
# Sie werden nicht automatisch gelöscht, aber zentral markiert, damit man im
# Modellierungs- und Diskussionsteil bewusst darüber entscheidet (siehe Ethik-Kapitel).
SENSITIVE_OR_PROXY_FEATURES: list[str] = [
    "CODE_GENDER",            # Geschlecht: in vielen Jurisdiktionen unzulässiges Merkmal
    "DAYS_BIRTH",             # Alter: in der EU/DE ein geschütztes Merkmal (AGG/AGG-Recht)
    "NAME_FAMILY_STATUS",     # Familienstand: kann mit Geschlecht/Alter korrelieren
    "CNT_CHILDREN",           # Kinderzahl: Proxy für Familienstand/Geschlecht
    "NAME_HOUSING_TYPE",      # Wohnsituation: Proxy für sozioökonomischen Status
    "REGION_RATING_CLIENT",   # Regions-Rating: möglicher geografischer Proxy (Redlining-Risiko)
    "REGION_RATING_CLIENT_W_CITY",
]

if __name__ == "__main__":
    # Kleiner Selbsttest / Übersicht beim direkten Aufruf.
    print(f"PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"RANDOM_STATE      = {RANDOM_STATE}")
    print(f"Pflicht-Tabellen  = {REQUIRED_TABLES}")
    print(f"Optionale Tabellen= {OPTIONAL_TABLES}")
