"""
data_download.py
================

Datenbeschaffung für den Home-Credit-Default-Risk-Datensatz.

WAS:    Lädt die Wettbewerbsdaten von Kaggle herunter (sofern API konfiguriert) und
        entpackt sie nach data/raw/. Bei fehlender API-Konfiguration wird eine klare,
        manuelle Anleitung ausgegeben statt eines kryptischen Fehlers.
WARUM:  Datenbeschaffung ist der erste reproduzierbare Schritt jeder Pipeline. Ein
        Skript dokumentiert die Datenherkunft eindeutig (Provenance) und macht das
        Setup auf einem neuen Rechner in einem Befehl möglich.
ALTERNATIVEN:
        - `opendatasets`-Paket (vereinfacht Kaggle-Login, aber Zusatzabhängigkeit)
        - direkter Download über die Weboberfläche (nicht skriptbar, nicht reproduzierbar)
        - DVC / Git-LFS für Datenversionierung (Best Practice in größeren Teams, hier Overkill)
RISIKEN/LIMITATIONEN:
        - Die Daten unterliegen den Kaggle-Wettbewerbsregeln; das Akzeptieren der Regeln
          ist Voraussetzung für den API-Download und muss einmalig manuell erfolgen.
        - Datenschutz: Auch wenn der Datensatz anonymisiert ist, handelt es sich um
          (ehemals) personenbezogene Finanzdaten -> nur für akademische Zwecke verwenden.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

try:
    # Relativer Import, wenn als Modul ausgeführt; Fallback für direkten Aufruf.
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

# Kaggle-Wettbewerbs-Slug (Teil der URL kaggle.com/c/<slug>).
COMPETITION_SLUG = "home-credit-default-risk"


def _kaggle_credentials_present() -> bool:
    """
    Prüft, ob Kaggle-Zugangsdaten vorhanden sind.

    WARUM zwei Orte: Die Kaggle-API akzeptiert entweder die Datei ~/.kaggle/kaggle.json
    ODER die Umgebungsvariablen KAGGLE_USERNAME/KAGGLE_KEY. Wir prüfen beide, damit
    sowohl lokale Setups als auch CI-/Cloud-Umgebungen funktionieren.
    """
    import os

    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _manual_instructions() -> str:
    """Gibt eine ausführliche Anleitung für den manuellen Download zurück."""
    return f"""
================================================================================
KAGGLE-API NICHT KONFIGURIERT  ->  MANUELLER DOWNLOAD ERFORDERLICH
================================================================================

Variante A — Kaggle-API einrichten (empfohlen, reproduzierbar):
  1. Kaggle-Konto erstellen: https://www.kaggle.com
  2. Wettbewerbsregeln EINMALIG akzeptieren (Pflicht!):
       https://www.kaggle.com/c/{COMPETITION_SLUG}/rules
     -> Button "I Understand and Accept". Ohne diesen Schritt verweigert die API
        den Download mit einem 403-Fehler.
  3. API-Token erzeugen: Kaggle -> Account -> Settings -> "Create New API Token".
     Es wird eine Datei kaggle.json heruntergeladen.
  4. Datei an den richtigen Ort legen und Rechte einschränken:
       Linux/macOS:
         mkdir -p ~/.kaggle
         mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
         chmod 600 ~/.kaggle/kaggle.json     # nur eigener Lesezugriff (Sicherheit)
       Windows (PowerShell):
         mkdir $env:USERPROFILE\\.kaggle
         move $env:USERPROFILE\\Downloads\\kaggle.json $env:USERPROFILE\\.kaggle\\
  5. Skript erneut ausführen:  python -m src.data_download

  Alternativ ohne Datei (z. B. in der Cloud) Umgebungsvariablen setzen:
       export KAGGLE_USERNAME="dein_username"
       export KAGGLE_KEY="dein_api_key"

Variante B — Manueller Download über den Browser:
  1. Seite öffnen: https://www.kaggle.com/c/{COMPETITION_SLUG}/data
  2. "Download All" klicken (ein ZIP-Archiv).
  3. Archiv entpacken und ALLE CSV-Dateien hier ablegen:
       {config.RAW_DIR}
  4. Erwartete Dateien (mindestens die Pflicht-Tabellen):
       {", ".join(config.TABLES[t]["filename"] for t in config.REQUIRED_TABLES)}
================================================================================
"""


def download_via_api() -> None:
    """
    Lädt das Wettbewerbsarchiv über die Kaggle-CLI und entpackt es nach data/raw/.

    WARUM subprocess statt kaggle-Python-API: Die CLI ist die offiziell dokumentierte,
    stabilste Schnittstelle und vermeidet ein hartes Import-Coupling an interne
    Klassen des kaggle-Pakets, die sich zwischen Versionen ändern können.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data_download] Lade Wettbewerb '{COMPETITION_SLUG}' nach {config.RAW_DIR} ...")

    # competitions download lädt EIN ZIP mit allen Dateien.
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", COMPETITION_SLUG,
        "-p", str(config.RAW_DIR),
    ]
    subprocess.run(cmd, check=True)

    # Haupt-ZIP entpacken; einige Dateien sind ihrerseits noch einmal gezippt.
    _extract_all_zips(config.RAW_DIR)
    print("[data_download] Download und Entpacken abgeschlossen.")
    verify_raw_data()


def _extract_all_zips(directory: Path) -> None:
    """Entpackt rekursiv alle .zip-Dateien im Verzeichnis (idempotent)."""
    for zip_path in directory.glob("*.zip"):
        print(f"[data_download] Entpacke {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)
    # Erneut suchen, falls verschachtelte ZIPs entstanden sind.
    nested = list(directory.glob("*.zip"))
    if nested:
        for zip_path in nested:
            target = directory / zip_path.stem
            if not target.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(directory)


def verify_raw_data() -> bool:
    """
    Prüft, ob die Pflicht-Tabellen tatsächlich vorliegen.

    WARUM: Frühes, explizites Scheitern mit klarer Meldung ist besser als ein
    späterer, schwer zu deutender FileNotFoundError mitten in der Pipeline.
    """
    missing = []
    for table in config.REQUIRED_TABLES:
        fpath = config.RAW_DIR / config.TABLES[table]["filename"]
        if not fpath.exists():
            missing.append(fpath.name)

    if missing:
        print(f"[data_download] FEHLT (Pflicht): {missing}")
        return False
    print(f"[data_download] OK: Alle {len(config.REQUIRED_TABLES)} Pflicht-Tabellen vorhanden.")
    return True


def main() -> None:
    """Einstiegspunkt: API nutzen falls möglich, sonst Anleitung ausgeben."""
    if verify_raw_data():
        print("[data_download] Daten bereits vorhanden – kein Download nötig.")
        return

    if _kaggle_credentials_present():
        try:
            download_via_api()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[data_download] API-Download fehlgeschlagen: {exc}")
            print(_manual_instructions())
    else:
        print(_manual_instructions())


if __name__ == "__main__":
    main()
