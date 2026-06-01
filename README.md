# Kreditwürdigkeitsprüfung mit Machine Learning
### Vorhersage von Kreditausfallrisiken und Segmentierung von Antragstellern

> **Akademisches Portfolio-Projekt.** Dieses Projekt dient ausschließlich zu
> Demonstrations- und Lernzwecken. Es darf **nicht** als alleinige Grundlage für
> reale Kreditentscheidungen verwendet werden (siehe Abschnitt *Kritische Diskussion & Ethik*).

---

## 1. Projektziel

Untersucht wird, ob sich das Risiko eines Kreditausfalls anhand von Antragsdaten,
früheren Kreditinformationen, Zahlungsdaten und weiteren Kundendaten vorhersagen lässt –
und ob sich Antragsteller sinnvoll in Gruppen (Cluster) einteilen lassen.

## 2. Forschungsfrage & Zieldefinition

**Zentrale Forschungsfrage**
> *Inwiefern können Machine-Learning-Modelle das Kreditausfallrisiko von Antragstellern
> anhand historischer Kredit- und Antragsdaten vorhersagen, und welche Gruppen von
> Antragstellern lassen sich durch Clustering identifizieren?*

**Teilfragen**
1. Welche Merkmale haben den größten Einfluss auf das vorhergesagte Ausfallrisiko?
2. Wie stark verbessert Feature Engineering die Modellleistung?
3. Welche Modelle eignen sich besonders für tabellarische Kreditrisikodaten?
4. Wie lassen sich Antragsteller anhand ihrer Merkmale sinnvoll segmentieren?
5. Welche ethischen und methodischen Grenzen bestehen bei automatisierter
   Kreditwürdigkeitsprüfung?

**Aufgabentyp:** (a) **Binäre Klassifikation** (überwacht) zur Vorhersage von `TARGET`;
(b) **Clustering** (unüberwacht) zur Segmentierung. Bewusst getrennt: Clustering nutzt
`TARGET` **nicht**.

## 3. Datenquelle

**Kaggle – „Home Credit Default Risk“**
(https://www.kaggle.com/c/home-credit-default-risk)

Relationaler Datensatz mit Haupt- und Nebentabellen. Haupt-ID: `SK_ID_CURR`.
Zielvariable `TARGET`: `1` = Zahlungsschwierigkeiten/erhöhtes Ausfallrisiko, `0` = sonst.

**Priorisierung der Tabellen**

| Priorität | Tabelle | Begründung |
|---|---|---|
| Pflicht | `application_train` | Zielebene, demografische & finanzielle Kernmerkmale |
| Pflicht | `bureau` | externe Kredithistorie (andere Institute) |
| Pflicht | `previous_application` | interne Antragshistorie (Home Credit) |
| Pflicht | `installments_payments` | konkretes Zahlungsverhalten (Verzug/Unterzahlung) |
| Optional | `bureau_balance`, `POS_CASH_balance`, `credit_card_balance` | monatliche Verlaufsdetails, sehr groß, abnehmender Grenznutzen |

> **Demo-Modus:** Ohne Kaggle-Daten erzeugt `src/make_demo_data.py` einen kleinen,
> **synthetischen** Datensatz mit identischer Struktur, sodass die gesamte Pipeline
> sofort lauffähig ist. Synthetische Ergebnisse sind **nicht** inhaltlich auf den
> echten Datensatz übertragbar.

## 4. Projektstruktur

```
credit-risk-ml/
├── data/{raw,interim,processed}/   # Rohdaten / Zwischendaten / Modellierungsdatensatz
├── notebooks/                      # 01..06: nachvollziehbare Analyse-Schritte
├── src/                            # wiederverwendbare Module (config, load, clean, ...)
├── models/                         # gespeicherte Modelle & Preprocessing-Pipelines
├── reports/{figures,tables}/       # erzeugte Abbildungen & Tabellen
├── app/streamlit_app.py            # interaktive Demo-App
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

```bash
# 1) Repository klonen und in das Verzeichnis wechseln
cd credit-risk-ml

# 2) Virtuelle Umgebung (empfohlen, isoliert Abhängigkeiten -> Reproduzierbarkeit)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3) Abhängigkeiten installieren
pip install -r requirements.txt

# 4) Jupyter-Kernel registrieren (optional)
python -m ipykernel install --user --name credit-risk-ml
```

## 6. Ausführung

```bash
# A) Daten beschaffen
#    Variante 1 – Kaggle-API (kaggle.json einrichten, siehe src/data_download.py):
python -m src.data_download
#    Variante 2 – manuell herunterladen und CSVs nach data/raw/ legen.

# B) Schneller End-to-End-Test ohne echte Daten (synthetisch):
python -m src.make_demo_data

# C) Notebooks der Reihe nach ausführen:
jupyter lab notebooks/

# D) Demo-App starten (nach Training):
streamlit run app/streamlit_app.py
```

### Schnellstart für die App (alles in einem Befehl)

Statt alle Notebooks durchzuklicken, kann die komplette Pipeline (Feature-Matrix +
Modell + Clustering) in einem Schritt erzeugt werden:

```bash
python -m src.build_pipeline          # erzeugt alle Artefakte in models/ und data/processed/
streamlit run app/streamlit_app.py    # startet die Weboberfläche
```

Die App öffnet sich automatisch im Browser (Standard: http://localhost:8501). Falls
das gespeicherte Modell fehlt, zeigt die App eine klare Anleitung an, statt abzustürzen.

## 7. Methodik (CRISP-DM)

Das Projekt folgt dem **CRISP-DM**-Prozess (Business/Data Understanding → Data
Preparation → Modeling → Evaluation → Deployment):

1. **01 Data Understanding** – Struktur, Granularität, Schlüssel, Datenqualität.
2. **02 Data Engineering** – Aggregation der Nebentabellen, Joins auf `SK_ID_CURR`.
3. **03 EDA** – Verteilungen, Korrelationen, Ausfall- vs. Nicht-Ausfall-Vergleich.
4. **04 Klassifikation** – Baseline → Logistic Regression → Random Forest → Gradient Boosting.
5. **05 Clustering** – K-Means (+ optional GMM/Hierarchisch), PCA-Visualisierung.
6. **06 Interpretation & Diskussion** – Feature Importance, SHAP, Ethik, Limitationen.

## 8. Modelle (geplant)

`DummyClassifier` (Baseline) · `LogisticRegression` · `RandomForestClassifier` ·
`HistGradientBoostingClassifier` (+ optional LightGBM/XGBoost).

## 9. Metriken

Wegen der **unausgewogenen** Zielvariable stehen **PR-AUC / Average Precision**,
**ROC-AUC**, **Recall** und **F1** im Vordergrund; Accuracy nur ergänzend.
Zusätzlich Confusion Matrix, Classification Report und – wo sinnvoll – Kalibrierungskurve.

## 10. Zentrale Ergebnisse

*Wird nach Durchlauf der Notebooks ergänzt.*

## 11. Kritische Diskussion & Ethik

*Ausführlich in Notebook 06.* Kernpunkte: Datenqualität & fehlende Werte, historische
Verzerrungen, Proxy-Variablen für sensible Merkmale (z. B. Geschlecht/Alter), Fairness,
Transparenz/Interpretierbarkeit, Unterschied zwischen **Vorhersage** und **Entscheidung**,
Datenschutz (DSGVO) und regulatorische Anforderungen.

## 12. Erklärung zur KI-Nutzung

Teile dieses Projekts (Code-Gerüst, Dokumentation, methodische Erläuterungen) wurden mit
Unterstützung eines KI-Assistenzsystems erstellt und anschließend geprüft, angepasst und
inhaltlich verantwortet durch die*den Autor*in. Alle Modellergebnisse sind eigenständig
nachvollzogen worden. Quellen im Theorieteil sind, soweit nicht eindeutig belegbar, als
*„zu prüfen“* markiert und nicht erfunden.

---
*Status: Schritte 1–5 (Projektplan, Forschungsfrage, Datenbeschaffung, Struktur,
Datenladen) implementiert und lauffähig. Folgeschritte 6–14 in Vorbereitung.*
