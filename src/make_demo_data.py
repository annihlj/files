"""
make_demo_data.py
=================

Erzeugt einen KLEINEN, SYNTHETISCHEN Datensatz, der die Struktur des echten
Home-Credit-Datensatzes nachbildet (gleiche Spaltennamen, Schlüssel, Beziehungen).

WAS:    Schreibt synthetische CSVs nach data/raw/, sodass die komplette Pipeline
        (Laden -> Aggregation -> Cleaning -> Features -> Modell -> Clustering -> App)
        ohne den echten Kaggle-Download getestet werden kann.
WARUM:  (1) Reproduzierbarkeit/Onboarding: Wer das Repo klont, kann sofort `make demo`
        ausführen und sieht eine lauffähige Pipeline.
        (2) CI/Tests: Schnelle, deterministische Daten für automatisierte Tests.
        (3) Datenschutz: Es werden KEINE echten personenbezogenen Daten benötigt.
WICHTIG / LIMITATION:
        Die synthetischen Daten enthalten teils EINGEBAUTE Zusammenhänge (z. B. höhere
        Ausfallwahrscheinlichkeit bei hoher Kredit-Einkommens-Relation), damit Modelle
        überhaupt etwas lernen können. Reale Ergebnisse, Kennzahlen und insbesondere
        Feature-Importances sind NICHT auf den echten Datensatz übertragbar. Für die
        wissenschaftliche Auswertung MUSS der echte Datensatz verwendet werden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


def _rng(seed: int = config.RANDOM_STATE) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_application(n: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Synthetische Haupttabelle auf SK_ID_CURR-Ebene inkl. eingebautem Signal."""
    rng = _rng(seed)
    sk_id = np.arange(100001, 100001 + n)

    income = rng.lognormal(mean=11.9, sigma=0.5, size=n).round(-2)        # ~ Jahreseinkommen
    credit = (income * rng.uniform(0.5, 6.0, size=n)).round(-2)           # Kreditbetrag
    annuity = (credit / rng.uniform(10, 40, size=n)).round(-1)           # Jahresrate
    goods = (credit * rng.uniform(0.7, 1.0, size=n)).round(-2)

    days_birth = -rng.integers(21 * 365, 69 * 365, size=n)               # negativ = Vergangenheit
    # Employed: ein Teil "arbeitslos/Rentner" mit Sentinel 365243 (wie im Original!)
    days_employed = -rng.integers(0, 40 * 365, size=n).astype(float)
    unemployed_mask = rng.random(n) < 0.18
    days_employed[unemployed_mask] = 365243  # bewusst der berüchtigte Ausreißer-Sentinel

    ext1 = rng.uniform(0, 1, size=n)
    ext2 = rng.uniform(0, 1, size=n)
    ext3 = rng.uniform(0, 1, size=n)
    # Externe Scores teils fehlend (wie im Original häufig).
    ext1[rng.random(n) < 0.55] = np.nan
    ext3[rng.random(n) < 0.20] = np.nan

    df = pd.DataFrame({
        "SK_ID_CURR": sk_id,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.66, 0.34]),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.34, 0.66]),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.69, 0.31]),
        "CNT_CHILDREN": rng.poisson(0.4, n),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": rng.choice(
            ["Working", "Commercial associate", "Pensioner", "State servant"], n,
            p=[0.52, 0.23, 0.18, 0.07]),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary / secondary special", "Higher education",
             "Incomplete higher", "Lower secondary"], n, p=[0.71, 0.24, 0.03, 0.02]),
        "NAME_FAMILY_STATUS": rng.choice(
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            n, p=[0.64, 0.18, 0.10, 0.06, 0.02]),
        "NAME_HOUSING_TYPE": rng.choice(
            ["House / apartment", "With parents", "Municipal apartment",
             "Rented apartment", "Office apartment", "Co-op apartment"], n,
            p=[0.89, 0.05, 0.03, 0.015, 0.01, 0.005]),
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.16, 0.74, 0.10]),
        "REGION_RATING_CLIENT_W_CITY": rng.choice([1, 2, 3], n, p=[0.16, 0.74, 0.10]),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_REGISTRATION": -rng.integers(0, 25 * 365, size=n).astype(float),
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "OCCUPATION_TYPE": rng.choice(
            ["Laborers", "Sales staff", "Core staff", "Managers", np.nan, "Drivers"], n,
            p=[0.26, 0.16, 0.15, 0.10, 0.23, 0.10]),
    })

    # ---- Eingebautes Signal fuer die Zielvariable -------------------------
    # Latentes Risiko steigt mit Kredit/Einkommen, sinkt mit externem Score & Alter.
    credit_income = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].clip(lower=1)
    ext_mean = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1).fillna(0.5)
    age = -df["DAYS_BIRTH"] / 365
    logit = (
        -2.6
        + 0.18 * (credit_income - credit_income.mean())
        - 2.2 * (ext_mean - 0.5)
        - 0.02 * (age - age.mean())
        + 0.4 * (df["NAME_INCOME_TYPE"].eq("Working").astype(int))
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df[config.TARGET] = (rng.random(n) < prob).astype(int)
    return df


def make_bureau(app_ids: np.ndarray, seed: int = config.RANDOM_STATE):
    """Synthetische bureau- und bureau_balance-Tabellen (1:n bzw. nested)."""
    rng = _rng(seed + 1)
    rows, bureau_ids = [], []
    next_bureau_id = 500001
    for cur in app_ids:
        k = rng.integers(0, 8)  # 0..7 frühere Bureau-Kredite
        for _ in range(k):
            bid = next_bureau_id
            next_bureau_id += 1
            bureau_ids.append(bid)
            status = rng.choice(["Active", "Closed"], p=[0.35, 0.65])
            rows.append({
                "SK_ID_CURR": cur,
                "SK_ID_BUREAU": bid,
                "CREDIT_ACTIVE": status,
                "CREDIT_TYPE": rng.choice(
                    ["Consumer credit", "Credit card", "Car loan", "Mortgage"],
                    p=[0.55, 0.30, 0.10, 0.05]),
                "DAYS_CREDIT": -rng.integers(1, 9 * 365),
                "AMT_CREDIT_SUM": np.round(rng.lognormal(11.5, 0.6), -2),
                "AMT_CREDIT_SUM_OVERDUE": float(rng.choice(
                    [0, 0, 0, rng.uniform(0, 5000)], 1)[0]),
                "CREDIT_DAY_OVERDUE": int(rng.choice([0, 0, 0, rng.integers(1, 120)], 1)[0]),
            })
    bureau = pd.DataFrame(rows)

    # bureau_balance: monatliche Status je SK_ID_BUREAU
    bb_rows = []
    for bid in bureau_ids:
        months = rng.integers(1, 24)
        for m in range(months):
            bb_rows.append({
                "SK_ID_BUREAU": bid,
                "MONTHS_BALANCE": -m,
                "STATUS": rng.choice(
                    ["C", "0", "1", "2", "X"], p=[0.55, 0.30, 0.08, 0.02, 0.05]),
            })
    bureau_balance = pd.DataFrame(bb_rows)
    return bureau, bureau_balance


def make_previous(app_ids: np.ndarray, seed: int = config.RANDOM_STATE):
    """Synthetische previous_application + installments_payments."""
    rng = _rng(seed + 2)
    prev_rows, prev_ids_map = [], {}
    next_prev_id = 900001
    for cur in app_ids:
        k = rng.integers(0, 6)
        ids = []
        for _ in range(k):
            pid = next_prev_id
            next_prev_id += 1
            ids.append(pid)
            prev_rows.append({
                "SK_ID_PREV": pid,
                "SK_ID_CURR": cur,
                "NAME_CONTRACT_STATUS": rng.choice(
                    ["Approved", "Refused", "Canceled", "Unused offer"],
                    p=[0.62, 0.19, 0.16, 0.03]),
                "NAME_CONTRACT_TYPE": rng.choice(
                    ["Cash loans", "Consumer loans", "Revolving loans"],
                    p=[0.45, 0.44, 0.11]),
                "AMT_CREDIT": np.round(rng.lognormal(11.6, 0.6), -2),
                "AMT_ANNUITY": np.round(rng.lognormal(9.5, 0.5), -1),
            })
        prev_ids_map[cur] = ids

    previous = pd.DataFrame(prev_rows)

    # installments_payments: je SK_ID_PREV mehrere Raten
    inst_rows = []
    for cur, ids in prev_ids_map.items():
        for pid in ids:
            n_inst = rng.integers(1, 24)
            for i in range(n_inst):
                amt_instalment = np.round(rng.lognormal(8.7, 0.4), -1)
                # Zahlung manchmal zu spaet/zu wenig
                delay = max(0, rng.normal(2, 6))
                pay_ratio = np.clip(rng.normal(1.0, 0.1), 0.4, 1.2)
                inst_rows.append({
                    "SK_ID_PREV": pid,
                    "SK_ID_CURR": cur,
                    "NUM_INSTALMENT_NUMBER": i + 1,
                    "DAYS_INSTALMENT": -rng.integers(1, 5 * 365),
                    "DAYS_ENTRY_PAYMENT": -rng.integers(1, 5 * 365) + delay,
                    "AMT_INSTALMENT": amt_instalment,
                    "AMT_PAYMENT": np.round(amt_instalment * pay_ratio, -1),
                })
    installments = pd.DataFrame(inst_rows)
    return previous, installments


def main(n_applicants: int = 5000) -> None:
    """Erzeugt alle synthetischen Pflicht-Tabellen und schreibt sie nach data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[make_demo_data] Erzeuge synthetische Daten fuer {n_applicants} Antragsteller ...")

    app = make_application(n_applicants)
    app.to_csv(config.RAW_DIR / "application_train.csv", index=False)
    print(f"  application_train.csv  -> {app.shape}, "
          f"Positivrate={app[config.TARGET].mean():.3f}")

    bureau, bureau_balance = make_bureau(app["SK_ID_CURR"].to_numpy())
    bureau.to_csv(config.RAW_DIR / "bureau.csv", index=False)
    bureau_balance.to_csv(config.RAW_DIR / "bureau_balance.csv", index=False)
    print(f"  bureau.csv             -> {bureau.shape}")
    print(f"  bureau_balance.csv     -> {bureau_balance.shape}")

    previous, installments = make_previous(app["SK_ID_CURR"].to_numpy())
    previous.to_csv(config.RAW_DIR / "previous_application.csv", index=False)
    installments.to_csv(config.RAW_DIR / "installments_payments.csv", index=False)
    print(f"  previous_application.csv -> {previous.shape}")
    print(f"  installments_payments.csv-> {installments.shape}")

    print("[make_demo_data] Fertig. ACHTUNG: synthetische Daten, nur zum Testen!")


if __name__ == "__main__":
    main()
