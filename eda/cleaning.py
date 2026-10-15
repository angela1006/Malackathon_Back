"""Carga y limpieza del dataset del reto Mercedes-Benz (Malackathon).

Este modulo es reutilizable (por ejemplo desde la API FastAPI o desde un
notebook): `run_eda.py` lo usa para generar el analisis, y cualquier otro
consumidor puede llamar a `load_clean()` para obtener el dataset ya tratado.

Cada decision de tratamiento esta justificada en `TREATMENTS` y en el informe
`informe_eda.md`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
CSV_NAME = "datos_reto_mercedes_benz_malackathon.csv"
_CANDIDATES = [
    BACKEND_DIR / CSV_NAME,
    BACKEND_DIR.parent / CSV_NAME,
    BACKEND_DIR.parent / "Frontend" / CSV_NAME,
]
DEFAULT_DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

# --------------------------------------------------------------------------- #
# Clasificacion de columnas
# --------------------------------------------------------------------------- #
TELEMATICS_COLS = [
    "trip_distance_average",
    "trip_distance_mode",
    "trip_distance_median",
    "trip_distance_all_counter",
    "operating_hours_total",
    "fuel_consumption_average",
    "speed_average",
    "speed_mode",
    "speed_median",
    "speed_all_counter",
]
SOCIOECONOMIC_COLS = [
    "purchasing_power_per_inhabitant_eur",
    "purchasing_power_per_household_eur",
    "population",
    "number_of_households",
    "purchasing_power_millions_eur",
]

# Columnas excluidas del conjunto analitico: (categoria, motivo).
# Se mantienen en el dataset limpio (como claves o para trazabilidad) pero no
# se usan como variables para describir/modelar.
EXCLUDED_COLUMNS: dict[str, tuple[str, str]] = {
    "Unnamed: 0": (
        "Metadato",
        "Indice de fila exportado por pandas (0..n-1). No es informacion del negocio.",
    ),
    "client_id": (
        "Identificador",
        "Hash SHA-256 pseudonimizado del cliente. Cardinalidad ~173k, sin significado "
        "analitico; solo sirve como clave para agregar por cliente.",
    ),
    "vehicle_id": (
        "Identificador",
        "Hash SHA-256 pseudonimizado del vehiculo. Cardinalidad ~245k, sin significado "
        "analitico; solo sirve como clave de agregacion.",
    ),
    "zipcode": (
        "Identificador geografico",
        "Codigo postal como texto (~4,6k niveles). Es la clave con la que se enriquecieron "
        "las columnas socioeconomicas, cuya informacion ya esta en purchasing_power_* y "
        "population. Ademas tiene formatos inconsistentes.",
    ),
    "vehicle_type": (
        "Identificador de producto",
        "Denominacion comercial (p.ej. 'GLC 220 d 4MATIC'), 1.436 niveles. Alta "
        "cardinalidad y redundante con model_class + engine_concept.",
    ),
    "model_series": (
        "Codigo interno",
        "Codigo numerico de serie (p.ej. 177, 247). Es NOMINAL aunque sea un entero; "
        "tratarlo como numerico induciria orden falso. Redundante con model_class "
        "(82 de 88 series mapean a una unica clase).",
    ),
    "date_of_first_registration": (
        "Fecha cruda",
        "Se sustituye por la variable derivada vehicle_age_years (la fecha absoluta "
        "no generaliza y la edad es directamente interpretable).",
    ),
    "role_type": (
        "Redundante",
        "Funcion determinista de role (DRIVER=B2B; resto=B2C). Corr. perfecta, sin "
        "informacion adicional.",
    ),
    "consent_phone_communication_channel": (
        "Metadato de contacto/compliance",
        "Flag RGPD del canal de contacto. Informa de si se puede contactar, no del "
        "comportamiento del cliente/vehiculo. Se conserva solo para activacion.",
    ),
    "consent_epost_communication_channel": (
        "Metadato de contacto/compliance",
        "Flag RGPD del canal de contacto (e-mail). Ver nota anterior.",
    ),
    "consent_post_communication_channel": (
        "Metadato de contacto/compliance",
        "Flag RGPD del canal de contacto (correo postal). Ver nota anterior.",
    ),
    "purchasing_power_millions_eur": (
        "Derivada / redundante",
        "Es exactamente purchasing_power_per_inhabitant_eur * population (ratio "
        "0,9997-1,0007). No aporta informacion nueva.",
    ),
    "number_of_households": (
        "Redundante",
        "Spearman 0,998 con population. Multicolinealidad casi perfecta.",
    ),
    "purchasing_power_per_household_eur": (
        "Redundante",
        "Spearman 0,956 con purchasing_power_per_inhabitant_eur (ratio ~2,24 = "
        "tamano medio de hogar). Se conserva solo la version per capita.",
    ),
    "trip_distance_mode": (
        "Artefacto tecnico",
        "Moda de un histograma discretizado (marcas de bin). Columna de texto mixta: "
        "en ~2,3% de las filas es una lista con todos los bins (empate = sin moda real). "
        "Su informacion util ya esta en average/median.",
    ),
    "speed_mode": (
        "Artefacto tecnico",
        "Igual que trip_distance_mode: marcas de bin (0,125; 10,0625...) y listas "
        "multimodales (0,4%). Solo coincide con speed_median en el 43% de las filas.",
    ),
}

# Plausibilidad fisica: valores fuera de rango se convierten en NaN (se registra el recuento).
PLAUSIBILITY_RULES = {
    "speed_average": ("> 150 km/h de velocidad MEDIA no es fisicamente plausible", lambda s: s > 150),
    "fuel_consumption_average": ("> 30 l/100 km no es plausible en turismos/furgonetas", lambda s: s > 30),
}

SNAPSHOT_QUANTILE_NOTE = "fecha de referencia = maxima fecha de primer registro observada"


def resolve_data_path(path: str | os.PathLike | None = None) -> Path:
    p = Path(path or os.environ.get("EDA_DATA_PATH") or DEFAULT_DATA_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV en {p}. Pasalo como argumento o con EDA_DATA_PATH."
        )
    return p


def load_raw(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Lee el CSV tal cual, sin transformar (mantiene la columna indice sin nombre)."""
    df = pd.read_csv(resolve_data_path(path), low_memory=False)
    first = df.columns[0]
    if first.startswith("Unnamed"):
        df = df.rename(columns={first: "Unnamed: 0"})
    return df


# --------------------------------------------------------------------------- #
# Utilidades de limpieza
# --------------------------------------------------------------------------- #
_ZIP_OK = re.compile(r"^\d{4}$")


def _normalise_zip(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Devuelve (zip normalizado, mascara de valores que tenian formato invalido).

    AT y CH usan 4 digitos. Se detectan: prefijos de pais ('A-2100', 'CH-6340'),
    nombres de localidad ('Ottenbach') y codigos con 5+ digitos ('55605').
    """
    raw = s.astype("string").str.strip()
    stripped = raw.str.replace(r"^(CH|A|AT|D|DE)-", "", regex=True)
    ok = stripped.str.fullmatch(_ZIP_OK).fillna(False)
    invalid = raw.notna() & ~raw.str.fullmatch(_ZIP_OK).fillna(False)
    out = stripped.where(ok)
    return out, invalid


def _parse_mode(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convierte la columna *_mode a numerica. Las listas ('2.5,7.5,...') indican que
    todos los bins empatan (sin moda): se marcan como multimodal y valor NaN."""
    txt = s.astype("string")
    multimodal = txt.str.contains(",", na=False)
    num = pd.to_numeric(txt.where(~multimodal), errors="coerce")
    return num, multimodal


def clean(df_raw: pd.DataFrame, log: dict | None = None) -> pd.DataFrame:
    """Aplica el tratamiento propuesto. `log` (dict) recoge cuantos valores se tocaron."""
    log = log if log is not None else {}
    df = df_raw.copy()

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # 1. Codigos postales con formato inconsistente --------------------------
    df["zipcode"], invalid_zip = _normalise_zip(df["zipcode"])
    log["zipcode_formato_invalido"] = int(invalid_zip.sum())

    # 2. Columnas *_mode: texto mixto -> numerico + flag multimodal ----------
    for col in ("trip_distance_mode", "speed_mode"):
        df[col], multi = _parse_mode(df[col])
        df[f"{col}_multimodal"] = multi
        log[f"{col}_multimodal"] = int(multi.sum())

    # 3. Valores fisicamente implausibles -> NaN -----------------------------
    for col, (_, rule) in PLAUSIBILITY_RULES.items():
        mask = rule(df[col])
        log[f"{col}_implausible_a_nan"] = int(mask.sum())
        df.loc[mask, col] = np.nan

    # 4. Fecha de registro y edad del vehiculo -------------------------------
    df["date_of_first_registration"] = pd.to_datetime(
        df["date_of_first_registration"], errors="coerce"
    )
    ref_date = df["date_of_first_registration"].max()
    df["vehicle_age_years"] = (
        (ref_date - df["date_of_first_registration"]).dt.days / 365.25
    ).round(2)
    df["registration_date_missing"] = df["date_of_first_registration"].isna()
    log["fecha_referencia"] = str(ref_date.date())

    # 5. Categoricas: nulos informativos -> categoria explicita --------------
    for col in ("gender_derived", "engine_concept", "fuel_type", "model_class", "vehicle_type"):
        n = int(df[col].isna().sum())
        df[col] = df[col].fillna("UNKNOWN")
        log[f"{col}_a_UNKNOWN"] = n
    df["model_class_missing"] = df["model_class"].eq("UNKNOWN")

    # 6. Telematica: ausencia estructural -> flag, NO imputar ----------------
    core = ["trip_distance_average", "operating_hours_total", "fuel_consumption_average", "speed_average"]
    df["has_telematics"] = df[core].notna().any(axis=1)
    log["filas_con_telematica"] = int(df["has_telematics"].sum())

    # 7. Socioeconomicas: imputar por mediana del pais + flag ----------------
    # population == 0 en un codigo postal con clientes es incoherente -> NaN antes de imputar
    n_pop0 = int((df["population"] == 0).sum())
    df.loc[df["population"] == 0, "population"] = np.nan
    log["population_cero_a_nan"] = n_pop0
    df["socio_missing"] = df["population"].isna() | df["purchasing_power_per_inhabitant_eur"].isna()
    log["socio_missing"] = int(df["socio_missing"].sum())
    for col in ("purchasing_power_per_inhabitant_eur", "population"):
        med = df.groupby("country_code")[col].transform("median")
        df[col] = df[col].fillna(med)

    # 8. months_as_mb_customer: 4 nulos -> mediana ---------------------------
    n = int(df["months_as_mb_customer"].isna().sum())
    df["months_as_mb_customer"] = df["months_as_mb_customer"].fillna(df["months_as_mb_customer"].median())
    log["months_as_mb_customer_imputados"] = n

    return df


def analytic_features(df_clean: pd.DataFrame) -> list[str]:
    """Columnas que SI entran en el analisis/modelado (clean menos excluidas + derivadas)."""
    derived = ["vehicle_age_years", "has_telematics", "socio_missing"]
    keep = [c for c in df_clean.columns if c not in EXCLUDED_COLUMNS and not c.endswith("_multimodal")
            and c not in ("registration_date_missing", "model_class_missing")]
    return [c for c in keep if c not in derived] + derived


def load_clean(path: str | os.PathLike | None = None) -> pd.DataFrame:
    return clean(load_raw(path))

