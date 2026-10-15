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
_CANDIDATES = [BACKEND_DIR / CSV_NAME, BACKEND_DIR.parent / "Frontend" / CSV_NAME]
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
        "Derivada (combinada)",
        "Es exactamente purchasing_power_per_inhabitant_eur * population (ratio "
        "0,9997-1,0007): se reconstruye sin error a partir de esas dos.",
    ),
    "number_of_households": (
        "Redundante (combinada)",
        "Spearman 0,998 con population. Su informacion se conserva en personas_por_hogar "
        "= population / number_of_households.",
    ),
    "purchasing_power_per_household_eur": (
        "Redundante (combinada)",
        "Spearman 0,956 con purchasing_power_per_inhabitant_eur. pp_hogar / pp_habitante es "
        "exactamente personas_por_hogar, que sustituye a ambas columnas.",
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
    "trip_distance_average": (
        "Telemetria (>40 % nulos)",
        "~60 % nulos estructurales (rol, eléctricos, grupo F, año). No se imputa; se excluye "
        "del analisis. La presencia de dato queda en has_telematics.",
    ),
    "trip_distance_median": (
        "Telemetria (>40 % nulos)",
        "Marcas de bin discretas + ~60 % nulos estructurales. Redundante con average; excluida.",
    ),
    "trip_distance_all_counter": (
        "Telemetria (>40 % nulos)",
        "~60 % nulos estructurales. Contador bruto; excluido (viajes_por_hora tampoco entra en el analisis).",
    ),
    "fuel_consumption_average": (
        "Telemetria (>40 % nulos)",
        "~60 % nulos (no aplica a eléctricos + ausencia estructural). Excluida del analisis.",
    ),
    "speed_average": (
        "Telemetria (>40 % nulos)",
        "~60 % nulos estructurales. Excluida; el perfil de uso no entra en la segmentacion de clientes.",
    ),
    "speed_median": (
        "Telemetria (>40 % nulos)",
        "Marcas de bin + ~60 % nulos. Excluida.",
    ),
    "speed_all_counter": (
        "Telemetria (>40 % nulos)",
        "~60 % nulos estructurales. Excluido.",
    ),
    "viajes_por_hora": (
        "Telemetria derivada",
        "Ratio sobre columnas excluidas (~60 % nulos de base). Excluida del analisis.",
    ),
    "seg_registrados_por_hora": (
        "Telemetria derivada",
        "Ratio sobre speed_all_counter (excluido). Excluida del analisis.",
    ),
    "asimetria_distancia": (
        "Telemetria derivada",
        "Ratio sobre trip_distance_* (excluidas). Excluida del analisis.",
    ),
    "asimetria_velocidad": (
        "Telemetria derivada",
        "Ratio sobre speed_* (excluidas). Excluida del analisis.",
    ),
    "operating_hours_total": (
        "Telemetria (>40 % nulos)",
        "~52 % nulos estructurales (eléctricos, rol, etc.). Excluida del analisis; "
        "la presencia de cualquier telemetria queda en has_telematics.",
    ),
}

# Plausibilidad fisica: valores fuera de rango se convierten en NaN (se registra el recuento).
PLAUSIBILITY_RULES = {
    "speed_average": ("> 150 km/h de velocidad MEDIA no es fisicamente plausible", lambda s: s > 150),
    "fuel_consumption_average": ("> 30 l/100 km no es plausible en turismos/furgonetas", lambda s: s > 30),
}

# Imputacion por catalogo: una denominacion comercial solo se usa si en los datos conocidos
# su valor es unico en >= 95 % de las filas y tiene al menos 5 filas.
CATALOG_PURITY = 0.95
CATALOG_MIN_N = 5

# Estrategia de nulos por tramo de % de nulos y mecanismo de ausencia.
NULL_TIERS = [
    (0, 5, "BAJO (<5 %)",
     "Imputacion simple (mediana / moda / categoria UNKNOWN). Eliminar filas solo si la variable es "
     "objetivo del analisis. Impacto despreciable en las distribuciones."),
    (5, 40, "MEDIO (5-40 %)",
     "Imputar con informacion relacionada (mediana por grupo, catalogo) + flag de imputado. "
     "Nunca media global: deforma la distribucion."),
    (40, 101, "ALTO (>40 %)",
     "No imputar a ciegas. 1) Recuperar desde otra fuente/columna si es posible; 2) si la ausencia es "
     "estructural, mantener NaN + indicador y analizar en una tabla separada; 3) eliminar la columna "
     "solo si ademas no aporta valor."),
]

NULL_STRATEGY: dict[str, tuple[str, str]] = {
    # columna: (mecanismo, estrategia aplicada)
    "trip_distance_average": ("Estructural/MAR (~60 %)", "Columna EXCLUIDA del analisis + has_telematics"),
    "trip_distance_median": ("Estructural/MAR (~60 %)", "Columna EXCLUIDA del analisis"),
    "trip_distance_all_counter": ("Estructural/MAR (~60 %)", "Columna EXCLUIDA del analisis"),
    "operating_hours_total": ("Estructural (~52 %)", "Columna EXCLUIDA del analisis + has_telematics"),
    "fuel_consumption_average": ("Estructural (~60 %)", "Columna EXCLUIDA del analisis"),
    "speed_average": ("Estructural (~60 %)", "Columna EXCLUIDA del analisis"),
    "speed_median": ("Estructural (~60 %)", "Columna EXCLUIDA del analisis"),
    "speed_all_counter": ("Estructural (~60 %)", "Columna EXCLUIDA del analisis"),
    "trip_distance_mode": ("Estructural + listas multimodales", "Columna EXCLUIDA"),
    "speed_mode": ("Estructural + listas multimodales", "Columna EXCLUIDA"),
    "viajes_por_hora": ("Derivada de telemetria excluida", "Columna EXCLUIDA del analisis"),
    "seg_registrados_por_hora": ("Derivada de telemetria excluida", "Columna EXCLUIDA del analisis"),
    "asimetria_distancia": ("Derivada de telemetria excluida", "Columna EXCLUIDA del analisis"),
    "asimetria_velocidad": ("Derivada de telemetria excluida", "Columna EXCLUIDA del analisis"),
    "fuel_type": ("MAR por modelo", "Catalogo vehicle_type + reglas nombre (d/CDI/EQ/e/de); ELECTRIC desde engine; resto UNKNOWN + flag"),
    "engine_concept": ("MAR por modelo", "Catalogo vehicle_type + reglas de nombre; resto UNKNOWN + flag"),
    "gender_derived": ("MAR por rol (USER/TEMP_USER)", "Categoria UNKNOWN (prediccion probada y descartada: no supera la linea base)"),
    "purchasing_power_per_inhabitant_eur": ("CP sin cruzar con tabla externa", "Mediana por pais + flag socio_missing"),
    "population": ("CP sin cruzar con tabla externa (+73 ceros)", "Mediana por pais + flag socio_missing"),
    "purchasing_power_per_household_eur": ("CP sin cruzar", "Columna EXCLUIDA (redundante)"),
    "purchasing_power_millions_eur": ("CP sin cruzar", "Columna EXCLUIDA (derivada)"),
    "number_of_households": ("CP sin cruzar", "Columna EXCLUIDA (redundante)"),
    "zipcode": ("Falta en origen", "Columna EXCLUIDA (identificador geografico)"),
    "date_of_first_registration": ("MAR (modelos antiguos, rol LEGACY)",
        "Prediccion por vehiculo (gradient boosting, acotada al rango de su serie) + flag registration_date_missing"),
    "model_class": ("Series nuevas sin class en datos (540, 244…)",
        "Catalogo model_series -> vehicle_type + reglas de nombre (GLC/GLB/C/EQ…); resto UNKNOWN + flag"),
    "vehicle_type": ("Probable MCAR", "Categoria UNKNOWN (columna excluida del analisis)"),
    "months_as_mb_customer": ("MCAR (4 filas)", "Mediana"),
}


def null_tier(pct: float) -> tuple[str, str]:
    for lo, hi, name, rec in NULL_TIERS:
        if lo <= pct < hi:
            return name, rec
    return NULL_TIERS[-1][2], NULL_TIERS[-1][3]


def null_strategy_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla por columna con nulos: % nulos, tramo, mecanismo, estrategia y % nulos tras limpiar."""
    raw = df_raw.drop(columns=["Unnamed: 0"], errors="ignore")
    pct = raw.isna().mean() * 100
    rows = []
    for col, p in pct[pct > 0].sort_values(ascending=False).items():
        tier, _ = null_tier(p)
        mech, strat = NULL_STRATEGY.get(col, ("", ""))
        row = {"columna": col, "% nulos crudo": round(p, 2), "tramo": tier, "mecanismo": mech, "estrategia": strat}
        if df_clean is not None and col in df_clean.columns:
            s = df_clean[col]
            after = (s.isna() | s.astype("string").eq("UNKNOWN")).mean() * 100
            row["% nulo/UNKNOWN tras limpiar"] = round(after, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("columna")


# Funciones a aplicar por tipo de problema de fila (nombre -> para que sirve).
ROW_TREATMENT_FUNCTIONS: dict[str, str] = {
    "clean": "Orquestador: aplica todo el tratamiento y devuelve el dataset listo",
    "drop_exact_duplicates": "Elimina filas 100 % identicas (si las hubiera)",
    "normalize_zipcode": "Normaliza CP (quita prefijo pais; formato invalido -> NaN)",
    "nullify_implausible": "Pasa a NaN valores fisicamente imposibles (speed/consumo)",
    "impute_engine_fuel_from_catalog": "Recupera engine_concept/fuel_type desde vehicle_type + reglas",
    "fill_unknown_categoricals": "Nulos categoricos informativos -> categoria UNKNOWN",
    "flag_has_telematics": "Marca filas con/sin telemetria; NO imputa telemetria",
    "impute_socio_by_country": "Imputa socioeconomicas con mediana del pais + flag socio_missing",
    "impute_months_median": "Imputa los pocos nulos de months_as_mb_customer con la mediana",
    "derive_vehicle_age": "Crea vehicle_age_years desde date_of_first_registration",
}


def diagnose_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Diagnostico de TODAS las filas: nulos, repeticiones, accion y funciones a aplicar.

    Una fila = relacion cliente-vehiculo. No se borran filas por tener nulos;
    solo se eliminarian duplicados exactos (en este dataset: 0).
    """
    df = df_raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    tel_core = [c for c in (
        "trip_distance_average", "operating_hours_total",
        "fuel_consumption_average", "speed_average",
    ) if c in df.columns]

    n_per_client = df.groupby("client_id")["client_id"].transform("size")
    n_per_vehicle = df.groupby("vehicle_id")["vehicle_id"].transform("size")
    nulls = df.isna()

    out = pd.DataFrame({
        "fila": np.arange(len(df), dtype=np.int32),
        "client_id": df["client_id"].values,
        "vehicle_id": df["vehicle_id"].values,
        "tiene_nulo": nulls.any(axis=1).values,
        "n_nulos": nulls.sum(axis=1).astype(np.int16).values,
        "duplicado_exacto": df.duplicated(keep=False).values,
        "par_cliente_vehiculo_repetido": df.duplicated(
            ["client_id", "vehicle_id"], keep=False
        ).values,
        "vehiculo_en_varias_filas": (n_per_vehicle > 1).values,
        "cliente_flota_ge10": (n_per_client >= 10).values,
        "sin_telematica": df[tel_core].isna().all(axis=1).values if tel_core else False,
        "nulo_motor_o_fuel": (
            df["engine_concept"].isna() | df["fuel_type"].isna()
        ).values,
        "nulo_genero": df["gender_derived"].isna().values,
        "nulo_socio": (
            df["population"].isna() | df["purchasing_power_per_inhabitant_eur"].isna()
        ).values,
        "valor_imposible": (
            df["speed_average"].gt(150).fillna(False)
            | df["fuel_consumption_average"].gt(30).fillna(False)
        ).values,
    })

    # Accion prioritaria por fila + funciones concretas a ejecutar
    accion = np.full(len(df), "CONSERVAR + clean()", dtype=object)
    funciones = np.full(len(df), "clean", dtype=object)

    # Prioridad: duplicados > imposibles > bloques de nulos > flota (aviso) > OK
    m_dup = out["duplicado_exacto"] | out["par_cliente_vehiculo_repetido"]
    accion[m_dup] = "ELIMINAR (duplicado)"
    funciones[m_dup] = "drop_exact_duplicates"

    m_imp = (~m_dup) & out["valor_imposible"]
    accion[m_imp] = "CONSERVAR fila; celda imposible -> NaN"
    funciones[m_imp] = "nullify_implausible + clean"

    m_tel = (~m_dup) & (~m_imp) & out["sin_telematica"] & out["tiene_nulo"]
    accion[m_tel] = "CONSERVAR; telemetria NaN + flag (no imputar / no borrar)"
    funciones[m_tel] = "flag_has_telematics + fill_unknown_categoricals + clean"

    m_mot = (~m_dup) & (~m_imp) & (~out["sin_telematica"]) & out["nulo_motor_o_fuel"]
    accion[m_mot] = "CONSERVAR; recuperar motor/fuel o UNKNOWN"
    funciones[m_mot] = "impute_engine_fuel_from_catalog + fill_unknown_categoricals + clean"

    m_soc = (
        (~m_dup) & (~m_imp) & (~out["sin_telematica"])
        & (~out["nulo_motor_o_fuel"]) & out["nulo_socio"]
    )
    accion[m_soc] = "CONSERVAR; imputar socio (mediana pais) + flag"
    funciones[m_soc] = "impute_socio_by_country + clean"

    m_gen = (
        (~m_dup) & (~m_imp) & (~out["sin_telematica"])
        & (~out["nulo_motor_o_fuel"]) & (~out["nulo_socio"]) & out["nulo_genero"]
    )
    accion[m_gen] = "CONSERVAR; genero -> UNKNOWN"
    funciones[m_gen] = "fill_unknown_categoricals + clean"

    m_other_null = (
        (~m_dup) & (~m_imp) & out["tiene_nulo"]
        & (~out["sin_telematica"]) & (~out["nulo_motor_o_fuel"])
        & (~out["nulo_socio"]) & (~out["nulo_genero"])
    )
    accion[m_other_null] = "CONSERVAR; imputar/UNKNOWN segun columna (clean)"
    funciones[m_other_null] = "clean"

    m_ok = (~m_dup) & (~out["tiene_nulo"]) & (~out["valor_imposible"])
    accion[m_ok] = "CONSERVAR (fila completa; sin nulos ni duplicados)"
    funciones[m_ok] = "clean (solo derivadas/normalizacion)"

    # Aviso de flota: no cambia la accion principal, se anota
    out["accion"] = accion
    out["funciones_a_aplicar"] = funciones
    out["aviso_flota"] = np.where(
        out["cliente_flota_ge10"],
        "Cliente-flota: pondera en medias a nivel fila; reportar tambien a nivel cliente",
        "",
    )
    out["tiene_valor_repetido"] = (
        out["duplicado_exacto"]
        | out["par_cliente_vehiculo_repetido"]
        | out["vehiculo_en_varias_filas"]
    )
    return out


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


def catalog_mapping(df: pd.DataFrame, target: str, key: str,
                    purity: float = CATALOG_PURITY, min_n: int = CATALOG_MIN_N) -> pd.Series:
    """key -> valor de target, solo para claves cuyo valor es (casi) unico en los datos conocidos."""
    known = df.dropna(subset=[target, key])
    counts = known.groupby([key, target]).size()
    total = counts.groupby(level=0).sum()
    top = counts.sort_values(ascending=False).groupby(level=0).head(1).reset_index(level=1)
    top.columns = ["val", "n_top"]
    top["pureza"] = top["n_top"] / total.reindex(top.index)
    top["n"] = total.reindex(top.index)
    return top.loc[(top["pureza"] >= purity) & (top["n"] >= min_n), "val"]


def impute_from_catalog(df: pd.DataFrame, target: str, key: str) -> tuple[pd.Series, pd.Series]:
    mapping = catalog_mapping(df, target, key)
    filled = df[target].fillna(df[key].map(mapping))
    return filled, df[target].isna() & filled.notna()


def _engine_from_name(engine: pd.Series, vehicle_type: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reglas sobre la denominacion comercial, solo para lo que el catalogo no cubrio.
    Validadas sobre filas con motor conocido: EQ*/e-Vito... -> ELECTRIC (100 % acierto),
    '<num> e'/'<num> de' -> PLUGIN_HYBRID (99 %)."""
    vt = vehicle_type.fillna("")
    out = engine.copy()
    ev = vt.str.match(r"^(?:AMG )?EQ[A-Z]?\b|^e(?:Vito|Sprinter|Citan)")
    phev = vt.str.contains(r"\b\d{3} (?:e|de)\b", case=False)
    m_ev = out.isna() & ev
    out[m_ev] = "ELECTRIC"
    m_ph = out.isna() & phev
    out[m_ph] = "PLUGIN_HYBRID"
    return out, m_ev | m_ph


# Denominacion comercial -> model_class (solo patrones con 100 % acierto en filas conocidas;
# orden de mas especifico a mas general: EQ* antes que E, GLC antes que G…).
_MODEL_CLASS_PATTERNS: list[tuple[str, str]] = [
    (r"^(?:AMG )?EQE\b", "EQE-Class"),
    (r"^(?:AMG )?EQS\b", "EQS-Class"),
    (r"^(?:AMG )?EQA\b", "EQA-/EQB-Class"),
    (r"^(?:AMG )?EQB\b", "EQA-/EQB-Class"),
    (r"^(?:AMG )?EQC\b", "EQC-Class"),
    (r"^GLC\b", "GLC-Class"),
    (r"^GLB\b", "B-/GLA-/GLB-Class"),
    (r"^CLS\b", "CLS-Class"),
    (r"^C \d", "C-Class"),
    (r"^G \d", "G-Class"),
    (r"VLE|Vito|Viano", "V-Class / Vito"),
]


def _model_class_from_name(model_class: pd.Series, vehicle_type: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Clase de modelo desde vehicle_type para series nuevas sin filas con class en el dataset."""
    vt = vehicle_type.fillna("").astype(str)
    out = model_class.copy()
    applied = pd.Series(False, index=out.index)
    miss = out.isna()
    for pattern, cls in _MODEL_CLASS_PATTERNS:
        m = miss & ~applied & vt.str.contains(pattern, case=False, regex=True, na=False)
        out.loc[m] = cls
        applied |= m
    return out, applied


def _fuel_from_name(fuel: pd.Series, vehicle_type: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Combustible desde la denominacion comercial (MB: '220 d' = diesel, 'EQ*' = electrico).

    Orden de reglas (validado sobre filas con fuel conocido):
    EQ/eVito -> ELECTRIC (100 %); '350 de' -> DIESEL,ELECTRIC; '350 e' -> PETROL,ELECTRIC (~72 %);
    CDI/Bluetec o '220 d' -> DIESEL (100 %).
    """
    vt = vehicle_type.fillna("").astype(str)
    out = fuel.copy()
    miss = out.isna()
    applied = pd.Series(False, index=out.index)

    rules: list[tuple[pd.Series, str]] = [
        (vt.str.match(r"^(?:AMG )?EQ[A-Z]?\b|^e(?:Vito|Sprinter|Citan)", na=False), "ELECTRIC"),
        (vt.str.contains(r"\b\d{3}\s+de\b", case=False, na=False), "DIESEL,ELECTRIC"),
        (vt.str.contains(r"\b\d{3}\s+e\b", case=False, na=False), "PETROL,ELECTRIC"),
        (vt.str.contains(r"CDI|BLUETEC", case=False, na=False), "DIESEL"),
        (vt.str.contains(r"\b\d{2,3}\s+d\b", case=False, na=False), "DIESEL"),
    ]
    for mask, val in rules:
        m = miss & mask & ~applied
        out.loc[m] = val
        applied |= m
    return out, applied


REG_FEATURES = ["model_series", "vehicle_type", "model_class", "product_group", "engine_concept", "role",
                "is_current_vehicle", "months_as_mb_customer", "country_code"]


def predict_registration_year(df: pd.DataFrame, train: pd.Series | None = None) -> pd.Series:
    """Anyo decimal de primer registro predicho por vehiculo (gradient boosting sobre REG_FEATURES).

    `train`: mascara de filas cuya fecha se usa para entrenar (por defecto, todas las que la tienen).
    La prediccion se acota al rango P5-P95 de los anyos conocidos de su model_series: el modelo
    extrapola mal en series antiguas que casi nunca tienen fecha (p. ej. CLK).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    fecha = pd.to_datetime(df["date_of_first_registration"], errors="coerce")
    year = fecha.dt.year + (fecha.dt.dayofyear - 1) / 365.25
    train = year.notna() if train is None else (train & year.notna())

    X = df[REG_FEATURES].copy()
    for c in REG_FEATURES:
        if c == "months_as_mb_customer":
            continue
        s = X[c].astype(str)
        X[c] = s.where(s.isin(s.value_counts().index[:250]), "OTRO").astype("category")
    first = ~df["vehicle_id"].duplicated()
    fit = first & train
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", loss="absolute_error",
                                          max_iter=400, learning_rate=0.08, random_state=0)
    model.fit(X[fit], year[fit])
    pred = pd.Series(model.predict(X[first]), index=df.index[first])

    serie = df.loc[first, "model_series"].astype(str)
    known = pd.DataFrame({"serie": df.loc[fit, "model_series"].astype(str), "y": year[fit]})
    rango = known.groupby("serie")["y"].agg(lo=lambda s: s.quantile(.05), hi=lambda s: s.quantile(.95), n="size")
    rango = rango[rango["n"] >= 5]
    pred = pred.clip(serie.map(rango["lo"]), serie.map(rango["hi"])).clip(year[fit].min(), year[fit].max())
    return df["vehicle_id"].map(pd.Series(pred.to_numpy(), index=df.loc[first, "vehicle_id"]))


def _year_to_date(y: pd.Series) -> pd.Series:
    base = pd.to_datetime(np.floor(y).astype("Int64").astype(str) + "-01-01", errors="coerce")
    return (base + pd.to_timedelta((y - np.floor(y)) * 365.25, unit="D")).dt.normalize()


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
    # nulos concentrados en modelos antiguos (no MCAR): se predicen desde el vehiculo + flag
    df["registration_date_missing"] = df["date_of_first_registration"].isna()
    if df["registration_date_missing"].any():
        pred = _year_to_date(predict_registration_year(df))
        df["date_of_first_registration"] = df["date_of_first_registration"].fillna(pred)
    log["date_of_first_registration_imputada"] = int(df["registration_date_missing"].sum())
    df["vehicle_age_years"] = (
        (ref_date - df["date_of_first_registration"]).dt.days / 365.25
    ).round(2)
    log["fecha_referencia"] = str(ref_date.date())

    # 5a. Motor/combustible: recuperar desde el catalogo (vehicle_type) ------
    eng_before = df["engine_concept"].isna()
    df["engine_concept"], m_cat = impute_from_catalog(df, "engine_concept", "vehicle_type")
    df["engine_concept"], m_rule = _engine_from_name(df["engine_concept"], df["vehicle_type"])
    df["engine_concept_imputed"] = eng_before & df["engine_concept"].notna()
    log["engine_concept_imputado_catalogo"] = int(m_cat.sum())
    log["engine_concept_imputado_regla_nombre"] = int(m_rule.sum())

    fuel_before = df["fuel_type"].isna()
    df["fuel_type"], m_cat = impute_from_catalog(df, "fuel_type", "vehicle_type")
    df["fuel_type"], m_rule = _fuel_from_name(df["fuel_type"], df["vehicle_type"])
    m_ev = df["fuel_type"].isna() & df["engine_concept"].eq("ELECTRIC")
    df.loc[m_ev, "fuel_type"] = "ELECTRIC"
    df["fuel_type_imputed"] = fuel_before & df["fuel_type"].notna()
    log["fuel_type_imputado_catalogo"] = int(m_cat.sum())
    log["fuel_type_imputado_regla_nombre"] = int(m_rule.sum())
    log["fuel_type_imputado_desde_ELECTRIC"] = int(m_ev.sum())

    mc_before = df["model_class"].isna()
    df["model_class"], m_mc_ser = impute_from_catalog(df, "model_class", "model_series")
    df["model_class"], m_mc_vt = impute_from_catalog(df, "model_class", "vehicle_type")
    df["model_class"], m_mc_rule = _model_class_from_name(df["model_class"], df["vehicle_type"])
    log["model_class_imputado_model_series"] = int(m_mc_ser.sum())
    log["model_class_imputado_vehicle_type"] = int(m_mc_vt.sum())
    log["model_class_imputado_regla_nombre"] = int(m_mc_rule.sum())
    df["model_class_imputed"] = mc_before & df["model_class"].notna()

    # 5b. Categoricas restantes: nulos informativos -> categoria explicita ---
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
    # personas_por_hogar se calcula antes de imputar population (con los valores reales)
    df["personas_por_hogar"] = df["population"] / df["number_of_households"].where(df["number_of_households"] > 0)
    for col in ("purchasing_power_per_inhabitant_eur", "population", "personas_por_hogar"):
        med = df.groupby("country_code")[col].transform("median")
        df[col] = df[col].fillna(med)

    # 7b. Multicolinealidad: combinar en "nivel + ratio" en vez de eliminar --
    df = add_combined_features(df)

    # 8. months_as_mb_customer: 4 nulos -> mediana ---------------------------
    n = int(df["months_as_mb_customer"].isna().sum())
    df["months_as_mb_customer"] = df["months_as_mb_customer"].fillna(df["months_as_mb_customer"].median())
    log["months_as_mb_customer_imputados"] = n

    return df


def add_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sustituye pares muy correlacionados por (nivel, ratio). La transformacion es reversible:
    B = ratio * A, de modo que no se pierde informacion pero desaparece la colinealidad."""
    df["log_population"] = np.log10(df["population"])
    # con < 1 h de uso el cociente es inestable (denominador ~0) -> NaN
    hours = df["operating_hours_total"].where(df["operating_hours_total"] >= 1)
    df["viajes_por_hora"] = df["trip_distance_all_counter"] / hours
    df["seg_registrados_por_hora"] = df["speed_all_counter"] / hours
    # media / mediana > 1: unos pocos trayectos largos (o tramos rapidos) tiran de la media hacia arriba
    df["asimetria_distancia"] = df["trip_distance_average"] / df["trip_distance_median"].where(df["trip_distance_median"] > 0)
    df["asimetria_velocidad"] = df["speed_average"] / df["speed_median"].where(df["speed_median"] > 0)
    return df


# Columna original -> (variable que la sustituye en el analisis, relacion)
COMBINED_FEATURES: dict[str, tuple[str, str]] = {
    "population": ("log_population", "log10(population): misma variable, escala que corrige la asimetria"),
    "number_of_households": ("personas_por_hogar", "population / number_of_households"),
    "purchasing_power_per_household_eur": ("personas_por_hogar",
        "pp_hogar / pp_habitante = population / hogares (cociente exacto 1,0000)"),
    "purchasing_power_millions_eur": ("purchasing_power_per_inhabitant_eur + log_population",
        "pp_millones = pp_habitante x population (exacto)"),
    "trip_distance_all_counter": ("viajes_por_hora", "n_viajes / operating_hours_total"),
    "speed_all_counter": ("seg_registrados_por_hora",
        "speed_all_counter / operating_hours_total ~ 3.600: son segundos registrados; el ratio mide la cobertura del registro"),
    "trip_distance_median": ("asimetria_distancia", "trip_distance_average / trip_distance_median"),
    "speed_median": ("asimetria_velocidad", "speed_average / speed_median"),
    "trip_distance_average": ("speed_average / viajes_por_hora",
        "km por trayecto ~ velocidad media / viajes por hora (aprox.: mediana del cociente 1,07)"),
}


def analytic_features(df_clean: pd.DataFrame) -> list[str]:
    """Columnas que SI entran en el analisis/modelado (clean menos excluidas + derivadas)."""
    derived = ["vehicle_age_years", "has_telematics", "socio_missing"]
    keep = [c for c in df_clean.columns if c not in EXCLUDED_COLUMNS and not c.endswith("_multimodal")
            and c not in COMBINED_FEATURES
            and c not in ("registration_date_missing", "model_class_missing", "model_class_imputed",
                          "engine_concept_imputed", "fuel_type_imputed")]
    return [c for c in keep if c not in derived] + derived


def drop_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas que conservan algun NaN (UNKNOWN cuenta como categoria, no como nulo)."""
    return df.loc[:, df.notna().all()]


def load_clean(path: str | os.PathLike | None = None, drop_nulls: bool = False) -> pd.DataFrame:
    df = clean(load_raw(path))
    return drop_null_columns(df) if drop_nulls else df

