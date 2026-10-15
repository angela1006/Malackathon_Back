"""Cubos (tablas agregadas por dimensiones) construidos sobre el dataset limpio.

Separan los datos segun su nivel y su completitud, para no mezclar en una misma tabla
columnas completas con el bloque de telemetria (~52 % nulo):

    tabla_clientes     1 fila por cliente (perfil + agregados de sus vehiculos), sin nulos estructurales
    tabla_telematica   solo filas cliente-vehiculo CON telemetria (tabla de hechos separada)
    cubo_clientes      clientes por pais x edad x genero x financiacion
    cubo_vehiculos     relaciones por pais x clase de modelo x motor x grupo de producto x vehiculo actual
    cubo_telematica    uso medio por clase de modelo x motor (solo filas con telemetria)

    python -m eda.cubes   -> escribe eda/output/cubos/*.csv y *.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda import cleaning as C

OUT = Path(__file__).resolve().parent / "output" / "cubos"
ELECTRIFIED = ["ELECTRIC", "PLUGIN_HYBRID", "MILD_HYBRID"]
# Todas las metricas de telemetria excluidas del analisis; tabla auxiliar = claves + filas con dato
TELEMATICS_KEEP: list[str] = []
MIN_CELL = 30  # celdas con menos observaciones se descartan (estadisticos inestables)


def tabla_clientes(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(_electrificado=df["engine_concept"].isin(ELECTRIFIED)).groupby("client_id")
    agg = g.agg(
        n_vehiculos=("vehicle_id", "nunique"),
        n_vehiculos_actuales=("is_current_vehicle", "sum"),
        n_electrificados=("_electrificado", "sum"),
        visitas_taller_total=("workshop_visits_warranty_goodwill", "sum"),
        tiene_telematica=("has_telematics", "any"),
        edad_vehiculo_mediana=("vehicle_age_years", "median"),
    )
    roles = df.groupby(["client_id", "role"]).size().sort_values(ascending=False)
    agg["rol_principal"] = roles.reset_index(level=1).groupby(level=0)["role"].first()
    client_cols = ["age_range", "gender_derived", "country_code", "months_as_mb_customer",
                   "currently_connected_vehicles", "finance_leasing_status",
                   "purchasing_power_per_inhabitant_eur", "population", "socio_missing"]
    out = agg.join(df.drop_duplicates("client_id").set_index("client_id")[client_cols])
    out["segmento"] = pd.cut(out["n_vehiculos"], [0, 1, 9, 10**6], labels=["1 vehiculo", "2-9 vehiculos", "flota (>=10)"])
    return out.reset_index()


def tabla_telematica(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["client_id", "vehicle_id", "model_class", "engine_concept", "product_group",
            "country_code", "role", "is_current_vehicle", "vehicle_age_years"] + TELEMATICS_KEEP
    return df.loc[df["has_telematics"], cols].reset_index(drop=True)


def cubo_clientes(tc: pd.DataFrame) -> pd.DataFrame:
    dims = ["country_code", "age_range", "gender_derived", "finance_leasing_status"]
    cube = tc.groupby(dims, observed=True).agg(
        n_clientes=("client_id", "size"),
        vehiculos_medios=("n_vehiculos", "mean"),
        antiguedad_mediana_meses=("months_as_mb_customer", "median"),
        pct_con_electrificado=("n_electrificados", lambda s: (s > 0).mean() * 100),
        pct_con_telematica=("tiene_telematica", "mean"),
        poder_adquisitivo_mediano=("purchasing_power_per_inhabitant_eur", "median"),
    ).reset_index()
    cube["pct_con_telematica"] *= 100
    return cube[cube["n_clientes"] >= MIN_CELL].round(2)


def cubo_vehiculos(df: pd.DataFrame) -> pd.DataFrame:
    dims = ["country_code", "model_class", "engine_concept", "product_group", "is_current_vehicle"]
    cube = df.groupby(dims, observed=True).agg(
        n_relaciones=("vehicle_id", "size"),
        n_vehiculos=("vehicle_id", "nunique"),
        edad_vehiculo_mediana=("vehicle_age_years", "median"),
        visitas_taller_media=("workshop_visits_warranty_goodwill", "mean"),
        pct_con_telematica=("has_telematics", "mean"),
    ).reset_index()
    cube["pct_con_telematica"] *= 100
    return cube[cube["n_relaciones"] >= MIN_CELL].round(2)


def cubo_telematica(tt: pd.DataFrame) -> pd.DataFrame:
    dims = ["model_class", "engine_concept"]
    cube = tt.groupby(dims, observed=True).agg(
        n_relaciones=("vehicle_id", "size"),
        n_vehiculos=("vehicle_id", "nunique"),
    ).reset_index()
    return cube[cube["n_relaciones"] >= MIN_CELL].round(2)


def build_all(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    df = C.load_clean() if df is None else df
    tc = tabla_clientes(df)
    tt = tabla_telematica(df)
    return {
        "tabla_clientes": tc,
        "tabla_telematica": tt,
        "cubo_clientes": cubo_clientes(tc),
        "cubo_vehiculos": cubo_vehiculos(df),
        "cubo_telematica": cubo_telematica(tt),
    }


def save_all(cubes: dict[str, pd.DataFrame]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in cubes.items():
        t.to_parquet(OUT / f"{name}.parquet", index=False)
        if not name.startswith("tabla_"):
            t.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    return OUT


if __name__ == "__main__":
    cubes = build_all()
    out = save_all(cubes)
    for k, v in cubes.items():
        print(f"{k:18s} {v.shape}")
    print("Escrito en", out)
