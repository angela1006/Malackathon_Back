"""Módulo para la valoración y tratamiento de datos nulos en 'gender_derived'.

Uso:
    py -m eda.analizar_gender
    py eda/analizar_gender.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda import cleaning as C


def evaluar_nulos_gender(df: pd.DataFrame) -> dict[str, Any]:
    """Evalúa en profundidad los valores nulos de la columna 'gender_derived'.
    
    Calcula métricas a nivel de fila y de cliente único, así como correlaciones
    con rol, edad, país, antigüedad y financiación para diagnosticar el mecanismo
    de ausencia (MCAR vs MAR).
    """
    col = "gender_derived"
    total_filas = len(df)
    nulos_filas = int(df[col].isna().sum())
    pct_filas = round((nulos_filas / total_filas) * 100, 2)

    # Nivel cliente único
    cl = df.drop_duplicates("client_id").copy()
    total_clientes = len(cl)
    nulos_clientes = int(cl[col].isna().sum())
    pct_clientes = round((nulos_clientes / total_clientes) * 100, 2)

    # Distribución por categoría a nivel cliente
    dist_clientes = (
        cl[col]
        .fillna("NULL / DESCONOCIDO")
        .value_counts(dropna=False)
        .to_dict()
    )
    dist_clientes_pct = (
        cl[col]
        .fillna("NULL / DESCONOCIDO")
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
        .to_dict()
    )

    # Ausencia según el rol del usuario en el vehículo
    nulos_por_rol = (
        df.groupby("role", observed=True)[col]
        .agg(total="size", nulos=lambda s: int(s.isna().sum()), pct_nulo=lambda s: round(s.isna().mean() * 100, 2))
        .reset_index()
        .to_dict(orient="records")
    )

    # Ausencia según el rango de edad (nivel cliente)
    nulos_por_edad = (
        cl.groupby("age_range", observed=True)[col]
        .agg(total="size", nulos=lambda s: int(s.isna().sum()), pct_nulo=lambda s: round(s.isna().mean() * 100, 2))
        .reset_index()
        .to_dict(orient="records")
    )

    # Ausencia según el país (nivel cliente)
    nulos_por_pais = (
        cl.groupby("country_code", observed=True)[col]
        .agg(total="size", nulos=lambda s: int(s.isna().sum()), pct_nulo=lambda s: round(s.isna().mean() * 100, 2))
        .reset_index()
        .to_dict(orient="records")
    )

    # Ausencia según el estado financiero/leasing
    nulos_por_financiacion = (
        cl.groupby("finance_leasing_status", observed=True)[col]
        .agg(total="size", nulos=lambda s: int(s.isna().sum()), pct_nulo=lambda s: round(s.isna().mean() * 100, 2))
        .reset_index()
        .to_dict(orient="records")
    )

    # Comparativa de antigüedad y poder adquisitivo (Con dato vs Nulo)
    cl["gender_is_null"] = cl[col].isna()
    comp_antiguedad = {
        "con_genero_mediana_meses": float(cl.loc[~cl["gender_is_null"], "months_as_mb_customer"].median()),
        "con_genero_media_meses": round(float(cl.loc[~cl["gender_is_null"], "months_as_mb_customer"].mean()), 1),
        "nulo_mediana_meses": float(cl.loc[cl["gender_is_null"], "months_as_mb_customer"].median()),
        "nulo_media_meses": round(float(cl.loc[cl["gender_is_null"], "months_as_mb_customer"].mean()), 1),
    }

    comp_poder_adq = {
        "con_genero_mediana_eur": float(cl.loc[~cl["gender_is_null"], "purchasing_power_per_inhabitant_eur"].median()),
        "nulo_mediana_eur": float(cl.loc[cl["gender_is_null"], "purchasing_power_per_inhabitant_eur"].median()),
    }

    return {
        "resumen_filas": {
            "total": total_filas,
            "nulos": nulos_filas,
            "pct_nulos": pct_filas,
        },
        "resumen_clientes": {
            "total": total_clientes,
            "nulos": nulos_clientes,
            "pct_nulos": pct_clientes,
            "distribucion_conteo": dist_clientes,
            "distribucion_pct": dist_clientes_pct,
        },
        "patrones_ausencia": {
            "por_rol": nulos_por_rol,
            "por_edad": nulos_por_edad,
            "por_pais": nulos_por_pais,
            "por_financiacion": nulos_por_financiacion,
            "antiguedad_cliente": comp_antiguedad,
            "poder_adquisitivo": comp_poder_adq,
        },
        "diagnostico_mecanismo": "MAR (Missing at Random)",
        "recomendacion": "Imputar a categoría explícita 'UNKNOWN' y flag 'gender_derived_missing'. No eliminar ni imputar por moda.",
    }


def tratar_gender_derived(
    df: pd.DataFrame,
    fill_value: str = "UNKNOWN",
    add_missing_indicator: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Aplica el tratamiento recomendado para los nulos de 'gender_derived'.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene la columna 'gender_derived'.
    fill_value : str, default 'UNKNOWN'
        Valor con el que reemplazar los nulos.
    add_missing_indicator : bool, default True
        Si True, genera una columna booleana 'gender_derived_missing' indicando
        los registros que originalmente eran nulos.
    inplace : bool, default False
        Si True, modifica el DataFrame en el lugar. Si False, retorna una copia.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con la columna tratada.
    """
    target = df if inplace else df.copy()

    if "gender_derived" not in target.columns:
        raise KeyError("La columna 'gender_derived' no está presente en el DataFrame.")

    if add_missing_indicator:
        target["gender_derived_missing"] = target["gender_derived"].isna()

    target["gender_derived"] = target["gender_derived"].fillna(fill_value)
    return target


def imprimir_informe(res: dict[str, Any]) -> None:
    """Imprime por consola un informe formateado y legible."""
    print("=" * 78)
    print(" INFORME DE VALORACIÓN: DATOS NULOS EN 'gender_derived'")
    print("=" * 78)

    rf = res["resumen_filas"]
    rc = res["resumen_clientes"]
    print(f"\n1. MAGNITUD GENERAL:")
    print(f"  * A nivel de fila (relación):    {rf['nulos']:,} nulos de {rf['total']:,} ({rf['pct_nulos']} %)")
    print(f"  * A nivel de cliente único:      {rc['nulos']:,} nulos de {rc['total']:,} ({rc['pct_nulos']} %)")

    print(f"\n2. DISTRIBUCIÓN A NIVEL CLIENTE:")
    for cat, pct in rc["distribucion_pct"].items():
        cnt = rc["distribucion_conteo"].get(cat, 0)
        print(f"  * {cat:20s}: {cnt:>7,} clientes ({pct:>5.2f} %)")

    pat = res["patrones_ausencia"]
    print(f"\n3. PATRONES DE AUSENCIA (EVIDENCIA MAR - MISSING AT RANDOM):")

    print("\n  [A] Ausencia por Rol del usuario en el vehículo:")
    for r in pat["por_rol"]:
        print(f"      - {r['role']:10s}: {r['pct_nulo']:>5.2f} % nulos ({r['nulos']:,} / {r['total']:,})")

    print("\n  [B] Ausencia por Rango de edad (curva en 'U'):")
    for r in pat["por_edad"]:
        print(f"      - {str(r['age_range']):10s}: {r['pct_nulo']:>5.2f} % nulos ({r['nulos']:,} / {r['total']:,})")

    print("\n  [C] Ausencia por País:")
    for r in pat["por_pais"]:
        print(f"      - {r['country_code']:10s}: {r['pct_nulo']:>5.2f} % nulos ({r['nulos']:,} / {r['total']:,})")

    print("\n  [D] Ausencia por Financiación / Leasing:")
    for r in pat["por_financiacion"]:
        print(f"      - {r['finance_leasing_status']:20s}: {r['pct_nulo']:>5.2f} % nulos ({r['nulos']:,} / {r['total']:,})")

    ant = pat["antiguedad_cliente"]
    print(f"\n  [E] Antigüedad media/mediana como cliente (months_as_mb_customer):")
    print(f"      - Clientes con género conocido: mediana = {ant['con_genero_mediana_meses']} meses (media: {ant['con_genero_media_meses']})")
    print(f"      - Clientes con género NULO:     mediana = {ant['nulo_mediana_meses']} meses (media: {ant['nulo_media_meses']})")

    print(f"\n4. DIAGNÓSTICO Y RECOMENDACIÓN:")
    print(f"  * Mecanismo: {res['diagnostico_mecanismo']}")
    print(f"  * Acción:    {res['recomendacion']}")
    print("=" * 78)


def main(data_path: str | Path | None = None) -> None:
    """Carga los datos, ejecuta la valoración e implementa el tratamiento."""
    raw = C.load_raw(data_path)
    resultado = evaluar_nulos_gender(raw)
    imprimir_informe(resultado)

    # Ejemplo de aplicación del tratamiento
    print("\n[PROBANDO TRATAMIENTO]")
    df_tratado = tratar_gender_derived(raw, fill_value="UNKNOWN", add_missing_indicator=True)
    nulos_despues = int(df_tratado["gender_derived"].isna().sum())
    conteo_unknown = int((df_tratado["gender_derived"] == "UNKNOWN").sum())
    print(f"  Nulos después del tratamiento: {nulos_despues}")
    print(f"  Registros clasificados como 'UNKNOWN': {conteo_unknown:,}")
    print(f"  Columna indicadora creada: 'gender_derived_missing' (True: {int(df_tratado['gender_derived_missing'].sum()):,})")


if __name__ == "__main__":
    main()
