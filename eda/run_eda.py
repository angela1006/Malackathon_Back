"""EDA completo del dataset del reto Mercedes-Benz (Malackathon).

Uso (desde la carpeta Backend):

    python -m eda.run_eda                       # usa ../Frontend/datos_reto_...csv
    python -m eda.run_eda ruta/al/archivo.csv   # CSV alternativo

Genera en `eda/output/`:
    tables/   tablas CSV (estadisticos, nulos, inconsistencias, exclusiones...)
    figures/  graficos PNG (distribuciones, nulos, correlaciones...)
    resumen.json            cifras clave (para el informe)
    dataset_limpio.parquet  dataset tras el tratamiento propuesto
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eda import cleaning as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
BLUE, GREY, RED, GREEN = "#1f4e79", "#9aa5b1", "#c0392b", "#2e8b57"
SUMMARY: dict = {}


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close()


def save_tab(df: pd.DataFrame, name: str, index=True):
    df.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8-sig")


def section(t: str):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# 1. Vision general y estructura (unidad de analisis)
# --------------------------------------------------------------------------- #
def overview(raw: pd.DataFrame) -> pd.DataFrame:
    section("1. VISION GENERAL")
    n_rows, n_cols = raw.shape
    n_clients, n_veh = raw["client_id"].nunique(), raw["vehicle_id"].nunique()
    per_client = raw.groupby("client_id").size()
    per_vehicle = raw.groupby("vehicle_id").size()
    mem = raw.memory_usage(deep=True).sum() / 1e6
    SUMMARY["overview"] = {
        "filas": n_rows, "columnas": n_cols, "clientes_unicos": int(n_clients),
        "vehiculos_unicos": int(n_veh), "memoria_MB": round(mem, 1),
        "filas_duplicadas_exactas": int(raw.drop(columns="Unnamed: 0").duplicated().sum()),
        "pares_cliente_vehiculo_duplicados": int(raw.duplicated(["client_id", "vehicle_id"]).sum()),
        "vehiculos_en_>1_filas": int((per_vehicle > 1).sum()),
        "filas_con_vehiculo_repetido": int(raw["vehicle_id"].duplicated(keep=False).sum()),
        "clientes_con_>1_vehiculo": int((per_client > 1).sum()),
        "max_vehiculos_por_cliente": int(per_client.max()),
        "p90_vehiculos_por_cliente": float(per_client.quantile(0.9)),
        "clientes_flota_>=10_veh": int((per_client >= 10).sum()),
        "pct_filas_de_clientes_flota_>=10": round(per_client[per_client >= 10].sum() / n_rows * 100, 2),
    }
    for k, v in SUMMARY["overview"].items():
        print(f"{k:42s} {v}")

    # Clasificacion automatica nivel cliente vs nivel fila
    cand = [c for c in raw.columns if c not in ("Unnamed: 0", "client_id")]
    g = raw.groupby("client_id")[cand].nunique(dropna=False)
    share_const = (g <= 1).mean()
    level = pd.DataFrame({
        "pct_clientes_con_valor_unico": (share_const * 100).round(2),
        "nivel": np.where(share_const >= 0.999, "cliente", "fila (cliente-vehiculo)"),
    })
    save_tab(level, "00_nivel_variable")
    SUMMARY["columnas_nivel_cliente"] = level.index[level["nivel"] == "cliente"].tolist()
    print("\nColumnas a nivel de CLIENTE (constantes por client_id):")
    print(", ".join(SUMMARY["columnas_nivel_cliente"]))

    # Atributos de vehiculo coherentes entre filas del mismo vehiculo?
    veh_cols = ["model_series", "vehicle_type", "model_class", "engine_concept", "fuel_type",
                "date_of_first_registration", "product_group"]
    gv = raw.groupby("vehicle_id")[veh_cols].nunique(dropna=True)
    SUMMARY["vehiculos_con_atributos_contradictorios"] = {c: int((gv[c] > 1).sum()) for c in veh_cols}
    tel_conf = raw.groupby("vehicle_id")[["operating_hours_total", "trip_distance_average", "speed_average"]
                                          ].nunique(dropna=True)
    SUMMARY["vehiculos_con_telematica_distinta_entre_clientes"] = {c: int((tel_conf[c] > 1).sum()) for c in tel_conf}
    print("\nVehiculos con atributos contradictorios entre filas:", SUMMARY["vehiculos_con_atributos_contradictorios"])
    print("Vehiculos con telematica distinta segun cliente:", SUMMARY["vehiculos_con_telematica_distinta_entre_clientes"])

    # fig: estructura cliente-vehiculo
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    vc = per_client.clip(upper=30).value_counts().sort_index()
    ax[0].bar(vc.index, vc.values, color=BLUE)
    ax[0].set_yscale("log")
    ax[0].set_title("Filas (vehiculos) por cliente  [>=30 agrupado]")
    ax[0].set_xlabel("n filas por client_id"); ax[0].set_ylabel("n clientes (log)")
    vv = per_vehicle.clip(upper=10).value_counts().sort_index()
    ax[1].bar(vv.index, vv.values, color=GREEN)
    ax[1].set_yscale("log")
    ax[1].set_title("Filas (clientes) por vehiculo  [>=10 agrupado]")
    ax[1].set_xlabel("n filas por vehicle_id"); ax[1].set_ylabel("n vehiculos (log)")
    save_fig("01_estructura_cliente_vehiculo")
    return level


# --------------------------------------------------------------------------- #
# 2. Tipos, nulos, cardinalidad
# --------------------------------------------------------------------------- #
def group_of(col: str) -> str:
    if col in C.TELEMATICS_COLS:
        return "Telematica"
    if col in C.SOCIOECONOMIC_COLS or col == "zipcode":
        return "Socioeconomico/geo"
    if col in ("client_id", "vehicle_id", "Unnamed: 0"):
        return "Identificador/metadato"
    if col.startswith("consent_"):
        return "Consentimiento"
    if col in ("age_range", "gender_derived", "months_as_mb_customer", "currently_connected_vehicles",
               "finance_leasing_status", "role", "role_type", "country_code"):
        return "Cliente"
    return "Vehiculo"


def nulls_and_types(raw: pd.DataFrame) -> pd.DataFrame:
    section("2. TIPOS, NULOS Y CARDINALIDAD (datos crudos)")
    df = raw.drop(columns="Unnamed: 0")
    t = pd.DataFrame({
        "grupo": [group_of(c) for c in df.columns],
        "dtype": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "pct_nulos": (df.isna().mean() * 100).round(2),
        "n_unicos": df.nunique(),
        "ejemplo": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    save_tab(t, "02_tipos_nulos_cardinalidad")
    print(t.sort_values("pct_nulos", ascending=False).to_string())

    # fig barras de nulos
    s = t.sort_values("pct_nulos")
    palette = {"Telematica": RED, "Socioeconomico/geo": "#e08e0b", "Cliente": BLUE, "Vehiculo": GREEN,
               "Consentimiento": GREY, "Identificador/metadato": GREY}
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(s.index, s["pct_nulos"], color=[palette[g] for g in s["grupo"]])
    ax.set_xlabel("% de nulos"); ax.set_title("Valores nulos por columna (datos crudos)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), loc="lower right", fontsize=8)
    save_fig("02_nulos_por_columna")

    # fig mapa de nulos (muestra)
    samp = df.sample(2500, random_state=7)
    order = t.sort_values(["grupo", "pct_nulos"]).index
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(samp[order].isna().to_numpy().T, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel("filas (muestra aleatoria de 2.500)"); ax.grid(False)
    ax.set_title("Mapa de nulos (negro = nulo): se ve el bloque telematico que falta en bloque")
    save_fig("03_mapa_nulos")
    return t


# --------------------------------------------------------------------------- #
# 3. Patrones de ausencia (MCAR / MAR / estructural)
# --------------------------------------------------------------------------- #
def missing_patterns(raw: pd.DataFrame, cl: pd.DataFrame):
    section("3. PATRONES DE AUSENCIA")
    d = raw.copy()
    d["yr"] = pd.to_datetime(d["date_of_first_registration"], errors="coerce").dt.year
    d["yr_bin"] = pd.cut(d["yr"], [0, 2014, 2016, 2018, 2020, 2022, 2024, 2030],
                         labels=["<=2014", "2015-16", "2017-18", "2019-20", "2021-22", "2023-24", "2025-26"])
    d["has_tel"] = d[["trip_distance_average", "operating_hours_total", "fuel_consumption_average",
                      "speed_average"]].notna().any(axis=1)
    d["engine"] = d["engine_concept"].fillna("(nulo)")
    rows = []
    for col in ["engine", "product_group", "country_code", "yr_bin", "is_current_vehicle", "role"]:
        r = d.groupby(col, observed=True)["has_tel"].agg(["mean", "size"])
        for k, (m, n) in r.iterrows():
            rows.append({"variable": col, "valor": k, "pct_con_telematica": round(m * 100, 1), "n_filas": int(n)})
    tel = pd.DataFrame(rows)
    save_tab(tel, "03_telematica_disponible_por_segmento", index=False)
    print(tel.to_string(index=False))
    SUMMARY["pct_filas_con_telematica"] = round(d["has_tel"].mean() * 100, 2)

    # Nulos de engine/fuel por anyo de registro
    en = d.groupby("yr_bin", observed=True).agg(engine_nulo=("engine_concept", lambda s: s.isna().mean() * 100),
                                                 fuel_nulo=("fuel_type", lambda s: s.isna().mean() * 100))
    save_tab(en.round(1), "03_nulos_engine_fuel_por_anyo")
    print("\n% nulos engine_concept / fuel_type por periodo de registro:\n", en.round(1))

    # Nulos de gender por rol
    gen = d.groupby("role")["gender_derived"].apply(lambda s: s.isna().mean() * 100).round(1)
    print("\n% nulos gender por role:\n", gen)

    # Relacion engine_concept vs fuel_type
    ct = pd.crosstab(d["engine_concept"].fillna("(nulo)"), d["fuel_type"].fillna("(nulo)"))
    save_tab(ct, "03_crosstab_engine_vs_fuel")
    print("\nengine_concept x fuel_type:\n", ct.to_string())

    # Telematica en vehiculos electricos: consumo de combustible
    tcols = ["trip_distance_average", "trip_distance_all_counter", "operating_hours_total",
             "fuel_consumption_average", "speed_average", "speed_all_counter"]
    te = d.groupby("engine")[tcols].agg(lambda s: s.notna().mean() * 100).round(1)
    save_tab(te, "03_telematica_columnas_por_motorizacion")
    print("\n% de filas con cada variable telematica, por motorizacion:\n", te.to_string())
    e = d[d["engine_concept"] == "ELECTRIC"]
    SUMMARY["electricos"] = {
        "filas": len(e),
        "pct_con_telematica": round(float(e["has_tel"].mean() * 100), 2),
        "pct_no_nulo": {c: round(float(e[c].notna().mean() * 100), 2) for c in tcols},
    }
    print("\nElectricos:", SUMMARY["electricos"])

    # engine_concept nulo por clase de modelo (ausencia dependiente del modelo)
    ec = d.groupby(d["model_class"].fillna("(nulo)")).agg(
        n=("engine_concept", "size"), pct_engine_nulo=("engine_concept", lambda s: s.isna().mean() * 100)
    ).query("n >= 3000").sort_values("pct_engine_nulo").round(1)
    save_tab(ec, "03_engine_nulo_por_clase_modelo")
    print("\n% engine_concept nulo por model_class (n>=3000):\n", ec.to_string())

    # figura
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, col in zip(axes.ravel(), ["engine", "product_group", "country_code", "yr_bin",
                                      "is_current_vehicle", "role"]):
        r = tel[tel["variable"] == col]
        ax.bar(r["valor"].astype(str), r["pct_con_telematica"], color=BLUE)
        ax.axhline(SUMMARY["pct_filas_con_telematica"], color=RED, ls="--", lw=1)
        ax.set_title(f"% con telematica por {col}"); ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Disponibilidad de telematica por segmento (linea roja = media global)", y=1.01)
    save_fig("04_disponibilidad_telematica")


# --------------------------------------------------------------------------- #
# 4. Inconsistencias
# --------------------------------------------------------------------------- #
def inconsistencies(raw: pd.DataFrame, clean_log: dict, cl_raw: pd.DataFrame) -> pd.DataFrame:
    section("4. INCONSISTENCIAS")
    n = len(raw)
    d = raw.copy()
    d["d"] = pd.to_datetime(d["date_of_first_registration"], errors="coerce")
    ref = d["d"].max()
    age_h = (ref - d["d"]).dt.days * 24
    tel_cols = ["trip_distance_average", "operating_hours_total", "fuel_consumption_average", "speed_average"]
    has_tel = d[tel_cols].notna().any(axis=1)
    raw_zip = d["zipcode"].astype("string").str.strip()
    zip_bad = raw_zip.notna() & ~raw_zip.str.fullmatch(r"\d{4}").fillna(False)
    plug_no_electric = (d["engine_concept"] == "PLUGIN_HYBRID") & d["fuel_type"].notna() & \
        ~d["fuel_type"].fillna("").str.contains("ELECTRIC")
    comb_electric = (d["engine_concept"] == "COMBUSTION") & d["fuel_type"].fillna("").str.contains("ELECTRIC")
    elec_not_elec = (d["engine_concept"] == "ELECTRIC") & (d["fuel_type"] != "ELECTRIC")
    mode_cols = {c: d[c].astype("string").str.contains(",", na=False) for c in ("trip_distance_mode", "speed_mode")}
    months_c = cl_raw["months_as_mb_customer"]
    top_m = months_c.value_counts()
    n_cl = len(cl_raw)

    rows = [
        # (id, descripcion, n, base, tratamiento)
        ("Metadato", "Columna indice sin nombre ('Unnamed: 0') = 0..n-1", n, n,
         "Eliminar: es el indice de pandas exportado al CSV."),
        ("Duplicados", "Filas exactamente duplicadas", SUMMARY["overview"]["filas_duplicadas_exactas"], n,
         "Ninguna accion (0)."),
        ("Duplicados", "Pares (client_id, vehicle_id) repetidos", SUMMARY["overview"]["pares_cliente_vehiculo_duplicados"], n,
         "Ninguna accion (0): la clave de la fila es (cliente, vehiculo)."),
        ("Estructura", "Filas cuyo vehicle_id aparece en >1 fila (vehiculo con varios clientes/roles)",
         SUMMARY["overview"]["filas_con_vehiculo_repetido"], n,
         "NO deduplicar: es una relacion N:M real (OWNER/USER/LEGACY...). Contar vehiculos unicos "
         "con drop_duplicates('vehicle_id') cuando se analice el parque."),
        ("Estructura", "Filas de clientes-flota (>=10 filas por client_id)",
         int(raw.groupby("client_id")["vehicle_id"].transform("size").ge(10).sum()), n,
         "Reportar estadisticos tambien a nivel cliente/vehiculo unico; evaluar excluir/ponderar flotas."),
        ("Formato", "zipcode con formato invalido (prefijo 'A-'/'CH-', texto, >4 digitos)", int(zip_bad.sum()), n,
         "Normalizar prefijos; resto -> NaN. (zipcode se excluye como variable de todos modos)."),
        ("Formato", "trip_distance_mode multimodal (lista de bins en texto)", int(mode_cols["trip_distance_mode"].sum()), n,
         "Parsear a numerico; lista -> NaN + flag *_multimodal; columna excluida del analisis."),
        ("Formato", "speed_mode multimodal (lista de bins en texto)", int(mode_cols["speed_mode"].sum()), n,
         "Idem."),
        ("Coherencia", "Registros con zipcode presente pero SIN datos socioeconomicos (cp no cruzado)",
         int((d["zipcode"].notna() & d["population"].isna()).sum()), n,
         "Imputar mediana por pais + flag socio_missing."),
        ("Coherencia", "population = 0 con clientes residentes", int((d["population"] == 0).sum()), n,
         "Tratar como NaN y reimputar (mediana por pais)."),
        ("Coherencia", "ELECTRIC con fuel_type distinto de ELECTRIC", int(elec_not_elec.sum()), n, "Ninguna (0)."),
        ("Coherencia", "COMBUSTION con fuel_type que incluye ELECTRIC", int(comb_electric.sum()), n, "Ninguna (0)."),
        ("Coherencia", "PLUGIN_HYBRID con fuel_type sin componente ELECTRIC", int(plug_no_electric.sum()), n, "Ninguna (0)."),
        ("Coherencia", "operating_hours_total > horas desde el primer registro", int((d["operating_hours_total"] > age_h).sum()), n,
         "Ninguna (0): coherente."),
        ("Coherencia", "Fecha de primer registro posterior a la fecha de referencia", int((d["d"] > ref).sum()), n, "Ninguna (0)."),
        ("Plausibilidad", "speed_average > 150 km/h (velocidad MEDIA)", int((d["speed_average"] > 150).sum()), int(d["speed_average"].notna().sum()),
         "Convertir a NaN (valor fisicamente imposible como media)."),
        ("Plausibilidad", "fuel_consumption_average > 30 l/100km", int((d["fuel_consumption_average"] > 30).sum()),
         int(d["fuel_consumption_average"].notna().sum()), "Convertir a NaN."),
        ("Plausibilidad", "trip_distance_average > 100 km (viaje medio)", int((d["trip_distance_average"] > 100).sum()),
         int(d["trip_distance_average"].notna().sum()), "Mantener (posibles furgonetas/larga distancia); winsorizar si se modela."),
        ("Plausibilidad", "speed_average < 5 km/h", int((d["speed_average"] < 5).sum()), int(d["speed_average"].notna().sum()),
         "Mantener; vehiculos casi sin uso (mediana 0,14 h): excluir al estimar habitos de conduccion."),
        ("Plausibilidad", "Telematica presente con trip_distance_all_counter = 0 (sin viajes)",
         int((d["trip_distance_all_counter"] == 0).sum()), int(d["trip_distance_all_counter"].notna().sum()),
         "Mantener; son vehiculos conectados sin uso. Excluir al estimar medias de conduccion."),
        ("Plausibilidad", "operating_hours_total = 0", int((d["operating_hours_total"] == 0).sum()), int(d["operating_hours_total"].notna().sum()),
         "Idem."),
        ("Heaping", f"months_as_mb_customer: valor 36 (a nivel cliente)", int(top_m.get(36.0, 0)), n_cl,
         "Pico anomalo (12% de clientes con exactamente 36 meses): posible valor por defecto/migracion. "
         "No se modifica; flag recomendado y cautela al interpretar antiguedad."),
        ("Heaping", f"months_as_mb_customer: valor 0 (a nivel cliente)", int(top_m.get(0.0, 0)), n_cl,
         "Mezcla de clientes realmente nuevos y valores por defecto; no se modifica."),
        ("Coherencia", "is_current_vehicle=False pero con telematica", int((~d["is_current_vehicle"] & has_tel).sum()),
         int((~d["is_current_vehicle"]).sum()), "Plausible (historico de uso); mantener."),
        ("Coherencia", "currently_connected_vehicles = 0 pero la fila tiene telematica", int(((d["currently_connected_vehicles"] == 0) & has_tel).sum()),
         int((d["currently_connected_vehicles"] == 0).sum()),
         "Atributo de cliente vs. dato de vehiculo: definiciones distintas (no es error), interpretar con cuidado."),
        ("Cobertura", "model_class nulo (serie sin clase asignada en ningun registro)", int(d["model_class"].isna().sum()), n,
         "No se puede inferir via model_series (esas series nunca tienen clase): categoria 'UNKNOWN'."),
        ("Cobertura", "Electricos con operating_hours/speed/fuel_consumption (todas deberian ser nulas si el esquema es distinto)",
         int(d.loc[d["engine_concept"] == "ELECTRIC", ["operating_hours_total", "speed_average", "fuel_consumption_average"]].notna().any(axis=1).sum()),
         int((d["engine_concept"] == "ELECTRIC").sum()),
         "0 filas: los electricos solo reportan distancia de viaje; velocidad, horas y consumo son NaN "
         "estructural (esquema telematico distinto). No imputar; analizar por separado."),
    ]
    inc = pd.DataFrame(rows, columns=["tipo", "comprobacion", "n_afectados", "base", "tratamiento"])
    inc["pct"] = (inc["n_afectados"] / inc["base"] * 100).round(2)
    save_tab(inc[["tipo", "comprobacion", "n_afectados", "base", "pct", "tratamiento"]], "04_inconsistencias", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(inc[["tipo", "comprobacion", "n_afectados", "pct"]].to_string(index=False))
    return inc


# --------------------------------------------------------------------------- #
# 5. Exclusion de columnas
# --------------------------------------------------------------------------- #
def exclusions(raw: pd.DataFrame):
    section("5. COLUMNAS EXCLUIDAS DEL ANALISIS")
    limpio = C.clean(raw)  # las derivadas de telemetria solo existen tras limpiar
    col = lambda c: raw[c] if c in raw else limpio[c]
    rows = [(c, cat, why, col(c).nunique(), round(col(c).isna().mean() * 100, 2))
            for c, (cat, why) in C.EXCLUDED_COLUMNS.items()]
    ex = pd.DataFrame(rows, columns=["columna", "categoria", "motivo", "n_unicos", "pct_nulos"])
    save_tab(ex, "05_columnas_excluidas", index=False)
    pd.set_option("display.max_colwidth", 120)
    print(ex[["columna", "categoria", "n_unicos"]].to_string(index=False))

    # evidencia numerica de redundancia
    ev = {}
    ev["millones/(per_inhab*pop) ratio min,max"] = [
        float((raw["purchasing_power_millions_eur"] * 1e6 / (raw["purchasing_power_per_inhabitant_eur"] * raw["population"])).min()),
        float((raw["purchasing_power_millions_eur"] * 1e6 / (raw["purchasing_power_per_inhabitant_eur"] * raw["population"])).max())]
    sp = raw[["purchasing_power_per_inhabitant_eur", "purchasing_power_per_household_eur", "population",
              "number_of_households", "purchasing_power_millions_eur"]].corr(method="spearman")
    ev["spearman_household_vs_inhabitant"] = float(sp.iloc[0, 1])
    ev["spearman_population_vs_households"] = float(sp.loc["population", "number_of_households"])
    ev["role_vs_role_type"] = pd.crosstab(raw["role"], raw["role_type"]).to_dict()
    ser_cls = raw.dropna(subset=["model_class"]).groupby("model_series")["model_class"].nunique()
    ev["series_con_una_unica_clase"] = f"{(ser_cls == 1).sum()} de {raw['model_series'].nunique()}"
    mode_eq = raw[["speed_mode", "speed_median"]].dropna()
    ev["speed_mode == speed_median (%)"] = round(
        (pd.to_numeric(mode_eq["speed_mode"], errors="coerce") == mode_eq["speed_median"]).mean() * 100, 1)
    SUMMARY["evidencia_exclusion"] = ev
    print("\nEvidencia:", json.dumps(ev, indent=2, default=str))

    # heatmap de correlacion para justificar
    cols = ["purchasing_power_per_inhabitant_eur", "purchasing_power_per_household_eur", "population",
            "number_of_households", "purchasing_power_millions_eur"]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(sp, cmap="RdBu_r", vmin=-1, vmax=1)
    short = ["pp_habitante", "pp_hogar", "poblacion", "n_hogares", "pp_millones"]
    ax.set_xticks(range(5)); ax.set_xticklabels(short, rotation=40, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(short); ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{sp.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(sp.iloc[i, j]) > .6 else "black")
    plt.colorbar(im, fraction=0.046)
    ax.set_title("Spearman bloque socioeconomico (redundancias)")
    save_fig("05_corr_socioeconomico")


# --------------------------------------------------------------------------- #
# 6. Estadisticos descriptivos
# --------------------------------------------------------------------------- #
def describe_numeric(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows.append({
            "variable": c, "n": len(s), "pct_nulos": round(df[c].isna().mean() * 100, 2),
            "media": s.mean(), "std": s.std(), "min": s.min(), "p1": s.quantile(.01), "p25": q1,
            "mediana": s.median(), "p75": q3, "p99": s.quantile(.99), "max": s.max(),
            "asimetria": s.skew(), "curtosis": s.kurt(),
            "outliers_IQR_n": int(out), "outliers_IQR_pct": round(out / len(s) * 100, 2),
        })
    t = pd.DataFrame(rows).set_index("variable")
    save_tab(t.round(3), name)
    return t


def describe_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        pct = vc / len(df) * 100
        top = ", ".join(f"{k}: {p:.1f}%" for k, p in pct.head(4).items())
        rows.append({"variable": c, "n_niveles": df[c].nunique(dropna=True), "pct_nulos": round(df[c].isna().mean() * 100, 2),
                     "moda": vc.index[0], "pct_moda": round(pct.iloc[0], 2), "top4": top})
    t = pd.DataFrame(rows).set_index("variable")
    return t


def descriptives(raw: pd.DataFrame, clean_df: pd.DataFrame, cl_clean: pd.DataFrame, level: pd.DataFrame):
    section("6. ESTADISTICOS DESCRIPTIVOS")
    num_raw = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and c not in ("Unnamed: 0",)
               and raw[c].nunique() > 2]
    t_raw = describe_numeric(raw, num_raw, "06_descriptivos_numericos_crudos")
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    pd.set_option("display.width", 250)
    print("\n--- Numericas (crudo, nivel fila) ---")
    print(t_raw.to_string())

    feats = C.analytic_features(clean_df)
    client_cols = set(level.index[level["nivel"] == "cliente"]) | {"personas_por_hogar", "log_population"}
    num = [c for c in feats if pd.api.types.is_numeric_dtype(clean_df[c]) and clean_df[c].dtype != bool and
           clean_df[c].nunique() > 2]
    cat = [c for c in feats if c not in num]
    SUMMARY["variables_analiticas"] = {"numericas": num, "categoricas": cat,
                                       "excluidas": list(C.EXCLUDED_COLUMNS)}

    # numericas limpias: a nivel cliente para las de cliente
    base_row = clean_df
    parts = []
    for c in num:
        src = cl_clean if c in client_cols else base_row
        t = describe_numeric(src, [c], "_tmp")
        t["nivel"] = "cliente (n=%d)" % len(cl_clean) if c in client_cols else "fila (n=%d)" % len(base_row)
        parts.append(t)
    t_clean = pd.concat(parts)
    (TAB / "_tmp.csv").unlink(missing_ok=True)
    save_tab(t_clean.round(3), "06_descriptivos_numericos_limpios")
    print("\n--- Numericas (limpio, variables analiticas; cliente o fila segun nivel) ---")
    print(t_clean[["nivel", "n", "pct_nulos", "media", "std", "min", "mediana", "max", "asimetria", "outliers_IQR_pct"]].to_string())

    # categoricas
    parts = []
    for c in cat:
        src = cl_clean if c in client_cols else base_row
        t = describe_categorical(src, [c])
        t["nivel"] = "cliente" if c in client_cols else "fila"
        parts.append(t)
    t_cat = pd.concat(parts)
    save_tab(t_cat, "06_descriptivos_categoricos")
    print("\n--- Categoricas (limpio) ---")
    print(t_cat.to_string())
    return num, cat, client_cols


# --------------------------------------------------------------------------- #
# 7. Distribuciones
# --------------------------------------------------------------------------- #
def distributions(clean_df, cl_clean, num, cat, client_cols):
    section("7. DISTRIBUCIONES (figuras)")
    # --- numericas ---
    n = len(num)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
    for ax, c in zip(axes.ravel(), num):
        src = cl_clean if c in client_cols else clean_df
        lvl = "clientes" if c in client_cols else "filas"
        if c in ("purchasing_power_per_inhabitant_eur", "population"):
            # se excluyen los valores imputados para no mostrar un pico artificial
            src = src[~src["socio_missing"]]
            lvl += ", sin imputados"
        s = src[c].dropna()
        if s.nunique() <= 25:
            vc = s.clip(upper=s.quantile(.999)).value_counts().sort_index() if s.nunique() > 15 else s.value_counts().sort_index()
            ax.bar([f"{v:g}" for v in vc.index], vc.values, color=BLUE)
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.set_title(f"{c}  ({lvl})", fontsize=9)
            continue
        use_log = s.min() >= 0 and s.skew() > 3
        x = np.log10(1 + s) if use_log else s
        lo, hi = x.quantile([.001, .999])
        ax.hist(x.clip(lo, hi), bins=60, color=BLUE, alpha=.9)
        ax.axvline(x.median(), color=RED, ls="--", lw=1)
        ax.set_title(f"{c}  ({lvl}){'  [log10(1+x)]' if use_log else ''}", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Distribuciones de variables numericas analiticas (linea roja = mediana; "
                 "recorte p0.1-p99.9; log si asimetria>3)", y=1.0)
    save_fig("06_distribuciones_numericas")

    # --- categoricas ---
    cats = [c for c in cat if c not in ("model_class",)] + ["model_class"]
    cats = [c for c in cats if c in clean_df.columns]
    ncol = 3
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.4 * nrow))
    for ax, c in zip(axes.ravel(), cats):
        src = cl_clean if c in client_cols else clean_df
        vc = src[c].astype(str).value_counts(normalize=True).head(15) * 100
        vc = vc.iloc[::-1]
        ax.barh(vc.index, vc.values, color=[RED if i == "UNKNOWN" else BLUE for i in vc.index])
        ax.set_title(f"{c}  ({'clientes' if c in client_cols else 'filas'}, top 15)", fontsize=9)
        ax.set_xlabel("%")
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    save_fig("07_distribuciones_categoricas")

    # --- anyo de registro ---
    yr = clean_df["date_of_first_registration"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(yr.index, yr.values, color=BLUE)
    ax.set_title("Vehiculos por anyo de primer registro (nulos: %d)" % clean_df["date_of_first_registration"].isna().sum())
    ax.set_xlabel("anyo")
    save_fig("08_anyo_registro")

    # --- heaping months ---
    mo = cl_clean["months_as_mb_customer"].round().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(mo.index, mo.values, color=[RED if i in (0, 36) else BLUE for i in mo.index])
    ax.set_title("months_as_mb_customer (clientes): picos anomalos en 0 y 36 meses")
    ax.set_xlabel("meses como cliente")
    save_fig("09_meses_cliente_heaping")

    # --- boxplots telematica por motorizacion ---
    tel = ["trip_distance_average", "speed_average", "fuel_consumption_average", "operating_hours_total"]
    order = ["COMBUSTION", "MILD_HYBRID", "PLUGIN_HYBRID", "ELECTRIC", "UNKNOWN"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, c in zip(axes, tel):
        data = [clean_df.loc[clean_df["engine_concept"] == e, c].dropna() for e in order]
        ax.boxplot(data, tick_labels=[f"{e}\n(n={len(x)})" for e, x in zip(order, data)], showfliers=False)
        ax.set_title(c, fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
    fig.suptitle("Telematica por motorizacion (sin outliers visibles; ELECTRIC solo reporta distancia de viaje)", y=1.02)
    save_fig("10_telematica_por_motorizacion")

    # --- correlacion variables analiticas ---
    cnum = [c for c in num if c in clean_df.columns]
    corr = clean_df[cnum].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cnum))); ax.set_xticklabels(cnum, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cnum))); ax.set_yticklabels(cnum, fontsize=7); ax.grid(False)
    for i in range(len(cnum)):
        for j in range(len(cnum)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > .6 else "black")
    plt.colorbar(im, fraction=0.04)
    ax.set_title("Correlacion de Spearman (variables analiticas, por pares)")
    save_fig("11_correlacion_analiticas")
    save_tab(corr.round(3), "07_correlacion_spearman")

    # --- categoricas clave vs telematica/talleres ---
    pairs = clean_df.groupby("model_class").agg(n=("vehicle_id", "size"),
                                                 fuel_cons=("fuel_consumption_average", "median"),
                                                 speed=("speed_average", "median"),
                                                 age=("vehicle_age_years", "median")).query("n>=1500").sort_values("n")
    save_tab(pairs.round(2), "07_clase_modelo_resumen")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(path: str | None = None):
    sys.stdout.reconfigure(encoding="utf-8")
    raw = C.load_raw(path)
    print(f"CSV: {C.resolve_data_path(path)}  ->  {raw.shape[0]:,} filas x {raw.shape[1]} columnas")
    cl_raw = raw.drop_duplicates("client_id")
    level = overview(raw)
    nulls_and_types(raw)

    clog: dict = {}
    clean_df = C.clean(raw, clog)
    cl_clean = clean_df.drop_duplicates("client_id")
    SUMMARY["limpieza"] = clog

    missing_patterns(raw, cl_clean)
    inconsistencies(raw, clog, cl_raw)
    exclusions(raw)
    num, cat, client_cols = descriptives(raw, clean_df, cl_clean, level)
    distributions(clean_df, cl_clean, num, cat, client_cols)

    section("8. LOG DE LIMPIEZA")
    for k, v in clog.items():
        print(f"{k:45s} {v}")

    out_cols = clean_df.columns.tolist()
    clean_df[out_cols].to_parquet(OUT / "dataset_limpio.parquet", index=False)
    sin_nulos = C.drop_null_columns(clean_df)
    sin_nulos.to_parquet(OUT / "dataset_sin_nulos.parquet", index=False)
    sin_nulos.to_csv(OUT / "dataset_sin_nulos.csv", index=False, encoding="utf-8-sig")
    print(f"dataset_sin_nulos: {sin_nulos.shape[1]} columnas (eliminadas {clean_df.shape[1] - sin_nulos.shape[1]} con NaN)")
    clean_df[C.analytic_features(clean_df)].head(0)  # valida que las columnas existen
    (OUT / "resumen.json").write_text(json.dumps(SUMMARY, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nListo. Resultados en {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
