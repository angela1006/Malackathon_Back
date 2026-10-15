"""Genera y EJECUTA el notebook del EDA: eda/EDA_Mercedes_Benz.ipynb

    python -m eda.build_notebook            # genera + ejecuta (deja las salidas guardadas)
    python -m eda.build_notebook --no-exec  # solo genera

Los cells estan definidos aqui para poder regenerar el notebook; el entregable es el
.ipynb ya ejecutado (con tablas y graficos incrustados).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "EDA_Mercedes_Benz.ipynb"

cells: list = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =========================================================================== #
md("""
# EDA – Reto Mercedes-Benz (Malackathon)

**Objetivo:** análisis exploratorio completo del dataset de clientes y vehículos: estadísticos descriptivos,
valores nulos e inconsistencias (con tratamiento justificado), distribuciones de variables clave y exclusión
justificada de columnas de metadatos/identificadores.

**Índice**
1. Carga y estructura (¿qué es una fila?)
2. Tipos, nulos y cardinalidad
3. Patrones de ausencia
4. Inconsistencias y calidad
5. Columnas excluidas
6. Limpieza propuesta (módulo `eda.cleaning`)
7. Estadísticos descriptivos
8. Distribuciones
9. Perfil del cliente
10. **Resumen de hallazgos y conclusiones**
""")

code("""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# permite importar el paquete `eda` tanto desde Backend/ como desde Backend/eda/
ROOT = Path.cwd() if (Path.cwd() / "eda").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from eda import cleaning as C

warnings.filterwarnings("ignore")
%matplotlib inline
pd.set_option("display.max_columns", None); pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
BLUE, RED, GREEN, GREY = "#1f4e79", "#c0392b", "#2e8b57", "#9aa5b1"

raw = C.load_raw()
print(f"{C.resolve_data_path()}\\n{raw.shape[0]:,} filas x {raw.shape[1]} columnas")
raw.head(3).T
""")

# =========================================================================== #
md("""
## 1. Estructura: ¿qué representa una fila?

Hay un `client_id` y un `vehicle_id` (hashes). Antes de calcular nada hay que saber cuál es la **unidad de análisis**,
porque si un cliente aparece en varias filas, las medias "por fila" sobre-representan a los clientes con muchos vehículos.
""")

code("""
per_client = raw.groupby("client_id").size()
per_vehicle = raw.groupby("vehicle_id").size()
n = len(raw)
estructura = pd.Series({
    "filas": n,
    "clientes únicos": raw.client_id.nunique(),
    "vehículos únicos": raw.vehicle_id.nunique(),
    "filas duplicadas exactas": raw.drop(columns="Unnamed: 0").duplicated().sum(),
    "pares (cliente, vehículo) repetidos": raw.duplicated(["client_id", "vehicle_id"]).sum(),
    "clientes con >1 vehículo": (per_client > 1).sum(),
    "vehículos en >1 fila (varios clientes/roles)": (per_vehicle > 1).sum(),
    "clientes-flota (>=10 filas)": (per_client >= 10).sum(),
    "% de filas que aportan las flotas": per_client[per_client >= 10].sum() / n * 100,
    "máx. filas de un cliente": per_client.max(),
}, name="valor")
display(estructura.to_frame())

fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
vc = per_client.clip(upper=30).value_counts().sort_index()
ax[0].bar(vc.index, vc.values, color=BLUE); ax[0].set_yscale("log")
ax[0].set_title("Filas por cliente [>=30 agrupado]"); ax[0].set_xlabel("n filas"); ax[0].set_ylabel("clientes (log)")
vv = per_vehicle.clip(upper=10).value_counts().sort_index()
ax[1].bar(vv.index, vv.values, color=GREEN); ax[1].set_yscale("log")
ax[1].set_title("Filas por vehículo [>=10 agrupado]"); ax[1].set_xlabel("n filas"); ax[1].set_ylabel("vehículos (log)")
plt.tight_layout(); plt.show()
""")

md("""
**Lectura:** la fila es una **relación cliente–vehículo** (la combinación `client_id + vehicle_id` nunca se repite).
Un vehículo puede tener varios clientes/roles (propietario actual, anterior, usuario...), por lo que **no se debe deduplicar**
por `vehicle_id`. Además, 1.881 clientes-flota (≥10 filas) generan ~21 % de las filas.

Ahora detectamos automáticamente qué columnas son atributos del **cliente** (constantes por `client_id`) y cuáles de la **fila/vehículo**,
y comprobamos la coherencia de los atributos de vehículo entre filas.
""")

code("""
cand = [c for c in raw.columns if c not in ("Unnamed: 0", "client_id")]
share_const = (raw.groupby("client_id")[cand].nunique(dropna=False) <= 1).mean()
nivel = pd.DataFrame({"% clientes con valor único": share_const * 100,
                      "nivel": np.where(share_const >= .999, "CLIENTE", "fila (cliente-vehículo)")})
CLIENT_COLS = nivel.index[nivel.nivel == "CLIENTE"].tolist()
print("Atributos a nivel de CLIENTE:", ", ".join(CLIENT_COLS))

veh_cols = ["model_series", "vehicle_type", "model_class", "engine_concept", "fuel_type",
            "date_of_first_registration", "product_group"]
contrad = (raw.groupby("vehicle_id")[veh_cols].nunique() > 1).sum()
print("\\nVehículos con atributos contradictorios entre sus filas:\\n", contrad.to_dict())

tel_dist = (raw.groupby("vehicle_id")[["operating_hours_total", "trip_distance_average", "speed_average"]].nunique() > 1).sum()
print("\\nVehículos cuya telemetría difiere según el cliente:", tel_dist.to_dict())
""")

md("""
**Lectura:** los atributos del vehículo son coherentes entre filas (0 contradicciones), pero la **telemetría se mide por relación
cliente–vehículo** (≈9.000 vehículos tienen valores distintos según el cliente). Por eso, de aquí en adelante:
las variables de cliente se describen **a nivel cliente** y las de vehículo/telemetría **a nivel fila**.
""")

# =========================================================================== #
md("""
## 2. Tipos, nulos y cardinalidad (datos crudos)
""")

code("""
def grupo(c):
    if c in C.TELEMATICS_COLS: return "Telemática"
    if c in C.SOCIOECONOMIC_COLS or c == "zipcode": return "Socioeconómico/geo"
    if c in ("client_id", "vehicle_id", "Unnamed: 0"): return "Identificador/metadato"
    if c.startswith("consent_"): return "Consentimiento"
    if c in CLIENT_COLS or c in ("role", "role_type"): return "Cliente"
    return "Vehículo"

df0 = raw.drop(columns="Unnamed: 0")
tipos = pd.DataFrame({
    "grupo": [grupo(c) for c in df0.columns],
    "dtype": df0.dtypes.astype(str),
    "nulos": df0.isna().sum(),
    "% nulos": df0.isna().mean() * 100,
    "n únicos": df0.nunique(),
}).sort_values("% nulos", ascending=False)
display(tipos)

pal = {"Telemática": RED, "Socioeconómico/geo": "#e08e0b", "Cliente": BLUE, "Vehículo": GREEN,
       "Consentimiento": GREY, "Identificador/metadato": GREY}
s = tipos.sort_values("% nulos")
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(s.index, s["% nulos"], color=[pal[g] for g in s.grupo])
ax.set_xlabel("% nulos"); ax.set_title("Valores nulos por columna")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()], pal.keys(), loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()
""")

code("""
# Mapa de nulos sobre una muestra: se aprecia que la telemetría falta EN BLOQUE
order = tipos.sort_values(["grupo", "% nulos"]).index
muestra = df0.sample(2500, random_state=7)
fig, ax = plt.subplots(figsize=(11, 5))
ax.imshow(muestra[order].isna().to_numpy().T, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6); ax.grid(False)
ax.set_xlabel("filas (muestra de 2.500)"); ax.set_title("Mapa de nulos (negro = nulo)")
plt.tight_layout(); plt.show()
""")

md("""
**Lectura:** hay cuatro bloques de nulos con comportamientos distintos:
* **Telemetría** (10 columnas, 57–61 % nulos): falta en bloque → ausencia estructural.
* **`engine_concept` (44 %) y `fuel_type` (59 %)**: muy altos; los analizamos en el siguiente apartado.
* **Socioeconómico/`zipcode`** (5–6 %): códigos postales sin cruzar con la tabla externa.
* **`gender_derived` (11 %), `date_of_first_registration` (1,4 %), `model_class` (1,1 %), `vehicle_type` (0,3 %)**: menores.
""")

# =========================================================================== #
md("""
## 3. Patrones de ausencia (¿aleatoria o estructural?)

La estrategia de tratamiento depende del mecanismo: si es aleatoria (MCAR) se puede imputar; si depende de otras variables (MAR)
o es estructural, imputar con medias introduce sesgo. Medimos la disponibilidad de telemetría por segmentos.
""")

code("""
d = raw.copy()
d["yr"] = pd.to_datetime(d.date_of_first_registration, errors="coerce").dt.year
d["periodo"] = pd.cut(d.yr, [0, 2014, 2016, 2018, 2020, 2022, 2024, 2030],
                      labels=["<=2014", "2015-16", "2017-18", "2019-20", "2021-22", "2023-24", "2025-26"])
d["motor"] = d.engine_concept.fillna("(nulo)")
core = ["trip_distance_average", "operating_hours_total", "fuel_consumption_average", "speed_average"]
d["con_telematica"] = d[core].notna().any(axis=1)
media = d.con_telematica.mean() * 100
print(f"Filas con telemetría: {media:.1f} %")

segs = ["role", "is_current_vehicle", "periodo", "motor", "product_group", "country_code"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, c in zip(axes.ravel(), segs):
    r = d.groupby(c, observed=True).con_telematica.mean() * 100
    ax.bar(r.index.astype(str), r.values, color=BLUE); ax.axhline(media, color=RED, ls="--", lw=1)
    ax.set_title(f"% con telemetría por {c}"); ax.set_ylim(0, 100); ax.tick_params(axis="x", rotation=40)
fig.suptitle("Disponibilidad de telemetría (línea roja = media)", y=1.01)
plt.tight_layout(); plt.show()
""")

code("""
tcols = ["trip_distance_average", "trip_distance_all_counter", "operating_hours_total",
         "fuel_consumption_average", "speed_average", "speed_all_counter"]
print("% de filas con cada variable de telemetría, por tipo de motor:")
display(d.groupby("motor")[tcols].agg(lambda s: s.notna().mean() * 100))
""")

md("""
**Lectura (telemetría):**
* No es MCAR: depende del **rol** (OWNER 50 %, USER 54 %, LEGACY 13 %, DRIVER 10 %, TEMP_USER 3 %), de si el vehículo es **actual**
  (58 % vs 33 %) y del **año de registro**. El país no influye.
* **Eléctricos:** solo reportan `trip_distance_*`; horas, velocidad y consumo son 100 % nulos (esquema de datos distinto).
* **Grupo de producto F:** 0 % de telemetría.

**Decisión:** *no imputar* la telemetría; crear el indicador `has_telematics` y analizar solo las filas con dato.

Veamos ahora el patrón de `engine_concept` / `fuel_type`:
""")

code("""
ec = d.groupby(d.model_class.fillna("(nulo)")).agg(n=("engine_concept", "size"),
        pct_engine_nulo=("engine_concept", lambda s: s.isna().mean() * 100)).query("n >= 3000").sort_values("pct_engine_nulo")
display(ec)
print("\\n% nulos de engine_concept por país:", (d.groupby("country_code").engine_concept.apply(lambda s: s.isna().mean()*100)).round(1).to_dict())
print("% nulos de engine_concept por is_current_vehicle:", (d.groupby("is_current_vehicle").engine_concept.apply(lambda s: s.isna().mean()*100)).round(1).to_dict())
print("\\nengine_concept x fuel_type")
display(pd.crosstab(d.engine_concept.fillna("(nulo)"), d.fuel_type.fillna("(nulo)")))
""")

md("""
**Lectura (motor/combustible):** los nulos dependen del **modelo** (Clase B y Citan 100 % nulos, GLA y EQC 0 %), no del país ni de si el
vehículo es actual. Cuando hay `fuel_type` siempre hay `engine_concept`, y ambos son coherentes (ELECTRIC ⇔ ELECTRIC, plug-in ⇔ componente eléctrico).

**Decisión:** conservar como categoría explícita `UNKNOWN` (la ausencia es informativa; imputar por moda inventaría información).
""")

# =========================================================================== #
md("""
## 4. Inconsistencias y problemas de calidad

Batería de comprobaciones: duplicados, formato, coherencia entre columnas, plausibilidad física y *heaping* (acumulación anómala de valores).
""")

code("""
d["fecha"] = pd.to_datetime(d.date_of_first_registration, errors="coerce")
ref = d.fecha.max()
horas_vida = (ref - d.fecha).dt.days * 24
zip_raw = d.zipcode.astype("string").str.strip()
zip_mal = zip_raw.notna() & ~zip_raw.str.fullmatch(r"\\d{4}").fillna(False)
mode_multi = lambda c: d[c].astype("string").str.contains(",", na=False)
clientes = raw.drop_duplicates("client_id")
m36 = (clientes.months_as_mb_customer == 36).sum(); m0 = (clientes.months_as_mb_customer == 0).sum()
tiene_tel = d.con_telematica

checks = [
 # (tipo, comprobación, afectados, base)
 ("Duplicados", "Filas exactamente duplicadas", int(raw.drop(columns="Unnamed: 0").duplicated().sum()), n),
 ("Duplicados", "Pares (cliente, vehículo) repetidos", int(raw.duplicated(["client_id", "vehicle_id"]).sum()), n),
 ("Estructura", "Filas de clientes-flota (>=10 filas)", int(d.groupby("client_id").vehicle_id.transform("size").ge(10).sum()), n),
 ("Formato", "zipcode con formato inválido ('A-2100', 'CH-6340', 'Ottenbach', '55605')", int(zip_mal.sum()), n),
 ("Formato", "trip_distance_mode multimodal (lista de bins como texto)", int(mode_multi("trip_distance_mode").sum()), n),
 ("Formato", "speed_mode multimodal (lista de bins como texto)", int(mode_multi("speed_mode").sum()), n),
 ("Cobertura", "zipcode presente pero sin datos socioeconómicos", int((d.zipcode.notna() & d.population.isna()).sum()), n),
 ("Coherencia", "population = 0 con clientes residentes", int((d.population == 0).sum()), n),
 ("Coherencia", "ELECTRIC con fuel_type != ELECTRIC", int(((d.engine_concept == "ELECTRIC") & (d.fuel_type != "ELECTRIC")).sum()), n),
 ("Coherencia", "PLUGIN_HYBRID sin componente eléctrico", int(((d.engine_concept == "PLUGIN_HYBRID") & d.fuel_type.notna() & ~d.fuel_type.fillna("").str.contains("ELECTRIC")).sum()), n),
 ("Coherencia", "horas de uso > horas desde el primer registro", int((d.operating_hours_total > horas_vida).sum()), n),
 ("Coherencia", "fecha de registro posterior a la fecha de referencia", int((d.fecha > ref).sum()), n),
 ("Plausibilidad", "speed_average > 150 km/h (como MEDIA)", int((d.speed_average > 150).sum()), int(d.speed_average.notna().sum())),
 ("Plausibilidad", "fuel_consumption_average > 30 l/100km", int((d.fuel_consumption_average > 30).sum()), int(d.fuel_consumption_average.notna().sum())),
 ("Plausibilidad", "trip_distance_average > 100 km", int((d.trip_distance_average > 100).sum()), int(d.trip_distance_average.notna().sum())),
 ("Plausibilidad", "speed_average < 5 km/h", int((d.speed_average < 5).sum()), int(d.speed_average.notna().sum())),
 ("Plausibilidad", "telemetría con 0 viajes (trip_distance_all_counter = 0)", int((d.trip_distance_all_counter == 0).sum()), int(d.trip_distance_all_counter.notna().sum())),
 ("Heaping", "months_as_mb_customer = 36 (clientes)", int(m36), len(clientes)),
 ("Heaping", "months_as_mb_customer = 0 (clientes)", int(m0), len(clientes)),
]
incons = pd.DataFrame(checks, columns=["tipo", "comprobación", "afectados", "base"])
incons["%"] = incons.afectados / incons.base * 100
pd.set_option("display.max_colwidth", 90)
display(incons)
""")

code("""
# Heaping de antigüedad: se ve un pico muy superior a los meses vecinos
mo = clientes.months_as_mb_customer.round().value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(mo.index, mo.values, color=[RED if i in (0, 36) else BLUE for i in mo.index])
ax.set_title("months_as_mb_customer (clientes): picos anómalos en 0 y 36"); ax.set_xlabel("meses")
plt.tight_layout(); plt.show()
""")

md("""
**Lectura y tratamiento propuesto:**

| Problema | Tratamiento | Justificación |
|---|---|---|
| Sin duplicados ni contradicciones | — | Calidad estructural buena |
| `vehicle_id` repetido | No deduplicar | Relación N:M real |
| Flotas (20,6 % de filas) | Reportar a nivel cliente / sensibilidad con y sin flotas | Sesgan las medias |
| `zipcode` mal formado (5) | Quitar prefijo de país; el resto → NaN | AT/CH usan 4 dígitos |
| `population = 0` (73) | → NaN y reimputar | Imposible con clientes residentes |
| Sin datos socioeconómicos (6,3 %) | Mediana por país + flag `socio_missing` | AT 8,3 % vs CH 5,1 %: imputar por país |
| `speed_average` > 150 (26) y consumo > 30 (401) | → NaN | Físicamente imposibles como medias |
| `trip_distance_average` > 100 (106) | Mantener; winsorizar al modelar | Posible uso de furgonetas |
| 0 viajes / vehículos casi parados | Mantener; excluir al estimar hábitos de conducción | Son vehículos conectados sin uso (dato válido) |
| `*_mode` con listas de bins | Parsear; lista → NaN + flag; excluir columna | Lista = empate, no hay moda real |
| Pico en 36 meses (12 %) y 0 meses (4 %) | **No modificar**; documentar | Probable valor por defecto; no se puede corregir sin la fuente |
""")

# =========================================================================== #
md("""
## 5. Identificación y exclusión de columnas

Criterios: (a) identificadores/metadatos sin significado analítico, (b) cardinalidad muy alta, (c) redundancia (derivadas o |ρ Spearman| ≥ 0,95),
(d) artefactos técnicos. Evidencia numérica de la redundancia:
""")

code("""
soc = raw[["purchasing_power_per_inhabitant_eur", "purchasing_power_per_household_eur", "population",
           "number_of_households", "purchasing_power_millions_eur"]]
ratio = soc.purchasing_power_millions_eur * 1e6 / (soc.purchasing_power_per_inhabitant_eur * soc.population)
print(f"millones / (per_cápita × población): min={ratio.min():.4f}  max={ratio.max():.4f}  -> columna DERIVADA")
sp = soc.corr(method="spearman")
short = ["pp_habitante", "pp_hogar", "población", "n_hogares", "pp_millones"]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(sp, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(5)); ax.set_xticklabels(short, rotation=40, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(short)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{sp.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(sp.iloc[i, j]) > .6 else "black")
plt.colorbar(im, fraction=.046); ax.set_title("Spearman – bloque socioeconómico"); plt.tight_layout(); plt.show()

print("\\nrole vs role_type:"); display(pd.crosstab(raw.role, raw.role_type))
ser = raw.dropna(subset=["model_class"]).groupby("model_series").model_class.nunique()
print(f"model_series con una única model_class: {(ser == 1).sum()} de {raw.model_series.nunique()}")
mm = raw[["speed_mode", "speed_median"]].dropna()
print(f"speed_mode == speed_median en el {(pd.to_numeric(mm.speed_mode, errors='coerce') == mm.speed_median).mean()*100:.0f} % de las filas")
""")

code("""
excl = pd.DataFrame([(c, cat, why, raw[c].nunique(), raw[c].isna().mean() * 100)
                     for c, (cat, why) in C.EXCLUDED_COLUMNS.items()],
                    columns=["columna", "categoría", "motivo", "n únicos", "% nulos"])
pd.set_option("display.max_colwidth", 140)
display(excl)
print(f"\\nSe excluyen {len(excl)} de {raw.shape[1]} columnas.")
""")

md("""
**Se conservan en el dataset limpio como claves o trazabilidad, pero no entran en el análisis.** `purchasing_power_per_household_eur` y
`number_of_households` se descartan por ρ = 0,96 y 0,998 con otras variables; `purchasing_power_millions_eur` es el producto exacto de dos columnas.
`model_series` es un código nominal (tratarlo como número induciría un orden falso).
""")

# =========================================================================== #
md("""
## 6. Limpieza propuesta

Todo el tratamiento está encapsulado en `eda/cleaning.py` (`clean()`), de modo que es reutilizable (API, modelos). Devuelve el dataset tratado y un log de cuántos valores se modificaron.
""")

code("""
log = {}
df = C.clean(raw, log)
cl = df.drop_duplicates("client_id")          # una fila por cliente (para atributos de cliente)
pd.Series(log, name="valores afectados").to_frame()
""")

code("""
feats = C.analytic_features(df)
print(f"Variables analíticas finales: {len(feats)}\\n")
print(feats)
print(f"\\nShape limpio: {df.shape}  (columnas nuevas: vehicle_age_years, has_telematics, socio_missing, flags *_multimodal...)")
""")

# =========================================================================== #
md("""
## 7. Estadísticos descriptivos (datos limpios)

Variables de cliente → una fila por cliente (n = 173.456). Variables de vehículo/telemetría → nivel fila.
""")

code("""
num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool and df[c].nunique() > 2]

def describe(src, c):
    s = src[c].dropna(); q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return {"variable": c, "nivel": "cliente" if c in CLIENT_COLS else "fila", "n": len(s),
            "% nulos": src[c].isna().mean() * 100, "media": s.mean(), "std": s.std(), "min": s.min(),
            "P25": q1, "mediana": s.median(), "P75": q3, "P99": s.quantile(.99), "max": s.max(),
            "asimetría": s.skew(), "% outliers IQR": ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean() * 100}

desc = pd.DataFrame([describe(cl if c in CLIENT_COLS else df, c) for c in num]).set_index("variable")
display(desc)
""")

code("""
cat = [c for c in feats if c not in num]
filas = []
for c in cat:
    src = cl if c in CLIENT_COLS else df
    vc = src[c].astype(str).value_counts(normalize=True) * 100
    filas.append({"variable": c, "nivel": "cliente" if c in CLIENT_COLS else "fila", "n niveles": src[c].nunique(),
                  "top 4 (%)": ", ".join(f"{k}: {v:.1f}" for k, v in vc.head(4).items())})
pd.set_option("display.max_colwidth", 120)
display(pd.DataFrame(filas).set_index("variable"))
""")

md("""
**Notas:** `trip_distance_median` y `speed_median` son discretas (marcas de bin), no continuas. `speed_all_counter` tiene asimetría 146 (máx. 1,45·10⁹) → usar log.
`currently_connected_vehicles` tiene un 19 % de outliers por las flotas (hasta 21 vehículos conectados).
""")

# =========================================================================== #
md("""
## 8. Distribuciones de variables clave
""")

code("""
n_pl = len(num); ncol = 4; nrow = int(np.ceil(n_pl / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
for ax, c in zip(axes.ravel(), num):
    src = cl if c in CLIENT_COLS else df
    lvl = "clientes" if c in CLIENT_COLS else "filas"
    if c in ("purchasing_power_per_inhabitant_eur", "population"):
        src = src[~src.socio_missing]; lvl += ", sin imputados"      # evita el pico artificial de la imputación
    s = src[c].dropna()
    if s.nunique() <= 25:
        vc = (s.clip(upper=s.quantile(.999)) if s.nunique() > 15 else s).value_counts().sort_index()
        ax.bar([f"{v:g}" for v in vc.index], vc.values, color=BLUE); ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.set_title(f"{c} ({lvl})", fontsize=9); continue
    use_log = s.min() >= 0 and s.skew() > 3
    x = np.log10(1 + s) if use_log else s
    lo, hi = x.quantile([.001, .999])
    ax.hist(x.clip(lo, hi), bins=60, color=BLUE); ax.axvline(x.median(), color=RED, ls="--", lw=1)
    ax.set_title(f"{c} ({lvl}){' [log10(1+x)]' if use_log else ''}", fontsize=9)
for ax in axes.ravel()[n_pl:]: ax.axis("off")
fig.suptitle("Numéricas analíticas (línea roja = mediana; log si asimetría > 3)", y=1.0)
plt.tight_layout(); plt.show()
""")

code("""
cats = [c for c in cat if c != "model_class"] + ["model_class"]
ncol = 3; nrow = int(np.ceil(len(cats) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow))
for ax, c in zip(axes.ravel(), cats):
    src = cl if c in CLIENT_COLS else df
    vc = (src[c].astype(str).value_counts(normalize=True).head(15) * 100).iloc[::-1]
    ax.barh(vc.index, vc.values, color=[RED if i == "UNKNOWN" else BLUE for i in vc.index])
    ax.set_title(f"{c} ({'clientes' if c in CLIENT_COLS else 'filas'})", fontsize=9); ax.set_xlabel("%"); ax.tick_params(axis="y", labelsize=7)
for ax in axes.ravel()[len(cats):]: ax.axis("off")
plt.tight_layout(); plt.show()
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 3.3))
yr = df.date_of_first_registration.dt.year.value_counts().sort_index()
ax[0].bar(yr.index, yr.values, color=BLUE); ax[0].set_title("Vehículos por año de primer registro (2026 parcial)")
orden = ["COMBUSTION", "MILD_HYBRID", "PLUGIN_HYBRID", "ELECTRIC", "UNKNOWN"]
data = [df.loc[df.engine_concept == e, "fuel_consumption_average"].dropna() for e in orden]
ax[1].boxplot(data, tick_labels=[f"{e}\\n(n={len(x)})" for e, x in zip(orden, data)], showfliers=False)
ax[1].set_title("Consumo (l/100 km) por motorización"); ax[1].tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.show()
""")

code("""
corr = df[num].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=7)
for i in range(len(num)):
    for j in range(len(num)):
        ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(corr.iloc[i, j]) > .6 else "black")
plt.colorbar(im, fraction=.04); ax.set_title("Correlación de Spearman (variables analíticas)"); plt.tight_layout(); plt.show()
""")

md("""
**Lectura:** velocidad media y viaje medio ≈ (log)normales; horas de uso y contadores log-normales con una masa en 0 (vehículos sin uso).
El consumo es coherente con la electrificación. Correlaciones a vigilar al modelar: `trip_distance_all_counter` ↔ `operating_hours_total` (0,8), ambos ↔ `speed_all_counter` (0,9),
`speed_average` ↔ `speed_median` (0,85); y las visitas a taller se asocian con el uso (ρ ≈ 0,4–0,5).
""")

# =========================================================================== #
md("""
## 9. Perfil del cliente

Para responder *¿cuáles son las principales características de los clientes?* se construye una tabla **a nivel cliente** (una fila por `client_id`),
agregando sus vehículos.
""")

code("""
g = df.groupby("client_id")
perfil = g.agg(n_vehiculos=("vehicle_id", "nunique"),
               tiene_vehiculo_actual=("is_current_vehicle", "any"),
               visitas_taller_total=("workshop_visits_warranty_goodwill", "sum"),
               tiene_telematica=("has_telematics", "any"),
               vehiculos_electrificados=("engine_concept", lambda s: s.isin(["ELECTRIC", "PLUGIN_HYBRID", "MILD_HYBRID"]).sum()),
               edad_veh_mediana=("vehicle_age_years", "median"))
perfil = perfil.join(cl.set_index("client_id")[["age_range", "gender_derived", "country_code", "months_as_mb_customer",
                                                 "finance_leasing_status", "currently_connected_vehicles",
                                                 "purchasing_power_per_inhabitant_eur"]])
perfil["segmento"] = np.where(perfil.n_vehiculos >= 10, "Flota (>=10)", np.where(perfil.n_vehiculos > 1, "Multi-vehículo (2-9)", "1 vehículo"))
print(f"Clientes: {len(perfil):,}")
display(perfil.segmento.value_counts(normalize=True).mul(100).round(1).to_frame("% clientes"))
""")

code("""
display(pd.DataFrame({
  "% por edad": perfil.age_range.value_counts(normalize=True).sort_index().mul(100).round(1),
}).T)
display(pd.DataFrame({
  "% género": perfil.gender_derived.value_counts(normalize=True).mul(100).round(1),
}).T)
display(pd.DataFrame({
  "% país": perfil.country_code.value_counts(normalize=True).mul(100).round(1),
}).T)
display(pd.DataFrame({
  "% financiación": perfil.finance_leasing_status.value_counts(normalize=True).mul(100).round(1),
}).T)
print(f"Antigüedad como cliente: mediana {perfil.months_as_mb_customer.median():.0f} meses (media {perfil.months_as_mb_customer.mean():.1f})")
print(f"Poder adquisitivo por habitante: mediana {perfil.purchasing_power_per_inhabitant_eur.median():,.0f} (incluye los valores imputados por país)")
print(f"Clientes con >1 vehículo: {(perfil.n_vehiculos>1).mean()*100:.1f} %  |  con algún vehículo actual: {perfil.tiene_vehiculo_actual.mean()*100:.1f} %")
print(f"Clientes con algún vehículo electrificado (EV/PHEV/mild): {(perfil.vehiculos_electrificados>0).mean()*100:.1f} % (lower bound: motor nulo en 44 % de filas)")
print(f"Clientes con telemetría en algún vehículo: {perfil.tiene_telematica.mean()*100:.1f} %")
""")

code("""
# ¿Cómo cambia el perfil con la edad? (financiación, antigüedad, electrificación)
por_edad = perfil.groupby("age_range").agg(
    clientes=("n_vehiculos", "size"),
    antigüedad_mediana_meses=("months_as_mb_customer", "median"),
    pct_financiacion_activa=("finance_leasing_status", lambda s: s.str.startswith("active").mean() * 100),
    pct_electrificado=("vehiculos_electrificados", lambda s: (s > 0).mean() * 100),
    vehiculos_medios=("n_vehiculos", "mean"),
    pct_mujer=("gender_derived", lambda s: (s == "female").mean() * 100),
)
display(por_edad)

top = df[df.model_class != "UNKNOWN"]
display(top.groupby("age_range").model_class.agg(lambda s: ", ".join(s.value_counts().head(3).index)).to_frame("top-3 clases de modelo (filas)"))
""")

# =========================================================================== #
md("""
## 10. Resumen de hallazgos y conclusiones

### ¿Cuáles son las principales características de los clientes?

* **Base de clientes:** 173.456 clientes y 244.519 vehículos. El 73 % de los clientes tiene un único vehículo en el dataset, el 26 % entre 2 y 9, y existe un pequeño grupo de **flotas** (1,1 % de los clientes, ≥10 vehículos, hasta 561) que aporta el 20,6 % de las filas.
* **Edad:** distribución muy centrada en la madurez: 45-54 (21 %), 65+ (20 %), 55-64 (20 %), 35-44 (19 %); los menores de 25 son solo el 4,7 %.
* **Género:** 65 % hombres, 18 % mujeres, 17 % desconocido (más frecuente en usuarios no propietarios).
* **Geografía:** 66 % Suiza y 34 % Austria; el poder adquisitivo por habitante es alto (mediana ≈ 46.800 €) y bimodal.
* **Relación con la marca:** antigüedad mediana de 36 meses (con un pico anómalo exactamente en 36); el 71 % nunca ha tenido financiación/leasing, el 24 % tiene financiación activa.
* **Rol (a nivel fila):** 87 % propietarios (OWNER); los DRIVER (B2B) son el 1,8 %. El 90 % de los clientes tiene algún vehículo actual y el 60 % tiene telemetría en al menos uno. El 81 % declara 1 vehículo conectado y el 10 %, ninguno.
* **Vehículo:** 84 % turismos (grupo P) y 14 % vans/comerciales (T); clases dominantes GLC (15 %), C (12 %), A (11 %). Antigüedad mediana del vehículo 4,7 años (registros 2009-2026, pico 2023-24). La electrificación (mild/plug-in/eléctrico) ronda el 23 % de las filas y al menos el 30 % de los clientes tiene algún vehículo electrificado (cota inferior: el 44 % de las filas no tiene motor informado).
* **Uso (solo filas con telemetría):** velocidad media 49 km/h, viaje mediano 14 km, ~400 h de funcionamiento; el consumo baja con la electrificación (combustión 8,8 → mild hybrid 7,7 → plug-in 5,8 l/100 km, medianas). El 44 % de los vehículos no tiene visitas a taller por garantía/buena voluntad; el uso intensivo se asocia con más visitas.

### ¿Qué problemas de calidad hemos encontrado y cómo los hemos resuelto?

| Problema | Solución |
|---|---|
| Telemetría ausente en ~52 % de las filas (estructural: rol, eléctricos, grupo F, año) | No se imputa; flag `has_telematics`; análisis solo con filas con dato |
| `engine_concept` / `fuel_type` nulos (44 % / 59 %) dependientes del modelo | Categoría `UNKNOWN` |
| Género nulo (17 % de clientes) | Categoría `UNKNOWN` |
| 6,3 % sin variables socioeconómicas; `population = 0` (73); 5 `zipcode` mal formados | Normalizar CP, `population = 0` → NaN, imputación por mediana del país + flag `socio_missing` |
| 427 valores físicamente imposibles (velocidad media > 150, consumo > 30 l/100 km) | → NaN |
| Columnas `*_mode` como texto con listas de bins | Parseo a numérico, lista → NaN + flag; columnas excluidas |
| Pico anómalo de 36 meses de antigüedad (12 % de clientes) | No se modifica; se documenta y se recomienda contrastar con la fuente |
| La unidad de análisis es la relación cliente–vehículo, con flotas que sesgan las medias | Variables de cliente analizadas a nivel cliente; no se deduplica por vehículo |
| Telemetría con 0 viajes / vehículos casi parados | Se mantienen; se excluyen al estimar hábitos de conducción |
| 16 de 39 columnas sin valor analítico (índice, IDs, códigos, redundantes, artefactos) | Excluidas del análisis y justificadas (sección 5) |

Calidad estructural buena: 0 duplicados, 0 contradicciones de atributos de vehículo, 0 incoherencias motor/combustible y 0 casos de horas de uso mayores que la edad del vehículo.

**Próximos pasos:** confirmar con la organización el significado del pico de 36 meses y la posibilidad de recuperar `engine_concept` desde `vehicle_type`; decidir el nivel de análisis (cliente vs. relación) y el tratamiento de flotas; para modelar, transformar a log los contadores y la población, winsorizar la telemetría (P1–P99) y regularizar los contadores correlacionados.
""")


def build(execute: bool = True) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if execute:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NB_PATH)
    return NB_PATH


if __name__ == "__main__":
    p = build(execute="--no-exec" not in sys.argv)
    print("Notebook escrito en", p)
