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
6. Tratamiento de valores nulos y limpieza (módulo `eda.cleaning`)
7. Estadísticos descriptivos
8. Distribuciones
9. Perfil del cliente
10. Cubos y tablas derivadas (módulo `eda.cubes`); 10.1 nulos tratados; 10.2 columnas escaladas
11. **Resumen de hallazgos y conclusiones**
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
con_nulos = tipos[tipos["nulos"] > 0][["grupo", "nulos", "% nulos"]]
print(f"Columnas con nulos: {len(con_nulos)} de {len(tipos)} ({len(con_nulos) / len(tipos) * 100:.1f} %)")
print(f"Columnas sin nulos: {len(tipos) - len(con_nulos)}")
print(f"Celdas nulas en total: {int(tipos['nulos'].sum()):,} de {df0.size:,} ({tipos['nulos'].sum() / df0.size * 100:.2f} %)")
display(con_nulos.style.format({"nulos": "{:,.0f}", "% nulos": "{:.2f} %"}))
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

**Decisión:** como el motor es un atributo del *modelo*, se puede **recuperar desde el catálogo** (`vehicle_type`) de otras filas
donde sí está informado. Lo que no se pueda recuperar queda como categoría `UNKNOWN` (imputar por moda inventaría información).
Se detalla y valida en la sección 6.
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
antes = soc.set_axis(["pp_habitante", "pp_hogar", "población", "n_hogares", "pp_millones"], axis=1)
despues = pd.DataFrame({"pp_habitante": soc.purchasing_power_per_inhabitant_eur, "población": soc.population,
                        "personas_por_hogar": soc.population / soc.number_of_households.where(soc.number_of_households > 0)})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [5, 3.6]})
for ax, (tit, X) in zip(axes, [("ANTES: 5 columnas originales", antes),
                                ("DESPUÉS: combinadas (sección 5.1)", despues)]):
    sp = X.corr(method="spearman"); k = len(sp)
    im = ax.imshow(sp, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(k)); ax.set_xticklabels(sp.columns, rotation=40, ha="right"); ax.set_yticks(range(k)); ax.set_yticklabels(sp.columns)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sp.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(sp.iloc[i, j]) > .6 else "black")
    ax.set_title(f"Spearman – {tit}", fontsize=9)
plt.colorbar(im, ax=axes, fraction=.03); plt.show()

print("\\nrole vs role_type:"); display(pd.crosstab(raw.role, raw.role_type))
ser = raw.dropna(subset=["model_class"]).groupby("model_series").model_class.nunique()
print(f"model_series con una única model_class: {(ser == 1).sum()} de {raw.model_series.nunique()}")
mm = raw[["speed_mode", "speed_median"]].dropna()
print(f"speed_mode == speed_median en el {(pd.to_numeric(mm.speed_mode, errors='coerce') == mm.speed_median).mean()*100:.0f} % de las filas")
""")

md("""
**Cómo leer las matrices:** rojo = suben juntas, azul = cuando una sube la otra baja, blanco = sin relación. La diagonal (1,00) es cada variable consigo misma.
El problema son los valores cercanos a ±1 **fuera** de la diagonal: a la izquierda hay varios (0,96–1,00), porque 5 columnas repiten solo 3 datos.
A la derecha quedan 3 columnas con información distinta y el máximo es −0,51 (`población` vs `personas_por_hogar`): una relación real y moderada,
**en las zonas más pobladas los hogares son más pequeños**, no una repetición. Menos columnas no implica perder datos: las originales se reconstruyen exactamente (sección 5.1).
""")

code("""
limpio = C.clean(raw)   # las derivadas de telemetría solo existen tras limpiar
col = lambda c: raw[c] if c in raw else limpio[c]
excl = pd.DataFrame([(c, cat, why, col(c).nunique(), col(c).isna().mean() * 100)
                     for c, (cat, why) in C.EXCLUDED_COLUMNS.items()],
                    columns=["columna", "categoría", "motivo", "n únicos", "% nulos"])
pd.set_option("display.max_colwidth", 140)
display(excl)
print(f"\\nSe excluyen {len(excl)} de {raw.shape[1]} columnas.")
""")

md("""
**Se conservan en el dataset limpio como claves o trazabilidad, pero no entran en el análisis.** `purchasing_power_per_household_eur` y
`number_of_households` (ρ = 0,96 y 0,998 con otras variables) y `purchasing_power_millions_eur` (producto exacto de dos columnas) no se eliminan sin más: se **combinan** en `personas_por_hogar` (sección 5.1).
`model_series` es un código nominal (tratarlo como número induciría un orden falso).
""")

md("""
### 5.1 Variables muy correlacionadas: combinar en vez de eliminar

La matriz anterior muestra dos grupos casi idénticos: **{pp_habitante, pp_hogar}** (ρ = 0,96) y **{población, n_hogares, pp_millones}** (ρ ≥ 0,96).
En la telemetría pasa lo mismo con los contadores (`operating_hours_total`, `trip_distance_all_counter`, `speed_all_counter`, ρ = 0,86–0,93)
y, en menor medida, con distancia y velocidad: media vs. mediana de la misma medida (ρ = 0,55–0,83) y distancia media vs. velocidad y viajes por hora (|ρ| ≈ 0,7).
Opciones para tratar la multicolinealidad, de más a menos drástica:

| Opción | Qué hace | Pérdida de información | Interpretabilidad |
|---|---|---|---|
| Eliminar una de cada par | Se queda con una columna | **Sí** (lo que la otra tuviera de distinto) | Alta |
| **Combinar en "nivel + ratio"** | Se mantiene una medida de tamaño y la otra se sustituye por su cociente con ella | **No**: B = ratio × A es reversible | Alta (los ratios tienen significado de negocio) |
| PCA / índice compuesto | Proyecta el bloque en componentes incorreladas | Pequeña (si se descartan componentes) | Baja |

Se elige **combinar**: los cocientes no son arbitrarios, salen de **identidades exactas** del propio dataset, que se comprueban a continuación.
""")

code("""
cl_raw = raw.drop_duplicates("client_id")          # variables socioeconómicas: nivel cliente
hog = cl_raw.number_of_households.where(cl_raw.number_of_households > 0)
r_pp = cl_raw.purchasing_power_per_household_eur / cl_raw.purchasing_power_per_inhabitant_eur
r_pop = cl_raw.population / hog
r_mill = cl_raw.purchasing_power_millions_eur * 1e6 / (cl_raw.purchasing_power_per_inhabitant_eur * cl_raw.population)
tel = raw[raw.operating_hours_total >= 1]
r_seg = tel.speed_all_counter / tel.operating_hours_total

ident = pd.DataFrame({
    "pp_hogar / pp_habitante  ÷  (población / n_hogares)": (r_pp / r_pop).describe(percentiles=[.05, .25, .5, .75, .95]),
    "pp_millones  ÷  (pp_habitante × población)": r_mill.describe(percentiles=[.05, .25, .5, .75, .95]),
    "speed_all_counter / horas de uso": r_seg.describe(percentiles=[.05, .25, .5, .75, .95]),
}).T[["count", "min", "5%", "25%", "50%", "75%", "95%", "max"]]
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
display(ident)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
""")

md("""
**Lectura:**
* `pp_hogar / pp_habitante` es **exactamente** `población / n_hogares` (cociente 1,0000): ambas columnas "de más" contienen una única información nueva, el **tamaño medio del hogar** → `personas_por_hogar`.
* `pp_millones` es **exactamente** `pp_habitante × población`: no contiene nada que no esté en esas dos.
* `speed_all_counter / horas` ≈ **3.600**: el contador de velocidad son los **segundos registrados** (una muestra por segundo); es la misma magnitud que las horas en otra unidad. Lo que sí aporta es su desviación de 3.600 (cobertura del registro) → `seg_registrados_por_hora`.
* `trip_distance_all_counter` crece con las horas de uso; lo informativo es la **frecuencia de viajes** → `viajes_por_hora`.

Transformación aplicada en `cleaning.add_combined_features()` (los ratios de telemetría solo se calculan con ≥ 1 h de uso, para evitar dividir por ~0):
""")

code("""
display(pd.DataFrame([(k, v[0], v[1]) for k, v in C.COMBINED_FEATURES.items()],
                     columns=["columna original", "se sustituye por", "relación"]).set_index("columna original"))
""")

code("""
def vif(X):
    # VIF_i = diagonal de la inversa de la matriz de correlación (>10 = multicolinealidad grave, <5 aceptable)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    R = np.corrcoef(X.to_numpy().T)
    return pd.Series(np.diag(np.linalg.pinv(R)), index=X.columns)

lg = lambda s: np.log10(s.where(s > 0))
antes_soc = pd.DataFrame({"pp_habitante": cl_raw.purchasing_power_per_inhabitant_eur, "pp_hogar": cl_raw.purchasing_power_per_household_eur,
                          "log_poblacion": lg(cl_raw.population), "log_n_hogares": lg(cl_raw.number_of_households),
                          "log_pp_millones": lg(cl_raw.purchasing_power_millions_eur)})
desp_soc = pd.DataFrame({"pp_habitante": cl_raw.purchasing_power_per_inhabitant_eur, "log_poblacion": lg(cl_raw.population),
                         "personas_por_hogar": r_pop})
antes_tel = pd.DataFrame({"log_horas": lg(tel.operating_hours_total), "log_n_viajes": lg(tel.trip_distance_all_counter),
                          "log_speed_counter": lg(tel.speed_all_counter)})
desp_tel = pd.DataFrame({"log_horas": lg(tel.operating_hours_total),
                         "log_viajes_por_hora": lg(tel.trip_distance_all_counter / tel.operating_hours_total),
                         "log_seg_por_hora": lg(tel.speed_all_counter / tel.operating_hours_total)})
antes_ruta = pd.DataFrame({"dist_media": tel.trip_distance_average, "dist_mediana": tel.trip_distance_median,
                           "vel_media": tel.speed_average, "vel_mediana": tel.speed_median,
                           "viajes_por_hora": tel.trip_distance_all_counter / tel.operating_hours_total})
desp_ruta = pd.DataFrame({"vel_media": tel.speed_average,
                          "asimetria_dist": tel.trip_distance_average / tel.trip_distance_median.where(tel.trip_distance_median > 0),
                          "asimetria_vel": tel.speed_average / tel.speed_median.where(tel.speed_median > 0),
                          "viajes_por_hora": antes_ruta.viajes_por_hora})

bloques = [("Socioeconómico ANTES", antes_soc), ("Socioeconómico DESPUÉS", desp_soc),
           ("Contadores telemetría ANTES", antes_tel), ("Contadores telemetría DESPUÉS", desp_tel),
           ("Distancia y velocidad ANTES", antes_ruta), ("Distancia y velocidad DESPUÉS", desp_ruta)]
fig, axes = plt.subplots(3, 2, figsize=(10.5, 13)); axes = axes.ravel()
for ax, (tit, X) in zip(axes, bloques):
    cr = X.corr(method="spearman")
    ax.imshow(cr, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(cr))); ax.set_xticklabels(cr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cr))); ax.set_yticklabels(cr.columns, fontsize=7)
    for i in range(len(cr)):
        for j in range(len(cr)):
            ax.text(j, i, f"{cr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(cr.iloc[i, j]) > .6 else "black")
    ax.set_title(tit, fontsize=9)
plt.suptitle("Correlación de Spearman antes y después de combinar", y=1.03); plt.tight_layout(); plt.show()

display(pd.concat({"socio ANTES": vif(antes_soc), "socio DESPUÉS": vif(desp_soc),
                   "telemetría ANTES": vif(antes_tel), "telemetría DESPUÉS": vif(desp_tel),
                   "dist/vel ANTES": vif(antes_ruta.apply(lg)), "dist/vel DESPUÉS": vif(desp_ruta.apply(lg))}).to_frame("VIF"))

km_trayecto_aprox = desp_ruta.vel_media / desp_ruta.viajes_por_hora
print("dist_media / (vel_media / viajes_por_hora):",
      (tel.trip_distance_average / km_trayecto_aprox).replace([np.inf, -np.inf], np.nan).quantile([.25, .5, .75]).round(2).to_dict())
""")

md("""
**Lectura:** en las matrices "ANTES" hay mucho rojo intenso (columnas que dicen lo mismo); en las "DESPUÉS" todos los valores son moderados
(|ρ| ≤ 0,51), y el azul indica relaciones inversas reales (más población → hogares más pequeños; más horas de uso → menos viajes por hora).
El **VIF** mide lo repetida que está cada columna (≈ 1 = información propia, > 10 = grave): pasa de hasta 2.500 a menos de 2,1.

**Distancia y velocidad:** media y mediana de una misma medida se sustituyen por la media + su **asimetría** (media / mediana: > 1 indica
que unos pocos trayectos largos o tramos rápidos tiran de la media). `trip_distance_average` sale del conjunto analítico porque se deduce de las otras:
km por trayecto ≈ velocidad media / viajes por hora (distancia real ÷ aproximada: mediana 1,07, P25–P75 = 1,02–1,16).
""")

code("""
# Comprobación de que NO se pierde información: se reconstruyen las columnas originales desde las combinadas
rec = pd.DataFrame({
    "n_hogares": cl_raw.population / r_pop,
    "pp_hogar": cl_raw.purchasing_power_per_inhabitant_eur * r_pop,
    "pp_millones": cl_raw.purchasing_power_per_inhabitant_eur * cl_raw.population / 1e6,
})
orig = cl_raw[["number_of_households", "purchasing_power_per_household_eur", "purchasing_power_millions_eur"]].set_axis(rec.columns, axis=1)
err = ((rec - orig).abs() / orig.abs()).max() * 100
display(err.to_frame("error relativo máximo de reconstrucción (%)").T)
""")

md("""
**Resultado:** tras combinar, las correlaciones dentro de cada bloque caen de hasta ρ = 1,00 a |ρ| ≤ 0,51, y el VIF pasa de 150–2.500 (socioeconómico),
6,6–12 (contadores) y 1,4–4,6 (distancia/velocidad) a 1,0–2,1 (por debajo de 5 se considera aceptable). Las socioeconómicas se reconstruyen con un
error < 0,1 % (redondeo del CSV); la distancia media, de forma aproximada. Las nuevas variables tienen lectura de negocio: **personas por hogar**
(zonas familiares vs. urbanas unipersonales), **viajes por hora** (trayectos cortos urbanos vs. carretera), **asimetría de distancia/velocidad**
(uso regular vs. viajes largos ocasionales) y **cobertura del registro telemático**.

**Inconsistencia que destapa el ratio:** el 10,8 % de las filas con telemetría registra más de 7.200 segundos por hora de uso y 578 filas superan
60 viajes por hora, valores imposibles si ambos contadores midieran lo mismo. Se mantienen, pero conviene winsorizarlos o marcarlos antes de modelar.

Las columnas originales se conservan en el dataset limpio; en el conjunto analítico se usan las combinadas.
""")

# =========================================================================== #
md("""
## 6. Tratamiento de valores nulos

No todas las columnas con nulos se tratan igual. La decisión depende de **dos cosas**:
1. **Cuántos nulos hay** (tramo): con pocos nulos cualquier imputación razonable apenas altera la distribución; con muchos, imputar "inventa" la mayor parte de la columna.
2. **Por qué faltan** (mecanismo, visto en la sección 3): si es aleatorio (MCAR) se puede imputar; si depende de otras variables (MAR) hay que imputar *usando esas variables*; si es estructural (el dato no existe, p. ej. consumo de combustible en un eléctrico) **no se debe imputar**.

### 6.1 Marco de decisión por tramos
""")

code("""
display(pd.DataFrame([(name, f"{lo}–{min(hi, 100)} %", rec) for lo, hi, name, rec in C.NULL_TIERS],
                     columns=["tramo", "rango", "qué hacer"]).set_index("tramo"))
""")

md("""
### 6.2 ¿Qué hacer con las columnas con **muchos** nulos (> 40 %)?

Hay 12 columnas en este tramo: las 10 de telemetría, `engine_concept` y `fuel_type`. Antes de decidir eliminarlas, se siguen tres pasos en orden:

| Paso | Pregunta | Aplicado a |
|---|---|---|
| 1. **Recuperar** | ¿El dato existe en otra columna/fila? | Catálogo `vehicle_type` + **reglas en el nombre**: EQ/eVito → eléctrico; **`220 d` / CDI → diésel**; **`350 de` → diésel+eléctrico (PHEV diésel)**; **`350 e` → gasolina+eléctrico** (no confundir `e` suelta con `de`) |
| 2. **Separar** | ¿La ausencia es estructural? | Telemetría: ~60 % nulos (eléctricos, rol, grupo F...) → **no se imputa**; se crea `has_telematics` |
| 3. **Eliminar del analisis** | ¿Aporta para segmentar clientes con tantos nulos? | **Excluidas** del conjunto analítico: todo el bloque de telemetría (`trip_distance_*`, `speed_*`, `fuel_consumption_average`, `operating_hours_total`, ratios derivados…). Se conservan en el CSV limpio para trazabilidad; el clustering usa solo el flag **`has_telematics`** |

Imputar distancia/velocidad/consumo haría que ~60 % de las filas compartieran un comportamiento inventado; meterlas en K-Means separaría sobre todo “tiene telemetría o no”.

### 6.3 ¿Y las de **pocos** nulos?

* `months_as_mb_customer` (4 filas, MCAR) → mediana.
* `date_of_first_registration` (1,4 %, concentrado en modelos antiguos) → **se predice** desde los datos del vehículo + flag `registration_date_missing` (validación en la sección 6.7).
* `model_class` (1,1 %) → catálogo `vehicle_type` + reglas por prefijo comercial (sección 6.4); `vehicle_type` (0,3 %) → `UNKNOWN` si sigue nulo.
* Socioeconómicas (6,3 %, tramo medio) → **mediana por país** (AT y CH tienen niveles distintos) + flag `socio_missing`.
* `gender_derived` (11 %, tramo medio, MAR por rol) → `UNKNOWN`: no hay ninguna columna de la que inferirlo de forma fiable.

### 6.4 Validación de la recuperación desde el catálogo

Antes de aplicar la imputación por catálogo se valida: se ocultan los valores conocidos del 20 % de los vehículos, se construye el catálogo con el 80 % restante y se mide el acierto.
""")

code("""
def validar_catalogo(target, key="vehicle_type", seed=0):
    known = raw.dropna(subset=[target, key])
    veh = known.vehicle_id.unique()
    test_v = set(np.random.default_rng(seed).choice(veh, len(veh) // 5, replace=False))
    tr, te = known[~known.vehicle_id.isin(test_v)], known[known.vehicle_id.isin(test_v)]
    pred = te[key].map(C.catalog_mapping(tr, target, key))
    ok = pred.notna()
    return {"variable": f"{target} (por {key})", "cobertura holdout %": ok.mean() * 100,
            "acierto holdout %": (pred[ok] == te.loc[ok, target]).mean() * 100}

val = pd.DataFrame([validar_catalogo("engine_concept"), validar_catalogo("fuel_type"),
                    validar_catalogo("model_class"), validar_catalogo("model_class", "model_series")]).set_index("variable")

# reglas por nombre comercial, validadas sobre filas con motor conocido
vt = raw.vehicle_type.fillna("")
reglas = {"ELECTRIC": vt.str.match(r"^(?:AMG )?EQ[A-Z]?\\b|^e(?:Vito|Sprinter|Citan)"),
          "PLUGIN_HYBRID": vt.str.contains(r"\\b\\d{3} (?:e|de)\\b")}
conocido = raw.engine_concept.notna()
val_reglas = pd.DataFrame([{"regla": f"nombre -> {k}", "filas conocidas que cumplen": int((m & conocido).sum()),
                            "acierto %": (raw.loc[m & conocido, "engine_concept"] == k).mean() * 100}
                           for k, m in reglas.items()]).set_index("regla")
fk = raw.dropna(subset=["fuel_type", "vehicle_type"]).copy()
sim = fk.assign(fuel_type=pd.NA)
pred_f, m_f = C._fuel_from_name(sim.fuel_type, sim.vehicle_type)
val_fuel_reglas = pd.Series({"cobertura sobre fuel conocido %": m_f.mean() * 100,
                             "acierto exacto %": (pred_f[m_f] == fk.loc[m_f, "fuel_type"]).mean() * 100})
mk = raw.dropna(subset=["model_class", "vehicle_type"]).copy()
sim_mc = mk.assign(model_class=pd.NA)
pred_mc, m_mc = C._model_class_from_name(sim_mc.model_class, sim_mc.vehicle_type)
val_mc_reglas = pd.Series({"cobertura sobre class conocida %": m_mc.mean() * 100,
                           "acierto exacto %": (pred_mc[m_mc] == mk.loc[m_mc, "model_class"]).mean() * 100})
nulos_mc = raw.model_class.isna().sum()
print(f"Nulos model_class en crudo: {nulos_mc} (series sin ninguna fila con class: "
      f\"{', '.join(map(str, raw.loc[raw.model_class.isna(), 'model_series'].astype(str).unique()[:8]))}\")
display(val, val_reglas, val_fuel_reglas.to_frame("fuel_type (reglas nombre)"), val_mc_reglas.to_frame("model_class (reglas nombre)"))
""")

md("""
El catálogo acierta en más del 99,8 % de los casos que cubre y las reglas de **motor** en 99–100 %. Las de **combustible** aciertan un ~96 % en conjunto:
100 % en diésel `d`/CDI y eléctricos EQ, pero `350 e` → gasolina+eléctrico solo un ~72 % (el resto es `PETROL_PLUS,ELECTRIC`). Por eso se aplican
**después** del catálogo, solo donde este no llega.
**`model_class` (1,1 % nulos):** casi todos son **series nuevas** (p. ej. 540 = GLC, 244 = GLB) sin ninguna fila con clase en el dataset → el catálogo por `model_series` no aplica; se recupera con catálogo por `vehicle_type` (cubre los CLA) + reglas en el nombre (`GLC` → GLC-Class, `GLB` → B-/GLA-/GLB-Class, `C 400` → C-Class, `VLE` → Vito).
Ojo: en Mercedes **`d` y `de` no son lo mismo**: `C 220 d` es diésel puro; `C 350 de` es híbrido enchufable diésel (`DIESEL,ELECTRIC`); `C 350 e` es híbrido gasolina (`PETROL,ELECTRIC`). Por eso no basta con buscar una `e` suelta sin mirar el patrón completo.

Con criterio conservador, el catálogo solo se usa si una denominación tiene **un único valor en ≥ 95 % de sus filas** y al menos 5 filas; después entran las reglas de nombre.

### 6.5 Aplicación y resultado

Todo el tratamiento está en `eda/cleaning.py` (`clean()`), reutilizable desde la API o los modelos. Devuelve el dataset tratado y un log con cuántos valores se modificaron en cada paso.
""")

code("""
log = {}
df = C.clean(raw, log)
cl = df.drop_duplicates("client_id")          # una fila por cliente (para atributos de cliente)
CLIENT_COLS += ["personas_por_hogar", "log_population"]   # derivadas de atributos de cliente
pd.Series(log, name="valores afectados").to_frame()
""")

code("""
# Dataset sin ninguna columna con NaN, listo para manipular (UNKNOWN se mantiene como categoría)
df_sin_nulos = C.drop_null_columns(df)
eliminadas = df.columns.difference(df_sin_nulos.columns)
print(f"Columnas eliminadas por tener NaN: {len(eliminadas)} -> quedan {df_sin_nulos.shape[1]} de {df.shape[1]}")
display(df[eliminadas].isna().agg(["sum", "mean"]).T.rename(columns={"sum": "nulos", "mean": "% nulos"})
        .assign(**{"% nulos": lambda t: t["% nulos"] * 100}).sort_values("% nulos", ascending=False)
        .style.format({"nulos": "{:,.0f}", "% nulos": "{:.2f} %"}))
df_sin_nulos.to_parquet(ROOT / "eda" / "output" / "dataset_sin_nulos.parquet", index=False)
df_sin_nulos.head()
""")

code("""
estrategia = C.null_strategy_table(raw, df)
pd.set_option("display.max_colwidth", 80)
display(estrategia)
""")

code("""
# Antes / después: % de nulos (o UNKNOWN) por columna, solo columnas que siguen en el análisis
cols = [c for c in estrategia.index if c not in C.EXCLUDED_COLUMNS]
e = estrategia.loc[cols].sort_values("% nulos crudo")
fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(e))
ax.barh(y + .2, e["% nulos crudo"], height=.4, color=GREY, label="crudo")
ax.barh(y - .2, e["% nulo/UNKNOWN tras limpiar"], height=.4, color=BLUE, label="tras tratamiento (NaN o UNKNOWN)")
ax.set_yticks(y); ax.set_yticklabels(e.index, fontsize=8); ax.set_xlabel("%")
for lim, lab in ((5, "5 %"), (40, "40 %")):
    ax.axvline(lim, color=RED, ls=":", lw=1); ax.text(lim, len(e) - .3, lab, color=RED, fontsize=8)
ax.set_title("Nulos antes y después del tratamiento (las de telemetría se mantienen NaN a propósito)")
ax.legend(loc="lower right"); plt.tight_layout(); plt.show()
""")

md("""
**Resultado:** `engine_concept` pasa de 44 % a ~21 % de desconocidos y `fuel_type` de 59 % a ~24 % (catálogo + reglas `d`/`e`/`de`), sin inventar valores
(se marcan con `engine_concept_imputed` / `fuel_type_imputed`). Las socioeconómicas quedan completas (con flag). La telemetría se mantiene
con sus nulos estructurales y se trabaja en su propia tabla (sección 10).
""")

md("""
### 6.6 ¿Se puede predecir `gender_derived` (16,9 % de clientes sin dato)?

Idea: usar lo que sabemos del cliente (modelo de coche, edad, país, leasing, uso…) y lo que se observa en el resto de clientes para "predecir" el género
de los nulos. Antes de imputar hay que comprobar dos cosas: **quiénes son los nulos** y **si el modelo acierta más que no hacer nada**.
""")

code("""
clg = df.drop_duplicates("client_id").set_index("client_id")
clg["sin_genero"] = clg.gender_derived == "UNKNOWN"
nrow = df.groupby("client_id").size()
clg["segmento"] = pd.cut(nrow, [0, 1, 9, 1e9], labels=["1 vehículo", "2-9", "flota"]).astype(str)
for c in ["role_type", "role", "age_range", "segmento"]:
    t = clg.groupby(c).sin_genero.agg(["mean", "size"]).rename(columns={"mean": "% sin género", "size": "clientes"})
    t["% sin género"] *= 100
    display(t.sort_values("% sin género", ascending=False).T)
""")

code("""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def moda(col):
    s = df.groupby(["client_id", col]).size().reset_index(name="n").sort_values("n", ascending=False)
    return s.drop_duplicates("client_id").set_index("client_id")[col]

Xg = clg[["age_range", "country_code", "finance_leasing_status", "months_as_mb_customer", "currently_connected_vehicles",
          "workshop_visits_warranty_goodwill", "purchasing_power_per_inhabitant_eur", "personas_por_hogar", "log_population"]].copy()
Xg["n_vehiculos"] = nrow
for col in ["model_class", "vehicle_type", "engine_concept", "role"]:
    Xg[col] = moda(col)
Xg["vehicle_type"] = Xg.vehicle_type.where(Xg.vehicle_type.isin(Xg.vehicle_type.value_counts().index[:250]), "OTRO")
for c in Xg.columns[Xg.dtypes == "object"].tolist() + Xg.select_dtypes("string").columns.tolist():
    Xg[c] = Xg[c].astype("category")

conocido = ~clg.sin_genero
yk = (clg.gender_derived[conocido] == "female").astype(int)
p = cross_val_predict(HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=300, learning_rate=0.08, random_state=0),
                      Xg[conocido], yk, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]

mc = pd.crosstab(Xg.model_class[conocido], yk, normalize="index")[1] * 100
mc = mc[Xg.model_class[conocido].value_counts().loc[lambda s: s >= 200].index].sort_values()
print("% mujeres por clase de modelo (las 4 más y menos femeninas):")
print(pd.concat([mc.head(4), mc.tail(4)]).round(1).to_string())

pred = (p >= .5).astype(int)
res = [("Sin modelo: decir 'male' a todos", (yk == 0).mean() * 100, 100.0, 0.0),
       ("Modelo (umbral 0,5)", (pred == yk).mean() * 100, 100.0, pred.mean() * 100)]
for thr in [.8, .9]:
    conf = np.maximum(p, 1 - p) >= thr
    res.append((f"Modelo, solo predicciones con confianza ≥ {thr:.0%}", (pred == yk)[conf].mean() * 100, conf.mean() * 100, pred[conf].mean() * 100))
print(f"\\nAUC (validación cruzada 5 folds): {roc_auc_score(yk, p):.3f}")
display(pd.DataFrame(res, columns=["estrategia", "% acierto", "% clientes cubiertos", "% predichos female"]).set_index("estrategia"))
""")

md("""
**Conclusión: no se imputa el género.**

* **Los nulos no son empresas** (el 99,8 % son B2C): son sobre todo **usuarios temporales/`USER` (50–63 % sin género), menores de 25 (40 %)
  y clientes con un solo vehículo (21 %)**. Es un nulo MAR ligado a clientes con poca relación con la marca.
* **El modelo de coche sí está asociado al género** (Clase A ≈ 40 % mujeres, Clase S ≈ 4 %), pero no basta para predecirlo: el modelo
  acierta un 79 %, prácticamente lo mismo que decir "male" a todos (78 %), y **todas las predicciones con confianza alta son "male"**:
  no es capaz de identificar a ninguna mujer con seguridad.
* Imputar haría desaparecer mujeres del dataset (bajaría su peso real) y metería un sesgo en los perfiles. Además, `gender_derived` ya es un dato
  **inferido** por Mercedes; inferirlo otra vez a partir del coche sería un estereotipo, no información.

**Tratamiento:** los nulos se sustituyen por la categoría **`UNKNOWN`** (ya no quedan NaN). Es una categoría con significado propio
(clientes jóvenes, usuarios no propietarios, poca vinculación). En el clustering, el género se usará solo para **describir** los segmentos,
no como variable de entrada.
""")

md("""
### 6.7 ¿Se puede predecir `date_of_first_registration` (1,4 % de filas sin fecha)?

Mismo procedimiento que con el género: primero **quién** tiene el nulo, después **si la predicción acierta** más que una imputación simple.
""")

code("""
r = raw.assign(sin_fecha=pd.to_datetime(raw.date_of_first_registration, errors="coerce").isna())
v1 = r.drop_duplicates("vehicle_id")
print(f"Vehículos sin fecha: {v1.sin_fecha.sum():,}; con la fecha en otra fila del mismo vehículo: "
      f"{(r.groupby('vehicle_id').sin_fecha.agg(['any', 'all']).pipe(lambda t: t['any'] & ~t['all'])).sum()}")
t = r.groupby("model_class").sin_fecha.agg(["mean", "size"]).query("size >= 50")
t["mean"] *= 100
display(t.sort_values("mean", ascending=False).head(6).rename(columns={"mean": "% sin fecha", "size": "filas"}).T.round(1))
t = r.groupby("role").sin_fecha.mean().mul(100).sort_values(ascending=False).round(1)
display(t.to_frame("% sin fecha").T)
""")

code("""
fecha = pd.to_datetime(raw.date_of_first_registration, errors="coerce")
anyo = fecha.dt.year + (fecha.dt.dayofyear - 1) / 365.25
veh = raw.vehicle_id.drop_duplicates()
es_test = raw.vehicle_id.isin(set(veh[np.random.default_rng(0).random(len(veh)) < .2]))   # 20 % de vehículos ocultos
primera = ~raw.vehicle_id.duplicated()
ev = primera & es_test & anyo.notna()
tr = primera & ~es_test & anyo.notna()

serie = raw.model_series.astype(str)
med_serie = anyo[tr].groupby(serie[tr]).median()
estrategias = {
    "Mediana global": pd.Series(anyo[tr].median(), index=raw.index),
    "Mediana de su model_series": serie.map(med_serie),
    "Modelo (gradient boosting, acotado a su serie)": C.predict_registration_year(raw, train=~es_test),
}
filas = []
for nombre, pred in estrategias.items():
    for grupo, m in [("todos", ev), ("vehículos < 2016", ev & (anyo < 2016))]:
        e = (anyo - pred)[m].dropna()
        filas.append((nombre, grupo, e.abs().median(), e.abs().quantile(.9), (e.abs() <= 1).mean() * 100, e.median()))
display(pd.DataFrame(filas, columns=["estrategia", "vehículos test", "error mediano (años)", "error P90 (años)",
                                     "% con error ≤ 1 año", "sesgo (real − predicho)"]).set_index(["estrategia", "vehículos test"]))
""")

md("""
**Conclusión: sí se imputa, con el modelo.**

* Los nulos **no son aleatorios**: son modelos que ya no se fabrican (CLK, SLK, Viano/Vito, M-/GL: 57–94 % sin fecha) y el rol `LEGACY` (11,5 %).
  Ningún vehículo tiene la fecha en otra de sus filas, así que no se puede recuperar.
* Rellenar con la mediana global sería un error grave justo en estos vehículos: los pondría en ~2021 cuando son de antes de 2016
  (error mediano de ~7,5 años).
* El **modelo** (serie, tipo y clase del modelo, motorización, rol, vehículo actual, antigüedad como cliente, país) acierta con un error mediano de
  ~0,7 años, también en los vehículos antiguos (~1 año). Se acota al rango de años conocido de su serie para no extrapolar en modelos con
  casi ninguna fecha (sin el límite, un CLK quedaba en 2014; con él, en 2009).

A diferencia del género, aquí la predicción **sí mejora mucho** la alternativa simple y no crea sesgo. Se imputan las 4.182 filas, se marca
`registration_date_missing = True` y `vehicle_age_years` queda completa: los vehículos imputados tienen una mediana de ~9 años frente a ~5 del resto,
como cabía esperar.
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
**Notas:** Toda la telemetría numérica (~52–60 % nulos) **no entra** en este descriptivo ni en el clustering (sección 6.2). Solo se usa el flag **`has_telematics`** (¿tiene algún dato de uso conectado?).
La sección 5.1 sigue mostrando en datos crudos cómo se combinarían si se usaran; no aplican al modelo final.
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
El consumo es coherente con la electrificación. Tras las combinaciones de la sección 5.1 **ninguna pareja supera |ρ| = 0,51**: la mayor es
`log_population` ↔ `personas_por_hogar` (−0,5, en zonas más pobladas los hogares son más pequeños). Le siguen, con ±0,4, visitas a taller ↔ horas de uso
(más uso → más visitas), meses como cliente ↔ vehículos conectados, viajes por hora ↔ cobertura del registro y antigüedad ↔ cobertura del registro
(los vehículos antiguos registran menos). Son relaciones moderadas con sentido, no columnas repetidas.
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
print(f"Clientes con algún vehículo electrificado (EV/PHEV/mild): {(perfil.vehiculos_electrificados>0).mean()*100:.1f} % (cota inferior: motor desconocido en el 21 % de las filas)")
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
## 10. Cubos y tablas derivadas

Para no arrastrar los nulos estructurales de la telemetría a todos los análisis, el dataset limpio se reorganiza en **tablas por nivel y
completitud** y en **cubos** (agregados por dimensiones) listos para el backend/dashboard (`eda/cubes.py`):

| Tabla | Grano | Uso |
|---|---|---|
| `tabla_clientes` | 1 fila por cliente | Perfil del cliente + agregados de sus vehículos (sin nulos estructurales) |
| `tabla_telematica` | 1 fila por relación **con** telemetría | Claves del vehículo/cliente (sin métricas: todas excluidas del analisis) |
| `cubo_clientes` | país × edad × género × financiación | Segmentación de clientes |
| `cubo_vehiculos` | país × clase × motor × grupo × vehículo actual | Parque y taller |
| `cubo_telematica` | clase × motor | Cuántas relaciones tienen telemetría (sus métricas quedan fuera del análisis) |

Se descartan las celdas con menos de 30 observaciones (estadísticos inestables). Se guardan en `eda/output/cubos/` (parquet; los cubos también en CSV).
""")

code("""
from eda import cubes as K
cubos = K.build_all(df)
K.save_all(cubos)
display(pd.DataFrame({k: {"filas": v.shape[0], "columnas": v.shape[1],
                          "% celdas nulas": v.isna().mean().mean() * 100} for k, v in cubos.items()}).T)
""")

code("""
print("tabla_clientes (muestra)")
display(cubos["tabla_clientes"].head())
print("Nulos en tabla_telematica por columna (%): solo quedan los estructurales de eléctricos y algunos parciales")
display((cubos["tabla_telematica"].isna().mean() * 100).round(1).to_frame("% nulos").T)
""")

code("""
print("cubo_clientes: segmentos más grandes")
display(cubos["cubo_clientes"].sort_values("n_clientes", ascending=False).head(10))
print("cubo_vehiculos: combinaciones más frecuentes")
display(cubos["cubo_vehiculos"].sort_values("n_relaciones", ascending=False).head(10))
print("cubo_telematica: relaciones con telemetría por clase y motor")
display(cubos["cubo_telematica"].sort_values("n_relaciones", ascending=False).head(15))
""")

code("""
# Ejemplo de uso del cubo: % de relaciones con telemetría por motorización en la clase con más datos
cv = cubos["cubo_vehiculos"]
top_clase = cv.groupby("model_class").n_relaciones.sum().idxmax()
sub = cv[cv.model_class == top_clase].groupby("engine_concept").apply(
    lambda g: np.average(g.pct_con_telematica, weights=g.n_relaciones), include_groups=False)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(sub.index, sub.values, color=BLUE); ax.set_ylabel("% con telemetría"); ax.set_title(f"{top_clase}: % con telemetría por motor")
ax.tick_params(axis="x", rotation=30); plt.tight_layout(); plt.show()
""")

md("""
### 10.1 Columnas que ya no tienen nulos tras el tratamiento

Todas las columnas que tenían nulos en el CSV original y que, tras `clean()`, ya no tienen ninguno. Para cada una: cuántos valores se rellenaron,
cómo, y con qué flag se pueden distinguir los valores reales de los imputados. `UNKNOWN` cuenta como relleno (no es NaN), pero se muestra aparte.
""")

code("""
FLAGS = {
    "engine_concept": "engine_concept_imputed", "fuel_type": "fuel_type_imputed", "model_class": "model_class_imputed",
    "date_of_first_registration": "registration_date_missing", "purchasing_power_per_inhabitant_eur": "socio_missing",
    "population": "socio_missing",
}
SUSTITUTA = {"population": "vía log_population", "date_of_first_registration": "vía vehicle_age_years"}
feats = C.analytic_features(df)
crudo = raw.drop(columns=["Unnamed: 0"], errors="ignore")
filas = []
for c in crudo.columns[crudo.isna().any()]:
    if c not in df or df[c].isna().any():
        continue
    n_nulos = int(crudo[c].isna().sum())
    unknown = int((df[c].astype(str) == "UNKNOWN").sum()) if not pd.api.types.is_numeric_dtype(df[c]) else 0
    filas.append({
        "columna": c,
        "nulos en crudo": n_nulos,
        "% nulos crudo": n_nulos / len(crudo) * 100,
        "rellenados con valor real/estimado": n_nulos - unknown,
        "quedan como UNKNOWN": unknown,
        "método": C.NULL_STRATEGY.get(c, ("", "—"))[1],
        "flag": FLAGS.get(c, "—"),
        "en el análisis": "sí" if c in feats else SUSTITUTA.get(c, "no (excluida)"),
    })
tratadas = pd.DataFrame(filas).set_index("columna").sort_values("% nulos crudo", ascending=False)
pd.set_option("display.max_colwidth", 110)
display(tratadas)
print(f"{len(tratadas)} columnas sin nulos tras el tratamiento; "
      f"{int(tratadas['rellenados con valor real/estimado'].sum()):,} valores recuperados o estimados y "
      f"{int(tratadas['quedan como UNKNOWN'].sum()):,} marcados como UNKNOWN.")
""")

md("""
### 10.2 Columnas de la tabla anterior, escaladas

Las mismas columnas de la 10.1, ya sin nulos, preparadas para modelar. Las **numéricas** se estandarizan con `StandardScaler`
(media 0, desviación 1) sobre la variable que entra en el análisis: `population` como `log_population` y `date_of_first_registration`
como `vehicle_age_years`. Las **categóricas** no se escalan: se codifican *one-hot* (una columna 0/1 por categoría), que ya están en la
misma escala que las numéricas estandarizadas.
""")

code("""
from sklearn.preprocessing import StandardScaler

VARIABLE_ESCALADA = {"population": "log_population", "date_of_first_registration": "vehicle_age_years"}
filas = []
for c in tratadas.index:
    v = VARIABLE_ESCALADA.get(c, c)
    if c not in feats and c not in VARIABLE_ESCALADA:
        filas.append({"columna": c, "variable escalada": "—", "transformación": "no se escala (excluida del análisis)"})
    elif pd.api.types.is_numeric_dtype(df[v]):
        x = df[[v]].to_numpy()
        z = StandardScaler().fit_transform(x).ravel()
        filas.append({"columna": c, "variable escalada": v, "transformación": "StandardScaler",
                      "media original": x.mean(), "desv. original": x.std(),
                      "media escalada": z.mean(), "desv. escalada": z.std(), "mín. escalado": z.min(), "máx. escalado": z.max()})
    else:
        k = df[v].nunique()
        filas.append({"columna": c, "variable escalada": v, "transformación": f"one-hot ({k} columnas 0/1)",
                      "mín. escalado": 0, "máx. escalado": 1})
escaladas = pd.DataFrame(filas).set_index("columna")[[
    "variable escalada", "transformación", "media original", "desv. original",
    "media escalada", "desv. escalada", "mín. escalado", "máx. escalado"]]
display(escaladas.style.format("{:,.2f}", subset=escaladas.columns[2:], na_rep="—"))
""")

# =========================================================================== #
md("""
## 11. Resumen de hallazgos y conclusiones

### ¿Cuáles son las principales características de los clientes?

* **Base de clientes:** 173.456 clientes y 244.519 vehículos. El 73 % de los clientes tiene un único vehículo en el dataset, el 26 % entre 2 y 9, y existe un pequeño grupo de **flotas** (1,1 % de los clientes, ≥10 vehículos, hasta 561) que aporta el 20,6 % de las filas.
* **Edad:** distribución muy centrada en la madurez: 45-54 (21 %), 65+ (20 %), 55-64 (20 %), 35-44 (19 %); los menores de 25 son solo el 4,7 %.
* **Género:** 65 % hombres, 18 % mujeres, 17 % desconocido (más frecuente en usuarios no propietarios).
* **Geografía:** 66 % Suiza y 34 % Austria; el poder adquisitivo por habitante es alto (mediana ≈ 46.800 €) y bimodal.
* **Relación con la marca:** antigüedad mediana de 36 meses (con un pico anómalo exactamente en 36); el 71 % nunca ha tenido financiación/leasing, el 24 % tiene financiación activa.
* **Rol (a nivel fila):** 87 % propietarios (OWNER); los DRIVER (B2B) son el 1,8 %. El 90 % de los clientes tiene algún vehículo actual y el 60 % tiene telemetría en al menos uno. El 81 % declara 1 vehículo conectado y el 10 %, ninguno.
* **Vehículo:** 84 % turismos (grupo P) y 14 % vans/comerciales (T); clases dominantes GLC (15 %), C (12 %), A (11 %). Antigüedad mediana del vehículo 4,7 años (registros 2009-2026, pico 2023-24). Tras recuperar el motor desde el catálogo, la electrificación (mild/plug-in/eléctrico) es el 34 % de las filas y **al menos el 42 % de los clientes** tiene algún vehículo electrificado (cota inferior: el motor sigue desconocido en el 21 % de las filas). La electrificación crece con la edad del cliente (30 % en <25 → 47 % en 55-64) y la proporción de mujeres decrece (23 % en 25-34 → 11 % en 65+).
* **Uso (solo filas con telemetría):** velocidad media 49 km/h, viaje mediano 14 km, ~400 h de funcionamiento; el consumo baja con la electrificación (combustión 8,5 → mild hybrid 7,5 → plug-in 5,9 l/100 km, medianas). El 44 % de los vehículos no tiene visitas a taller por garantía/buena voluntad; el uso intensivo se asocia con más visitas.

### ¿Qué problemas de calidad hemos encontrado y cómo los hemos resuelto?

| Problema | Solución |
|---|---|
| Telemetría ausente en ~52–60 % de las filas (estructural) | No se imputa; **excluido todo el bloque** del analisis; solo flag `has_telematics` para segmentar/describir |
| `engine_concept` / `fuel_type` nulos (44 % / 59 %) dependientes del modelo | Catálogo + reglas en `vehicle_type` (`d`→diésel, EQ→eléctrico, `de`/`e`→híbridos): ~21 % / ~24 % UNKNOWN; resto con flag `*_imputed` |
| Género nulo (17 % de clientes) | Categoría `UNKNOWN` |
| 6,3 % sin variables socioeconómicas; `population = 0` (73); 5 `zipcode` mal formados | Normalizar CP, `population = 0` → NaN, imputación por mediana del país + flag `socio_missing` |
| 427 valores físicamente imposibles (velocidad media > 150, consumo > 30 l/100 km) | → NaN |
| Columnas `*_mode` como texto con listas de bins | Parseo a numérico, lista → NaN + flag; columnas excluidas |
| Pico anómalo de 36 meses de antigüedad (12 % de clientes) | No se modifica; se documenta y se recomienda contrastar con la fuente (se nota en los cubos: la mediana es 36 en casi todos los segmentos) |
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
