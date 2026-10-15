"""Features por cliente y pipeline K-Means (escalado + modelo).

    from clustering import pipeline as P
    X_df = P.client_features()               # 1 fila por cliente
    modelos = P.run_kmeans_range(X_df[P.FEATURES], range(2, 11))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from eda import cleaning as C
from eda import cubes

AGE_MIDPOINT = {"<25": 21, "25-34": 30, "35-44": 40, "45-54": 50, "55-64": 60, "65+": 70}
# Prefijos (algunos nombres traen la "é" de Coupé mal codificada en el CSV)
PREMIUM_PREFIXES = ("S-Class", "E-Class", "GLE-", "M-/GL-/GLE-", "G-Class", "EQE-", "EQS-", "AMG",
                    "Maybach", "SL-Class", "SLR", "SLS", "CLS-", "CL-Class")
MAX_VEHICULOS = 10       # flotas: a partir de 10 vehiculos se tratan igual
MAX_EDAD_VEHICULO = 20   # años; por encima son clasicos muy escasos

FEATURES = ["edad_cliente", "months_as_mb_customer", "n_vehiculos", "edad_vehiculo_mediana",
            "pct_electrificados", "pct_premium", "usa_leasing"]
LOG_FEATURES = ["n_vehiculos"]
# Se conservan para describir los clusters, pero no entran en el modelo
DESCRIPTIVE_COLUMNS = ["client_id", "age_range", "gender_derived", "country_code", "rol_principal",
                       "finance_leasing_status", "segmento", "tiene_telematica"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Nombres de negocio con las medias por cluster del modelo k=5
CLUSTER_NAMES: dict[int, str] = {
    0: "Particular estable",
    1: "Multi-vehículo fiel",
    2: "Leasing electrificado",
    3: "Senior electrificado",
    4: "Premium tradicional",
}
CLUSTER_TAGLINE: dict[int, str] = {
    0: "Un solo vehículo, baja electrificación, perfil medio de la base.",
    1: "Varios vehículos o mini-flota, mucha antigüedad con la marca y alta telemetría.",
    2: "Casi siempre con leasing activo, flota joven y alta cuota de híbridos/eléctricos.",
    3: "Clientes mayores con parque muy electrificado y vehículos relativamente nuevos.",
    4: "Cartera mayoritariamente de gama alta (S/E/GLE/EQS…), electrificación moderada.",
}
FEATURE_LABELS: dict[str, str] = {
    "edad_cliente": "Edad (punto medio del rango)",
    "months_as_mb_customer": "Meses como cliente MB",
    "n_vehiculos": "N.º vehículos (tope 10)",
    "edad_vehiculo_mediana": "Antigüedad vehículos (años)",
    "pct_electrificados": "Cuota electrificados (0-1)",
    "pct_premium": "Cuota gama premium (0-1)",
    "usa_leasing": "Tiene leasing activo (0/1)",
}


def client_features(df_clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de clientes (eda.cubes.tabla_clientes) + las features del modelo.

    Parte del dataset sin columnas con NaN: ninguna feature puede salir de una columna con nulos."""
    df = C.drop_null_columns(C.load_clean() if df_clean is None else df_clean)
    tc = cubes.tabla_clientes(df).set_index("client_id")
    premium = df["model_class"].astype(str).str.startswith(PREMIUM_PREFIXES)
    tc["pct_premium"] = premium.groupby(df["client_id"]).mean()
    tc["pct_electrificados"] = tc["n_electrificados"] / tc["n_vehiculos"]
    tc["edad_cliente"] = tc["age_range"].map(AGE_MIDPOINT)
    tc["usa_leasing"] = tc["finance_leasing_status"].str.startswith("active").astype(int)
    tc["n_vehiculos"] = tc["n_vehiculos"].clip(upper=MAX_VEHICULOS)
    tc["edad_vehiculo_mediana"] = tc["edad_vehiculo_mediana"].clip(upper=MAX_EDAD_VEHICULO)
    return tc.reset_index()[DESCRIPTIVE_COLUMNS + FEATURES]


def build_pipeline(k: int, random_state: int = 42) -> Pipeline:
    """log1p en n_vehiculos -> estandarizar todo -> K-Means."""
    log = ColumnTransformer([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), LOG_FEATURES)],
                            remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([
        ("log", log),
        ("escalado", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=random_state)),
    ])


def run_kmeans_range(X: pd.DataFrame, ks=range(2, 11), random_state: int = 42) -> dict[int, Pipeline]:
    """Un pipeline ajustado por cada k: {k: pipeline}. El modelo está en pipeline['kmeans']."""
    return {k: build_pipeline(k, random_state).fit(X) for k in ks}


def elbow_table(modelos: dict[int, Pipeline]) -> pd.DataFrame:
    """Inercia por k, mejora respecto al k anterior y distancia de cada punto a la recta (k_min, k_max).

    El codo es el k más alejado de esa recta (criterio "kneedle"): a partir de ahí añadir clusters reduce poco la inercia."""
    t = pd.DataFrame({"inercia": {k: m["kmeans"].inertia_ for k, m in modelos.items()}}).sort_index()
    t.index.name = "k"
    t["mejora vs k-1 (%)"] = -t["inercia"].pct_change() * 100
    x = (t.index - t.index.min()) / (t.index.max() - t.index.min())
    y = (t["inercia"] - t["inercia"].min()) / (t["inercia"].max() - t["inercia"].min())
    t["distancia a la recta"] = (1 - x) - y
    return t


def elbow_k(modelos: dict[int, Pipeline]) -> int:
    return int(elbow_table(modelos)["distancia a la recta"].idxmax())


def silhouette_scores(modelos: dict[int, Pipeline], X: pd.DataFrame,
                      sample_size: int = 10_000, random_state: int = 42) -> pd.Series:
    """Silhouette Score de cada k, en el espacio ya escalado (donde K-Means mide distancias).

    Con ~173k clientes el cálculo completo es inviable (crece con el cuadrado de las filas), así que se usa una
    muestra fija: los mismos clientes para todos los k, para que los scores sean comparables entre sí."""
    from sklearn.metrics import silhouette_score

    pipe = next(iter(modelos.values()))
    Xt = np.asarray(pipe[:-1].transform(X))
    n = min(sample_size, len(Xt))
    idx = np.random.default_rng(random_state).choice(len(Xt), size=n, replace=False)
    Xs = Xt[idx]
    scores = {k: silhouette_score(Xs, m["kmeans"].labels_[idx]) for k, m in modelos.items()}
    return pd.Series(scores, name="silhouette").rename_axis("k")


def k_selection_table(modelos: dict[int, Pipeline], codo: pd.DataFrame, sil: pd.Series) -> pd.DataFrame:
    """Métricas por k para elegir y justificar el número de clusters (codo + silhouette + tamaños)."""
    runs = runs_summary(modelos)
    t = runs.join(codo[["mejora vs k-1 (%)", "distancia a la recta"]], how="left")
    t["silhouette (muestra 10k)"] = sil
    t = t.rename(columns={"cluster menor (%)": "cluster más pequeño (%)"})
    return t.sort_index()


def choose_k(k_codo: int, sil: pd.Series, candidatos: tuple[int, ...] = (4, 5, 6)) -> int:
    """k del modelo final: el del codo si está entre los candidatos; si no, el de mayor silhouette en candidatos."""
    if k_codo in candidatos:
        return k_codo
    sub = sil.loc[[k for k in candidatos if k in sil.index]]
    return int(sub.idxmax())


def label_clients(pipe: Pipeline, X_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna columna `cluster` (0..k-1) a cada fila de client_features."""
    out = X_df.copy()
    out["cluster"] = pipe.predict(X_df[FEATURES])
    return out


def cluster_distribution(labels: np.ndarray | pd.Series) -> pd.DataFrame:
    s = pd.Series(labels, name="cluster")
    t = s.value_counts().sort_index().rename("n_clientes").to_frame()
    t["pct_clientes"] = t["n_clientes"] / t["n_clientes"].sum() * 100
    t["pct_acumulado"] = t["pct_clientes"].cumsum()
    return t


def profile_numeric(clientes: pd.DataFrame) -> pd.DataFrame:
    """Medias de las features del modelo por cluster."""
    return clientes.groupby("cluster", observed=True)[FEATURES].mean()


def profile_zscores(clientes: pd.DataFrame) -> pd.DataFrame:
    """Desviación en sigmas respecto a la media global (qué destaca en cada cluster)."""
    prof = profile_numeric(clientes)
    std = clientes[FEATURES].std().replace(0, np.nan)
    return ((prof - clientes[FEATURES].mean()) / std).round(2)


def profile_categorical(clientes: pd.DataFrame, column: str, top: int = 4) -> pd.DataFrame:
    """Distribución (%) de una variable descriptiva dentro de cada cluster."""
    s = clientes[column].astype(str)
    t = pd.crosstab(clientes["cluster"], s, normalize="index") * 100
    cols = list(t.sum(axis=0).sort_values(ascending=False).head(top).index)
    return t.loc[:, cols].round(1)


def profile_summary_table(clientes: pd.DataFrame) -> pd.DataFrame:
    """Tabla exportable: tamaño, nombre y medias clave por cluster."""
    dist = cluster_distribution(clientes["cluster"])
    num = profile_numeric(clientes)
    rows = []
    for c in num.index:
        rows.append({
            "cluster": int(c),
            "nombre_segmento": CLUSTER_NAMES.get(int(c), f"Cluster {c}"),
            "n_clientes": int(dist.loc[c, "n_clientes"]),
            "pct_clientes": round(dist.loc[c, "pct_clientes"], 2),
            **{FEATURE_LABELS[k]: round(num.loc[c, k], 2) for k in FEATURES},
            "resumen": CLUSTER_TAGLINE.get(int(c), ""),
        })
    return pd.DataFrame(rows)


def runs_summary(modelos: dict[int, Pipeline]) -> pd.DataFrame:
    rows = []
    for k, pipe in modelos.items():
        km = pipe["kmeans"]
        sizes = np.bincount(km.labels_, minlength=k) / len(km.labels_) * 100
        rows.append({"k": k, "inercia": km.inertia_, "iteraciones": km.n_iter_,
                     "cluster menor (%)": sizes.min(), "cluster mayor (%)": sizes.max()})
    return pd.DataFrame(rows).set_index("k")
