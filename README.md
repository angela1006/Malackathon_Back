# Backend – Reto Mercedes-Benz (Malackathon)

Análisis de datos y segmentación de clientes de Mercedes-Benz a partir del CSV del reto
(`datos_reto_mercedes_benz_malackathon.csv`, ~303.000 relaciones cliente–vehículo de Suiza y Austria).

## Qué hemos hecho

### 1. EDA y limpieza (`eda/`)

- **Análisis exploratorio:** estructura, tipos, nulos, inconsistencias, estadísticos descriptivos y distribuciones.
- **Columnas excluidas:** identificadores, consentimientos, variables redundantes y todo el bloque de telemetría (falta en ~55 % de las
  filas); de la telemetría solo se conserva el flag `has_telematics`.
- **Tratamiento de nulos:**
  - `fuel_type` / `engine_concept`: catálogo por `vehicle_type` + reglas por nombre (d, CDI, EQ, e, de…); el resto queda `UNKNOWN` con flag `*_imputed`.
  - `model_class`: catálogo `model_series` → `vehicle_type` + reglas por nombre (GLC, GLB, C…).
  - `date_of_first_registration`: predicción con gradient boosting acotada al rango de cada serie + flag `registration_date_missing`.
  - Variables socioeconómicas: mediana por país + flag `socio_missing`.
  - `gender_derived`: categoría `UNKNOWN` (se probó a predecirlo y no superaba la línea base).
- **Variables combinadas:** `personas_por_hogar`, `log_population`, `vehicle_age_years`.
- **Escalado:** las numéricas con `StandardScaler` y las categóricas con one-hot (sección 10.2 del notebook).
- **Cubos:** tablas agregadas por cliente, vehículo y segmento (`cubes.py`).

### 2. Segmentación con K-Means (`clustering/`)

- Una fila por cliente (~173.000) con 7 variables: edad, meses como cliente, nº de vehículos, antigüedad de los vehículos,
  % electrificados, % gama premium y leasing activo.
- Pipeline de scikit-learn: `log1p(n_vehiculos)` → `StandardScaler` → `KMeans`.
- Se prueban k = 2…10 y se elige **k = 5** con el método del codo, el Silhouette Score y el tamaño mínimo de cada cluster.

| Cluster | Segmento | % clientes |
|---|---|---|
| 0 | Particular estable | 36,2 % |
| 1 | Multi-vehículo fiel | 9,4 % |
| 2 | Leasing electrificado | 19,1 % |
| 3 | Senior electrificado | 17,7 % |
| 4 | Premium tradicional | 17,6 % |

## Estructura

```
Backend/
├── eda/
│   ├── cleaning.py              # limpieza, imputaciones y features
│   ├── cubes.py                 # tablas agregadas (cubos)
│   ├── run_eda.py               # genera tablas y figuras en eda/output/
│   ├── build_notebook.py        # genera y ejecuta el notebook del EDA
│   ├── EDA_Mercedes_Benz.ipynb  # entregable del EDA
│   └── output/                  # datasets limpios, tablas, figuras y cubos
├── clustering/
│   ├── pipeline.py              # features por cliente + pipeline K-Means
│   ├── build_notebook.py        # genera y ejecuta el notebook de clustering
│   ├── Clustering_KMeans.ipynb  # entregable del clustering
│   └── output/                  # modelo (.joblib), clientes segmentados y perfiles
├── main.py                      # API FastAPI (esqueleto)
└── requirements-eda.txt
```

## Cómo ejecutarlo

Colocar el CSV del reto en `Backend/` (no se sube al repositorio) y, desde `Backend/`:

```bash
pip install -r requirements-eda.txt

python -m eda.run_eda               # tablas y figuras del EDA
python -m eda.cubes                 # cubos
python -m eda.build_notebook        # notebook del EDA ejecutado
python -m clustering.build_notebook # notebook de clustering ejecutado + modelo en clustering/output/
```

Los notebooks se generan desde `build_notebook.py`: los cambios se hacen ahí, no en el `.ipynb`. La opción `--no-exec` genera el
notebook sin ejecutarlo, así que sale sin gráficas ni tablas.
