# Valoración y Tratamiento de Datos Nulos en `date_of_first_registration`

Documento explicativo sobre el análisis estadístico, patrones de ausencia y las funciones implementadas en [`eda/analizar_registration.py`](file:///c:/Users/Paqui/Downloads/Malackathon_Back/eda/analizar_registration.py).

---

## 1. Contexto y Objetivos

La columna `date_of_first_registration` indica la fecha en la que el vehículo fue matriculado por primera vez. Es un atributo puramente **a nivel de vehículo** (`vehicle_id`) y abarca un rango temporal desde 1987 hasta octubre de 2026.

El objetivo de este módulo es:
1. **Evaluar la magnitud de nulos** tanto por fila (relación cliente–vehículo) como por vehículo único.
2. **Verificar la consistencia interna:** comprobar si un mismo vehículo tiene fechas contradictorias entre sus distintas filas.
3. **Identificar el patrón de ausencia:** analizar si los nulos son aleatorios (MCAR) o si dependen de la generación/clase del vehículo.
4. **Transformar la variable para modelado:** convertir la fecha absoluta en la variable analítica interpretable `vehicle_age_years` (antigüedad en años).

---

## 2. Magnitud de Nulos y Consistencia

| Métrica | Nivel Fila (Relación Cliente–Vehículo) | Nivel Vehículo Único (`vehicle_id`) |
|---|---|---|
| **Total registros** | 302.850 | 244.519 |
| **Registros nulos** | **4.182** | **3.909** |
| **Porcentaje de nulos** | **1,38 %** | **1,60 %** |
| **Inconsistencias entre filas del mismo coche** | — | **0** (100 % coherente) |

> **Comprobación de calidad superada:** En todos los vehículos que aparecen en múltiples filas (cambios de dueño o múltiples conductores), la fecha de matriculación es siempre idéntica. No existe ningún vehículo con fechas contradictorias ni mezclas de fecha con nulo.

---

## 3. Mecanismo de Ausencia: ¿MCAR o MAR?

A primera vista, la ausencia parece puramente aleatoria (MCAR), pero el análisis por modelo revela un matiz estructural:

### A. Perfectamente equilibrado geográficamente (apariencia MCAR)
* **Austria (`AT`):** 1,37 % nulos (1.503 de 109.972).
* **Suiza (`CH`):** 1,39 % nulos (2.679 de 192.878).
* No hay sesgo entre países; el sistema de registro de ambos reporta la misma tasa residual.

### B. Concentración en modelos descatalogados / legacy (patrón MAR)
Al desagregar por clase de vehículo (`model_class`), se observa que los nulos no se reparten de forma homogénea:

| Clase de Modelo | % Nulos en Fecha | Estado del Modelo / Gama |
|---|---|---|
| **SLR McLaren** | **100,00 %** | Descatalogado (superdeportivo 2003–2010) |
| **CLK-Class** | **94,12 %** | Descatalogado (sustituido por Clase E Coupé en 2010) |
| **M-Class** | **91,67 %** | Descatalogado (renombrado a GLE en 2015) |
| **SLK-Class** | **76,89 %** | Descatalogado (renombrado a SLC y finalizado en 2020) |
| **Viano / Vito antigua** | **66,83 %** | Descatalogado (sustituido por Clase V en 2014) |
| **Maybach (57/62)** | **66,67 %** | Descatalogado (gama previa a Mercedes-Maybach) |
| **GLC-Class** | **0,48 %** | **Gama moderna activa (volumen principal)** |
| **A-Class** | **1,08 %** | **Gama moderna activa (volumen principal)** |
| **C-Class** | **1,41 %** | **Gama moderna activa (volumen principal)** |

**Conclusión:** Los nulos proceden principalmente de **migraciones de bases de datos antiguas o vehículos históricos/descatalogados** cuyos registros originales de matriculación no se digitalizaron en el sistema telemático moderno. En modelos de producción actual, la tasa de nulos es inferior al 1 %.

### C. Correlación con la telemetría
* Vehículos con fecha informada: **48,23 %** tienen telemetría disponible.
* Vehículos con fecha nula: solo el **15,28 %** tienen telemetría.
* Esto confirma que los vehículos sin fecha corresponden mayoritariamente a unidades antiguas desconectadas del ecosistema digital.

---

## 4. Transformación y Tratamiento Recomendado

### ¿Por qué no usar la fecha cruda en modelos predictivos?
Una fecha absoluta (ej. `2018-04-12`) no generaliza en Machine Learning: un modelo entrenado hoy quedará obsoleto en el futuro conforme los coches envejezcan.

### Variable derivada: `vehicle_age_years`
Se toma como referencia la fecha máxima observada en el dataset (`2026-10-02`) y se calcula la antigüedad en años:

$$\text{vehicle\_age\_years} = \frac{\text{fecha\_referencia} - \text{date\_of\_first\_registration}}{365.25}$$

* **Mediana:** 4,70 años.
* **Media:** 5,06 años.
* **Rango:** 0,0 a 39,72 años.

### Decisión sobre los nulos de `vehicle_age_years`:
1. **Modelos basados en árboles (LightGBM, XGBoost, CatBoost):**  
   **Mantener `NaN`** y añadir el flag `registration_date_missing`. Los árboles gestionan nativamente los valores nulos encontrando la partición óptima sin necesidad de inventar datos.
2. **Modelos lineales / distancias (Regresión Logística, KMeans, PCA):**  
   Imputar por la **mediana de la clase de modelo (`model_class`)** preservando la coherencia por segmento, junto con el flag `registration_date_missing`.

---

## 5. Código Implementado ([analizar_registration.py](file:///c:/Users/Paqui/Downloads/Malackathon_Back/eda/analizar_registration.py))

El archivo implementa:
* `evaluar_nulos_registration(df)`: diagnóstico estadístico completo.
* `tratar_date_of_first_registration(df, impute_age_by_model=False, add_missing_indicator=True)`: parseo, derivación de antigüedad y tratamiento de nulos.
* `imprimir_informe(res)`: visualización en terminal.

### Ejecución directa desde terminal
```powershell
py -m eda.analizar_registration
```

### Uso en scripts o notebooks
```python
import pandas as pd
from eda.analizar_registration import evaluar_nulos_registration, tratar_date_of_first_registration

# 1. Carga de datos
df = pd.read_csv("datos_reto_mercedes_benz_malackathon.csv")

# 2. Diagnóstico
metricas = evaluar_nulos_registration(df)
print("Nulos en vehículos:", metricas["resumen_vehiculos"]["pct_nulos"], "%")

# 3. Tratamiento: deriva vehicle_age_years y crea registration_date_missing
df_tratado = tratar_date_of_first_registration(df, add_missing_indicator=True)
```
