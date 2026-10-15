# Valoración y Tratamiento de Datos Nulos en `gender_derived`

Documento explicativo sobre el análisis estadístico, diagnóstico de ausencia y las modificaciones introducidas en el código de la carpeta `eda/`.

---

## 1. Contexto y Objetivos

La columna `gender_derived` contiene el género atribuido al cliente (`male`, `female` o ausente). Se trata de un atributo **derivado** (inferido a partir de tratamientos de cortesía como *Herr/Frau* o nombres de pila durante el registro).

El objetivo ha sido:
1. **Analizar la magnitud y distribución** de los datos nulos (tanto por fila de relación como por cliente único).
2. **Determinar el mecanismo de ausencia** (si es aleatorio puro MCAR o dependiente de otros atributos MAR).
3. **Evaluar las alternativas de tratamiento** (eliminación vs. imputación por moda vs. categoría explícita).
4. **Implementar en código** las funciones de diagnóstico y limpieza.

---

## 2. Modificaciones realizadas en el Código

### A. Mejora en la resolución de rutas de datos (`eda/cleaning.py`)

* **Archivo modificado:** [`eda/cleaning.py`](file:///c:/Users/Paqui/Downloads/Malackathon_Back/eda/cleaning.py#L21-L26)
* **Motivo:** El archivo original del dataset (`datos_reto_mercedes_benz_malackathon.csv`) se encontraba ubicado en la carpeta padre (`Downloads/`). La lista de rutas candidatas no lo contemplaba por defecto.
* **Cambio aplicado:**
  ```python
  _CANDIDATES = [
      BACKEND_DIR / CSV_NAME,
      BACKEND_DIR.parent / CSV_NAME,               # <-- Añadida ruta padre
      BACKEND_DIR.parent / "Frontend" / CSV_NAME,
  ]
  DEFAULT_DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
  ```
* **Impacto:** Permite que cualquier módulo o notebook cargue automáticamente el dataset sin necesidad de pasarle rutas manuales.

---

### B. Creación del módulo de diagnóstico y limpieza (`eda/analizar_gender.py`)

* **Archivo creado:** [`eda/analizar_gender.py`](file:///c:/Users/Paqui/Downloads/Malackathon_Back/eda/analizar_gender.py)
* **Funcionalidades principales:**

#### 1. `evaluar_nulos_gender(df: pd.DataFrame) -> dict[str, Any]`
Realiza el cálculo estadístico exhaustivo:
* **Conteo y porcentajes** a dos niveles (filas y clientes únicos).
* **Distribución de clases** (`male`, `female`, `NULL`).
* **Cruce con variables clave** para identificar patrones:
  * `role`: proporción de nulos por rol en el vehículo.
  * `age_range`: distribución por grupo de edad.
  * `country_code`: comparativa Austria (`AT`) vs. Suiza (`CH`).
  * `finance_leasing_status`: relación con contratos financieros.
  * `months_as_mb_customer`: comparativa de antigüedad de clientes con dato vs. sin dato.
  * `purchasing_power_per_inhabitant_eur`: nivel adquisitivo medio/mediano.

#### 2. `tratar_gender_derived(df, fill_value="UNKNOWN", add_missing_indicator=True, inplace=False) -> pd.DataFrame`
Aplica el tratamiento estándar recomendado:
* Reemplaza los valores `NaN` por una categoría explícita (`'UNKNOWN'` por defecto).
* Genera opcionalmente una columna booleana `gender_derived_missing` para preservar la señal de ausencia en modelos predictivos posteriores.

#### 3. `imprimir_informe(res: dict[str, Any]) -> None`
Formatea los resultados y muestra un resumen ejecutivo por terminal.

#### 4. Bloque ejecutable `main()`
Permite ejecutar el archivo directamente desde consola para validar el flujo completo.

---

## 3. Hallazgos del Análisis Estadístico

| Métrica | Nivel Fila (Relación Cliente–Vehículo) | Nivel Cliente Único (`client_id`) |
|---|---|---|
| **Total registros** | 302.850 | 173.456 |
| **Registros nulos** | **33.923** | **29.398** |
| **Porcentaje de nulos** | **11,20 %** | **16,95 %** |

### Distribución por género a nivel cliente:
* **`male`:** 112.726 clientes (64,99 %)
* **`female`:** 31.332 clientes (18,06 %)
* **`NULL` / Desconocido:** 29.398 clientes (16,95 %)

*(Sobre clientes con género conocido, el 78,25 % son hombres y el 21,75 % mujeres).*

---

## 4. Diagnóstico del Mecanismo de Ausencia: MAR (*Missing at Random*)

Los datos confirman de forma rotunda que la ausencia **no es aleatoria (MCAR)**, sino que está condicionada por el tipo de usuario y el canal de alta:

1. **Por Rol en el vehículo (`role`):**
   * `TEMP_USER`: **38,15 %** nulos.
   * `USER`: **36,23 %** nulos.
   * `LEGACY`: **11,91 %** nulos.
   * `OWNER`: **9,25 %** nulos.
   * `DRIVER`: **4,09 %** nulos.
   * *Explicación:* Los conductores secundarios o temporales de la app *Mercedes me* no suelen completar un registro formal con tratamiento de cortesía.

2. **Por Antigüedad como cliente (`months_as_mb_customer`):**
   * Clientes con género conocido: mediana de **36,0 meses** (media: 37,5).
   * Clientes con género nulo: mediana de **17,0 meses** (media: 16,1).
   * *Explicación:* Los nulos corresponden a clientes sustancialmente más recientes, vinculados a altas digitales rápidas.

3. **Por Edad (`age_range`):**
   * Menores de 25 años (`<25`): **40,51 %** nulos.
   * Mayores de 65 años (`65+`): **27,08 %** nulos.
   * Grupo 55–64 años: **8,45 %** nulos (mínimo de la serie).
   * *Explicación:* Patrón en "U" donde los segmentos jóvenes registran cuentas con menos campos opcionales.

4. **Por Financiación (`finance_leasing_status`):**
   * Sin financiación (`never_active`): **19,00 %** nulos.
   * Financiación activa y contado (`active_FL_and_CASH`): **6,02 %** nulos.
   * *Explicación:* La formalización de créditos o leasing exige identificación oficial rigurosa.

5. **Por País (`country_code`):**
   * Austria (`AT`): **21,54 %** nulos.
   * Suiza (`CH`): **14,60 %** nulos.

---

## 5. Justificación de la Decisión de Tratamiento

* **¿Por qué NO eliminar filas (*drop*)?**  
  Eliminar el 17 % de los clientes sesgaría gravemente los análisis contra la gente joven, usuarios de app y clientes recientes.
* **¿Por qué NO imputar por moda (`male`)?**  
  Elevaría la proporción masculina a más del 81 %, inflando artificialmente el perfil de hombre y borrando un patrón de comportamiento real.
* **Decisión adoptada (`UNKNOWN` + flag):**  
  Conservar el 100 % de los datos creando la categoría explícita `UNKNOWN`. Representa fielmente la incertidumbre y permite que los modelos analíticos y de machine learning aprendan de este segmento sin inventar información ficticia.

---

## 6. Instrucciones de Uso

### Ejecución directa desde terminal
```powershell
py -m eda.analizar_gender
```

### Importación en otros scripts o notebooks
```python
import pandas as pd
from eda.analizar_gender import evaluar_nulos_gender, tratar_gender_derived

# Carga
df = pd.read_csv("datos_reto_mercedes_benz_malackathon.csv")

# Diagnóstico
informe = evaluar_nulos_gender(df)
print("Pct nulos en clientes:", informe["resumen_clientes"]["pct_nulos"])

# Limpieza
df_limpio = tratar_gender_derived(df, fill_value="UNKNOWN", add_missing_indicator=True)
```
