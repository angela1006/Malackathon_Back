# Informe EDA – Reto Mercedes-Benz (Malackathon)

Dataset: `datos_reto_mercedes_benz_malackathon.csv` · 302.850 filas × 39 columnas (38 + un índice sin nombre).
Todo el análisis es reproducible: `python -m eda.run_eda` (desde `Backend/`). Tablas en `eda/output/tables/`, gráficos en `eda/output/figures/`, dataset tratado en `eda/output/dataset_limpio.parquet` y la lógica de limpieza en `eda/cleaning.py` (`load_clean()`).

---

## 0. Conclusiones clave

1. **La fila no es un cliente ni un vehículo: es la relación cliente–vehículo** (clave `client_id + vehicle_id`, sin duplicados). Hay 173.456 clientes, 244.519 vehículos y 46.648 vehículos que aparecen en varias filas (distintos clientes/roles).
2. **20,6 % de las filas pertenecen a solo 1.881 clientes-flota** (≥10 filas; máximo 561). Un estadístico a nivel fila está sesgado hacia flotas: la antigüedad media pasa de 43,9 meses (fila) a 33,9 (cliente) y los vehículos conectados de 2,15 a 1,04. Por eso las variables de cliente se describen **a nivel cliente** y las de vehículo/telemetría **a nivel fila**.
3. **La telemetría está presente solo en el 47,8 % de las filas y falta en bloque** (no es ruido aleatorio): depende del rol, de si el vehículo es actual, del año de registro y del tipo de motor. Los eléctricos solo reportan distancia de viaje.
4. **`engine_concept` (44 %) y `fuel_type` (59 %) tienen muchos nulos que dependen del modelo** (p. ej. Citan y Clase B 100 % nulos; GLA y EQC 0 %), no de la antigüedad ni del país.
5. La calidad "dura" es buena: 0 duplicados, 0 contradicciones de atributos entre filas del mismo vehículo, 0 incoherencias motor/combustible, 0 casos de horas de uso mayores que la vida del vehículo. Los problemas reales son: pico anómalo en `months_as_mb_customer` (12 % de clientes con exactamente 36 meses), 427 valores físicamente imposibles de telemetría, 5 códigos postales mal formados y 6,3 % sin datos socioeconómicos.
6. **De las 39 columnas se excluyen 16** (índice, identificadores, metadatos, redundantes, artefactos) y se derivan 3 (`vehicle_age_years`, `has_telematics`, `socio_missing`).

---

## 1. Estructura y unidad de análisis

| Métrica | Valor |
|---|---|
| Filas / columnas | 302.850 / 39 |
| Clientes únicos / vehículos únicos | 173.456 / 244.519 |
| Filas duplicadas exactas / pares (cliente, vehículo) repetidos | 0 / 0 |
| Clientes con >1 vehículo | 46.375 (p90 = 2 filas) |
| Vehículos en >1 fila | 46.648 (104.979 filas, 34,7 %) |
| Clientes-flota (≥10 filas) | 1.881 (20,6 % de las filas) |

* Atributos de **cliente** (constantes por `client_id`, detectado automáticamente): edad, género, país, antigüedad, vehículos conectados, financiación, consentimientos, código postal y variables socioeconómicas.
* Atributos de **vehículo**: modelo, motor, combustible, fecha de registro (0 vehículos con valores contradictorios entre filas).
* La **telemetría varía entre clientes del mismo vehículo** (~9.000 vehículos con valores distintos): se mide por relación cliente–vehículo (uso por usuario), no por vehículo.
* **No se deduplica por `vehicle_id`**: es una relación N:M real (OWNER/USER/LEGACY…). Para contar parque, usar `drop_duplicates("vehicle_id")`.

Gráfico: `01_estructura_cliente_vehiculo.png`.

---

## 2. Estadísticas descriptivas (datos limpios)

Numéricas (cliente → n=173.456; fila → n=302.850, salvo nulos). Tabla completa con percentiles, asimetría, curtosis y outliers IQR: `06_descriptivos_numericos_limpios.csv`.

| Variable | Nivel | Media | Mediana | P25–P75 | Máx | Asimetría | Lectura |
|---|---|---|---|---|---|---|---|
| `months_as_mb_customer` | cliente | 33,9 | 36 | 18–47 | 120 | 0,4 | Pico anómalo en 36 y en 0 |
| `currently_connected_vehicles` | cliente | 1,04 | 1 | 1–1 | 21 | 10,8 | 81 % tiene 1; 10 % tiene 0; flotas hasta 21 |
| `workshop_visits_warranty_goodwill` | fila | 1,75 | 1 | 0–3 | 44 | 2,2 | 44 % con 0 visitas; cola larga |
| `purchasing_power_per_inhabitant_eur` | cliente | 46.709 | 46.845 | 31.518–53.733 | 148.354 | 1,7 | Distribución bimodal |
| `population` (del código postal) | cliente | 18.993 | 7.317 | 3.443–17.273 | 220.794 | 3,8 | Muy asimétrica → log |
| `vehicle_age_years` (derivada) | fila | 5,06 | 4,70 | 2,5–7,1 | 39,7 | 0,7 | Registros 2009-2026 (solo 33 vehículos anteriores) |
| `trip_distance_average` (km) | fila* | 16,2 | 14,0 | 10,2–19,5 | 300 | 7,6 | Cola larga |
| `trip_distance_all_counter` | fila* | 2.352 | 1.593 | 518–3.321 | 89.078 | 4,9 | 6.027 filas con 0 viajes |
| `operating_hours_total` | fila* | 620 | 407 | 137–856 | 26.842 | 4,1 | Log-normal |
| `fuel_consumption_average` (l/100 km) | fila* | 8,47 | 8,11 | 6,6–9,7 | 30 | 1,7 | Tras eliminar 401 imposibles |
| `speed_average` (km/h) | fila* | 48,6 | 49,3 | 41,2–57,2 | 148,8 | −0,5 | Casi normal con cola a ~0 |
| `speed_all_counter` | fila* | 2,5·10⁶ | 1,7·10⁶ | 0,63–3,4·10⁶ | 1,45·10⁹ | 145,8 | Outlier extremo → log |

\* solo filas con telemetría (≈ 39–43 % de las filas, según variable).
`trip_distance_median` y `speed_median` son **discretas** (marcas de bin: 3; 7,5; 15… / 30; 50; 75…), no continuas.

Categóricas (`06_descriptivos_categoricos.csv`):

* **Edad** (cliente): 45-54 (21,0 %), 65+ (20,2 %), 55-64 (19,6 %), 35-44 (18,9 %), 25-34 (15,6 %), <25 (4,7 %).
* **Género** (cliente): hombre 65,0 %, mujer 18,1 %, desconocido 16,9 %.
* **País** (cliente): CH 66,2 %, AT 33,8 %. **Financiación** (cliente): never_active 70,6 %, active_FL_only 23,7 %, was_active 3,3 %, active_FL_and_CASH 2,4 %.
* **Rol** (fila): OWNER 87,3 %, USER 5,8 %, LEGACY 3,7 %, DRIVER (B2B) 1,8 %, TEMP_USER 1,3 %. **Vehículo actual**: 59,5 %.
* **Grupo de producto**: P 84,4 %, T 13,8 %, G 1,4 %, F 0,3 %.
* **Clase de modelo**: GLC 14,6 %, C 11,6 %, A 10,7 %, B/GLA/GLB 9,4 %, V-Class/Vito 9,0 % (42 clases + `UNKNOWN`).
* **Motorización** (con nulos): COMBUSTION 32,5 %, MILD_HYBRID 9,2 %, PLUGIN_HYBRID 7,0 %, ELECTRIC 6,9 %, nulo 44,4 %.

---

## 3. Distribuciones de variables clave

* `06_distribuciones_numericas.png`: histogramas (escala log10(1+x) cuando la asimetría >3; recorte P0,1–P99,9; en población y poder adquisitivo se excluyen los valores imputados para no mostrar un pico artificial).
* `07_distribuciones_categoricas.png`: barras de todas las categóricas.
* `08_anyo_registro.png`: registros por año (2009–2026; pico 2023-24; 2026 parcial hasta el 02-oct).
* `09_meses_cliente_heaping.png`: picos anómalos de antigüedad.
* `10_telematica_por_motorizacion.png`, `11_correlacion_analiticas.png`.

Hallazgos:

* Velocidad media y viaje medio son aproximadamente log-normales/normales; horas de uso y contadores son log-normales con una **masa en 0** (vehículos conectados sin uso).
* Consumo: combustión 9,2 l/100 km (mediana 8,8) > mild hybrid 8,1 (7,7) > plug-in 6,4 (5,8) — coherente con la motorización.
* Correlaciones (Spearman) a tener en cuenta al modelar: `trip_distance_all_counter` ↔ `operating_hours_total` 0,80; ambos ↔ `speed_all_counter` 0,9; `speed_average` ↔ `speed_median` 0,85; `workshop_visits` ↔ contadores/horas 0,4–0,5 (el uso intensivo implica más visitas); `antigüedad_cliente` ↔ `currently_connected_vehicles` 0,4.

---

## 4. Valores nulos

Detalle por columna: `02_tipos_nulos_cardinalidad.csv`, `02_nulos_por_columna.png`, `03_mapa_nulos.png`.

| Bloque | Columnas | % nulos | Patrón | Mecanismo |
|---|---|---|---|---|
| Telemetría | 10 columnas `trip_*`, `speed_*`, `operating_hours`, `fuel_*` | 57,6 – 61,4 | Faltan **en bloque** | Estructural / MAR |
| Motor y combustible | `engine_concept`, `fuel_type` | 44,4 / 59,0 | Depende del modelo | MAR (por `model_class`) |
| Socioeconómico/geo | `zipcode` 5,3 %; resto 6,3 % | 5,3–6,3 | Código postal sin cruzar | Cobertura de la tabla externa |
| Género | `gender_derived` | 11,2 % (filas) · 16,9 % (clientes) | Más en USER/TEMP_USER (36–38 %) que en OWNER (9 %) y DRIVER (4 %) | MAR (por rol) |
| Fecha registro | `date_of_first_registration` | 1,4 | Disperso | Probable MCAR |
| Modelo | `model_class` 1,1 %, `vehicle_type` 0,3 % | | Series sin clase (540, 244, 178…) | Cobertura del catálogo |
| Cliente | `months_as_mb_customer` | 4 filas | Residual | MCAR |

Evidencia sobre la **ausencia de telemetría** (`04_disponibilidad_telematica.png`, `03_telematica_*.csv`):

* Por rol: OWNER 50 %, USER 54 %, LEGACY 13 %, DRIVER 10 %, TEMP_USER 3 %. Por vehículo actual: 58 % frente a 33 %. Por año de registro: ≤2014 10 % → 2021-22 57 % → 2025-26 40 %. País sin efecto (AT 49 % / CH 47 %). No es MCAR.
* **Grupo F: 0 %** de telemetría (936 filas).
* **Eléctricos (20.944 filas): solo tienen distancia de viaje** (`trip_distance_*`, 42–43 %); `operating_hours`, `speed_*` y `fuel_consumption` son 100 % nulos. Esquema de datos distinto, no un fallo.

Evidencia sobre `engine_concept` (`03_engine_nulo_por_clase_modelo.csv`): GLA 0 %, EQC 0 %, E-Class 29 %, C-Class 59 %, Sprinter 81 %, Clase B y Citan 100 %. No depende del país (43,6 % AT / 44,8 % CH) ni de si es el vehículo actual (43 % / 46 %). Relación con `fuel_type`: siempre que hay `fuel_type` hay `engine_concept` (0 excepciones); al revés, 44.355 filas tienen motor pero no combustible (43.915 son COMBUSTION).

---

## 5. Inconsistencias y tratamiento propuesto

Tabla completa: `04_inconsistencias.csv`. Resumen:

| Problema | Afectados | Tratamiento propuesto | Justificación |
|---|---|---|---|
| Columna índice `Unnamed: 0` | 100 % | Eliminar | Artefacto de exportación |
| Duplicados exactos / pares repetidos | 0 | — | — |
| `vehicle_id` en varias filas | 104.979 filas | **No deduplicar**; analizar a 3 niveles (fila, cliente, vehículo) | Relación N:M legítima |
| Clientes-flota (≥10 filas) | 62.280 filas (20,6 %) | Reportar a nivel cliente; sensibilidad con/sin flotas | Sesgan medias y estimadores |
| `zipcode` con formato inválido (`A-2100`, `CH-6340`, `Ottenbach`, `55605`) | 5 | Quitar prefijo país; resto → NaN | AT/CH usan 4 dígitos |
| Código postal sin datos socioeconómicos | 2.879 | Imputar mediana por país + flag `socio_missing` | Mediana robusta a la cola; flag preserva la señal |
| `population = 0` en un CP con clientes | 73 | → NaN y reimputar | Imposible con clientes residentes |
| Total socioeconómico faltante | 19.087 (6,3 %) | Idem | AT 8,3 % vs CH 5,1 % → imputar **por país** |
| `trip_distance_mode`/`speed_mode` como texto con listas de bins | 7.049 / 1.223 | Parsear a numérico; lista → NaN + flag; excluir columna | Lista = empate entre todos los bins, no hay moda real |
| `speed_average` > 150 km/h | 26 | → NaN | Imposible como media |
| `fuel_consumption_average` > 30 l/100 km | 401 | → NaN | Implausible; picos en 48-50 |
| `trip_distance_average` > 100 km | 106 | Mantener; winsorizar al modelar | Posible uso de furgoneta/largo recorrido |
| Telemetría con 0 viajes / 0 horas | 6.027 / 140 | Mantener, excluir al estimar hábitos de conducción | Vehículo conectado sin uso (información válida) |
| `speed_average` < 5 km/h | 1.784 | Mantener; excluir al estimar hábitos de conducción | Verificado: mediana de 0,14 h de uso (89 % con <1 h); vehículos casi sin uso |
| `months_as_mb_customer` = 36 | 21.179 clientes (12,2 %) | **No modificar**; documentar y evitar usarla como antigüedad real sin contrastar | Pico muy superior a los meses vecinos (36: 29.255 filas; 37: 4.568): posible valor por defecto/migración |
| `months_as_mb_customer` = 0 | 7.254 clientes (4,2 %) | Idem | Mezcla de clientes nuevos y valores por defecto |
| `months_as_mb_customer` nulo | 4 | Mediana | Residual |
| `is_current_vehicle=False` con telemetría | 40.452 | Mantener | Histórico plausible |
| `currently_connected_vehicles=0` con telemetría en la fila | 10.259 | Mantener, no cruzar ambas | Métricas con definición distinta (cliente vs. vehículo) |
| `model_class` nulo | 3.369 | Categoría `UNKNOWN` | Las series afectadas no tienen clase en ningún otro registro, no se puede inferir |
| `vehicle_type`, `gender_derived`, `engine_concept`, `fuel_type` nulos | 922 / 33.923 / 134.390 / 178.745 | Categoría `UNKNOWN` | Ausencia informativa; imputar por moda crearía información falsa |
| Telemetría ausente | ~52 % | **No imputar**; flag `has_telematics`; analizar solo filas con dato; los modelos de árbol pueden tratar NaN directamente | Ausencia estructural (rol, eléctricos, grupo F…): imputar con media sesgaría las distribuciones |
| `date_of_first_registration` nula | 4.182 | Mantener NaN en `vehicle_age_years` | 1,4 %, sin patrón; imputar no aporta |

Comprobaciones superadas (0 incidencias): `ELECTRIC` ⇒ `fuel_type = ELECTRIC`; `COMBUSTION` sin componente eléctrico; `PLUGIN_HYBRID` siempre con componente eléctrico; `operating_hours_total` ≤ horas desde el registro; ninguna fecha posterior a la referencia (2026-10-02); `role` ⇔ `role_type`.

---

## 6. Columnas excluidas (metadatos, identificadores y redundantes)

Tabla con motivos: `05_columnas_excluidas.csv`. Se **conservan en `dataset_limpio.parquet`** (claves y trazabilidad), pero no entran en el conjunto analítico (`cleaning.analytic_features()`).

| Columna | Tipo | Motivo |
|---|---|---|
| `Unnamed: 0` | Metadato | Índice de pandas 0..n-1 |
| `client_id` | Identificador | Hash SHA-256, 173k valores, sin significado; solo clave de agregación |
| `vehicle_id` | Identificador | Hash SHA-256, 245k valores; solo clave de agregación |
| `zipcode` | Identificador geográfico | ~4,6k niveles y es la clave con la que se enriquecieron las variables socioeconómicas, que ya capturan su información |
| `vehicle_type` | Identificador de producto | 1.436 denominaciones comerciales, redundante con `model_class` + `engine_concept` |
| `model_series` | Código interno | Nominal aunque sea entero (tratarlo como número induce un orden falso); 82 de 88 series mapean a una sola `model_class` |
| `date_of_first_registration` | Fecha cruda | Sustituida por `vehicle_age_years` |
| `role_type` | Redundante | Función exacta de `role` (DRIVER=B2B, resto B2C) |
| `consent_phone/epost/post_communication_channel` | Metadato de contacto | Flags RGPD: indican si se puede contactar, no el comportamiento; se conservan para activación de campañas |
| `purchasing_power_millions_eur` | Derivada | = `per_inhabitant` × `population` (cociente 0,9997–1,0007) |
| `number_of_households` | Redundante | Spearman 0,998 con `population` |
| `purchasing_power_per_household_eur` | Redundante | Spearman 0,956 con `per_inhabitant` (ratio ≈ 2,24 = tamaño medio de hogar) |
| `trip_distance_mode`, `speed_mode` | Artefacto técnico | Marcas de bin y listas multimodales; `speed_mode` solo coincide con `speed_median` en el 43 % de filas |

Umbral usado para redundancia numérica: |ρ Spearman| ≥ 0,95 (gráfico `05_corr_socioeconomico.png`).
Los contadores (`trip_distance_all_counter`, `operating_hours_total`, `speed_all_counter`, ρ ≈ 0,8–0,9) quedan por debajo del umbral y miden magnitudes distintas, **se mantienen**, pero conviene elegir uno o aplicar regularización/PCA al modelar.

**Variables analíticas finales** (26): edad, género, país, antigüedad, vehículos conectados, financiación, rol, `model_class`, `product_group`, `engine_concept`, `fuel_type`, `is_current_vehicle`, `vehicle_age_years`, `workshop_visits_warranty_goodwill`, `purchasing_power_per_inhabitant_eur`, `population`, 8 de telemetría y los flags `has_telematics`, `socio_missing`.

---

## 7. Siguientes pasos recomendados

1. Decidir si el análisis de negocio se hace a nivel cliente (agregando vehículos) o relación; definir tratamiento de flotas.
2. Confirmar con el organizador el significado de los picos de `months_as_mb_customer` (36 y 0) y de los nulos de `engine_concept` por modelo (¿se puede recuperar desde `vehicle_type`?).
3. Para modelado: transformación log a población, contadores y horas; winsorizar P1–P99 en telemetría; categóricas con `UNKNOWN`; evitar `model_series`/`zipcode` como numéricas.


