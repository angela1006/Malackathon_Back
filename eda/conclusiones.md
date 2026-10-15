# Conclusiones del EDA – Reto Mercedes-Benz (Malackathon)

Resumen ejecutivo de `informe_eda.md`. Dataset: 302.850 filas × 39 columnas · 173.456 clientes · 244.519 vehículos.

---

## 1. Conclusiones

1. **La fila es una relación cliente–vehículo**, no un cliente ni un vehículo. No hay filas duplicadas ni pares (cliente, vehículo) repetidos. 46.648 vehículos aparecen en varias filas (distintos clientes o roles): es una relación N:M real y **no se debe deduplicar** por `vehicle_id`.
2. **Las flotas sesgan las medias.** 1.881 clientes con ≥10 vehículos (máximo 561) aportan el 20,6 % de las filas. La antigüedad media pasa de 43,9 meses (por fila) a 33,9 (por cliente), y los vehículos conectados de 2,15 a 1,04. Las variables de cliente se deben analizar a nivel cliente.
3. **La telemetría falta en bloque y no es aleatoria.** Solo el 47,8 % de las filas la tiene. Depende del rol (OWNER 50 %, TEMP_USER 3 %), de si el vehículo es actual (58 % frente a 33 %), del año de registro y del motor. Los eléctricos solo reportan distancia de viaje; el grupo de producto F no tiene telemetría.
4. **Los nulos de motor y combustible dependen del modelo.** `engine_concept` (44 % nulo) y `fuel_type` (59 %) varían por `model_class`: Clase B y Citan 100 % nulos, GLA y EQC 0 %. No dependen del país ni de la antigüedad.
5. **Calidad estructural buena.** Hay 0 contradicciones de atributos entre filas del mismo vehículo, 0 incoherencias motor/combustible, 0 casos de horas de uso mayores que la vida del vehículo y 0 fechas futuras.
6. **Perfil del parque y los clientes:**
   - Edad: 45-54 años (21 %) y 65+ (20 %) son los grupos más grandes.
   - Género: 65 % hombres, 18 % mujeres, 17 % desconocido.
   - País: CH 66 %, AT 34 %.
   - Financiación: 71 % de los clientes nunca la ha tenido.
   - Producto: grupo P 84 %; clases GLC (15 %), C (12 %) y A (11 %).
   - Antigüedad del vehículo: mediana 4,7 años.
7. **Comportamiento de uso (solo filas con telemetría):** velocidad media 49 km/h, viaje medio 14 km (mediana), consumo 8,1 l/100 km. El consumo baja con la electrificación: combustión 8,8, mild hybrid 7,7 y plug-in 5,8 l/100 km (medianas).

---

## 2. Problemas de calidad

| # | Problema | Afectados | Tratamiento propuesto |
|---|---|---|---|
| 1 | Telemetría ausente (estructural) | ~52 % de filas | **No imputar.** Flag `has_telematics`; analizar solo filas con dato |
| 2 | `engine_concept` / `fuel_type` nulos | 134.390 / 178.745 | Categoría `UNKNOWN` |
| 3 | `gender_derived` nulo (más en USER/TEMP_USER) | 33.923 filas | Categoría `UNKNOWN` |
| 4 | Sin datos socioeconómicos | 19.087 (6,3 %) | Mediana por país + flag `socio_missing` (AT 8,3 % frente a CH 5,1 %) |
| 5 | `population = 0` con clientes residentes | 73 | → NaN y reimputar |
| 6 | `zipcode` con formato inválido (`A-2100`, `CH-6340`, `Ottenbach`, `55605`) | 5 | Quitar prefijo de país; resto → NaN |
| 7 | **Pico anómalo en `months_as_mb_customer`**: 36 meses | 21.179 clientes (12,2 %) | **No modificar.** Sospecha de valor por defecto; confirmar con la organización |
| 8 | `months_as_mb_customer` = 0 (mezcla de clientes nuevos y por defecto) | 7.254 clientes (4,2 %) | No modificar; documentar |
| 9 | `speed_average` > 150 km/h (imposible como media) | 26 | → NaN |
| 10 | `fuel_consumption_average` > 30 l/100 km | 401 | → NaN |
| 11 | `trip_distance_average` > 100 km | 106 | Mantener; winsorizar al modelar |
| 12 | Telemetría con 0 viajes / 0 horas (vehículos sin uso) | 6.027 / 140 | Mantener; excluir al estimar hábitos de conducción |
| 13 | `speed_average` < 5 km/h (mediana de 0,14 h de uso) | 1.784 | Mantener; excluir al estimar hábitos de conducción |
| 14 | `trip_distance_mode` / `speed_mode` como texto con listas de bins | 7.049 / 1.223 | Parsear a numérico; lista → NaN + flag; excluir columna |
| 15 | `model_class` nulo (series sin clase en ningún registro) | 3.369 | `UNKNOWN`; no se puede inferir desde `model_series` |
| 16 | `date_of_first_registration` nula | 4.182 (1,4 %) | Dejar NaN en `vehicle_age_years` |
| 17 | `months_as_mb_customer` nulo | 4 | Mediana |
| 18 | Flotas que dominan las medias | 62.280 filas (20,6 %) | Reportar a nivel cliente; probar con y sin flotas |
| 19 | Valores extremos en `speed_all_counter` (máx. 1,45·10⁹, asimetría 146) | | Transformación log |

Otras observaciones:
- `currently_connected_vehicles` (atributo de cliente) y la telemetría de la fila (atributo de vehículo) tienen definiciones distintas. 10.259 filas con 0 conectados tienen telemetría. No es un error, pero no hay que cruzarlas.
- La telemetría difiere entre clientes del mismo vehículo (~9.000 vehículos): se mide por relación cliente–vehículo.

---

## 3. Columnas a descartar

Se conservan en `dataset_limpio.parquet` (como claves o por trazabilidad), pero **no entran en el análisis ni el modelado**.

| Columna | Categoría | Motivo |
|---|---|---|
| `Unnamed: 0` | Metadato | Índice de pandas 0..n-1 |
| `client_id` | Identificador | Hash pseudonimizado, 173k valores; solo sirve como clave de agregación |
| `vehicle_id` | Identificador | Hash pseudonimizado, 245k valores; solo sirve como clave de agregación |
| `zipcode` | Identificador geográfico | ~4,6k niveles; ya está recogido en las variables socioeconómicas; formatos inconsistentes |
| `vehicle_type` | Identificador de producto | 1.436 denominaciones; redundante con `model_class` + `engine_concept` |
| `model_series` | Código interno | Nominal aunque sea un entero; redundante con `model_class` (82 de 88 series → una sola clase) |
| `date_of_first_registration` | Fecha cruda | Sustituida por `vehicle_age_years` |
| `role_type` | Redundante | Función exacta de `role` (DRIVER = B2B, resto B2C) |
| `consent_phone_communication_channel` | Metadato de contacto | Flag RGPD; no describe comportamiento (se usa solo para activación) |
| `consent_epost_communication_channel` | Metadato de contacto | Ídem |
| `consent_post_communication_channel` | Metadato de contacto | Ídem |
| `purchasing_power_millions_eur` | Derivada | = `per_inhabitant` × `population` (cociente 0,9997–1,0007) |
| `number_of_households` | Redundante | Spearman 0,998 con `population` |
| `purchasing_power_per_household_eur` | Redundante | Spearman 0,956 con `per_inhabitant` |
| `trip_distance_mode` | Artefacto técnico | Marcas de bin y listas multimodales; la información útil está en average/median |
| `speed_mode` | Artefacto técnico | Ídem; solo coincide con `speed_median` en el 43 % de filas |

Total: **16 de 39 columnas**. Umbral de redundancia numérica: |ρ Spearman| ≥ 0,95.

**Se mantienen** (26 variables analíticas): edad, género, país, antigüedad como cliente, vehículos conectados, financiación, rol, `model_class`, `product_group`, `engine_concept`, `fuel_type`, `is_current_vehicle`, `workshop_visits_warranty_goodwill`, `purchasing_power_per_inhabitant_eur`, `population`, 8 variables de telemetría y 3 derivadas (`vehicle_age_years`, `has_telematics`, `socio_missing`).

**Vigilar al modelar:** los contadores `trip_distance_all_counter`, `operating_hours_total` y `speed_all_counter` están muy correlacionados (ρ ≈ 0,8–0,9). Quedan bajo el umbral y miden magnitudes distintas, pero conviene elegir uno o usar regularización.

---

## 4. Decisiones que conviene validar

- **Consentimientos (`consent_*`):** se descartan como metadatos. Si se quieren para segmentar campañas, hay que quitarlas de `EXCLUDED_COLUMNS` en `cleaning.py`.
- **Pico de 36 meses:** pendiente de confirmar con la organización si es un valor por defecto.
- **Nulos de `engine_concept`:** preguntar si se pueden recuperar desde `vehicle_type`.
- **Nivel de análisis:** decidir si se trabaja por cliente o por relación, y cómo tratar las flotas.
