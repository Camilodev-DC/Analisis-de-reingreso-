# Reingreso hospitalario en pacientes diabéticos: decisiones, cambios y conclusiones

**Dataset:** *Diabetes 130-US hospitals for years 1999–2008* (UCI). 101 766 hospitalizaciones de pacientes diabéticos en 130 hospitales de EE. UU.
**Pregunta:** ¿qué pacientes diabéticos que salen del hospital van a **reingresar en menos de 30 días**?
**Modelos:** KNN (clasificación), DBSCAN y mezcla gaussiana (clustering).

> Todos los pacientes del dataset son diabéticos, así que la diabetes no se puede predecir. Lo que se predice es el **reingreso**.

**Archivos del proyecto**

| Archivo | Contenido |
|---|---|
| `EDA/EDA_diabetes.ipynb` | EDA del dataset original y limpieza |
| `Modelos/KNN_diabetes.ipynb` | Clasificación, calibración, comparación con otros modelos y propuesta de uso |
| `Modelos/DBSCAN_diabetes.ipynb` | Clustering por densidad |
| `Modelos/GMM_diabetes.ipynb` | Mezcla gaussiana |
| `Data/diabetic_clean.csv` | Dataset limpio (69 987 filas), generado por el EDA |
| `Informes/` | Informe ejecutivo e informe académico (4 páginas cada uno, .docx y .pdf) |

---

## 1. Qué hicimos (resumen)

1. **EDA del dataset original:** detectamos faltantes escondidos, pacientes repetidos, pacientes fallecidos y columnas inútiles.
2. **Limpieza:** dejamos un encuentro por paciente y convertimos los faltantes en categorías. Resultado: 69 987 pacientes.
3. **EDA del dataset limpio:** verificamos la limpieza y encontramos los problemas que siguen presentes (ese notebook no se incluye en el repositorio).
4. **Tres modelos en paralelo:** KNN, DBSCAN y GMM, cada uno con su notebook.
5. **Revisión crítica:** investigamos la literatura del dataset y las buenas prácticas metodológicas, y comparamos con lo que hicimos.
6. **Mejoras al KNN:** intervalos de confianza, calibración, comparación con modelos de referencia, curva de decisión y una propuesta de uso con análisis de costos.
7. **Mejoras a DBSCAN y GMM:** validación con DBCV e ICL, estabilidad por cluster, prueba del jitter, GMM de consenso y validación externa con intervalos de confianza.

---

## 2. Decisiones de datos (limpieza)

| Decisión | Por qué | Evidencia |
|---|---|---|
| Tratar `'?'` como faltante | El dataset marca los faltantes con `'?'`, no con vacíos | `weight` 97 %, `medical_specialty` 49 %, `payer_code` 40 %, `race` 2 % |
| Eliminar `weight` y `payer_code` | Demasiados faltantes y poca relevancia | 97 % y 40 % faltantes |
| Laboratorio no medido → categoría `NoMedido` | Que no se haya pedido el examen es información, no un dato perdido | `max_glu_serum` 95 %, `A1Cresult` 83 % sin medir |
| Traducir los IDs de admisión y alta a texto; los códigos NULL / Not Mapped → `Desconocido` | Son categorías, no números. Además esconden faltantes | ~10 % en tipo de admisión, ~7 % en origen, ~5 % en alta |
| Eliminar fallecidos y pacientes en hospicio | Un paciente fallecido no puede reingresar: su `NO` es trivial | 1 652 fallecidos y 771 en hospicio |
| **Quedarse con el primer encuentro de cada paciente** | Los encuentros del mismo paciente no son independientes; si caen en train y en test hay **fuga de información** | 46 % de las filas eran de pacientes repetidos (hasta 40 encuentros por paciente) |
| Agrupar diagnósticos ICD-9 en 9 capítulos + `Desconocido` | Eran ~700–800 códigos por columna | Agrupación estándar del paper original (Strack et al. 2014) |
| `medical_specialty`: top 10 + `Otra` + `Desconocido` | 73 categorías, muchas casi vacías | — |
| Eliminar medicamentos casi constantes | No aportan información | `examide` y `citoglipton` constantes; 10 medicamentos con > 99 % de "No" |
| Edad como número (punto medio del intervalo) | Es una variable ordinal | `[70-80)` → 75 |
| `log1p` en las visitas previas | Muy asimétricas, con valores extremos | Asimetría de 3.6 a 22.9; outliers de hasta 80 desviaciones estándar |
| Variables nuevas: `n_meds_activos`, `n_cambios_meds` | Resumen de los 23 medicamentos | — |
| Objetivo binario `readmitted_30` (`<30` contra el resto) | Es el reingreso clínicamente relevante. Con 3 clases el modelo casi no distingue | Con 3 clases la balanced accuracy es 0.35 (azar = 0.33) |

**Resultado:** 69 987 pacientes y 39 columnas, sin faltantes. Coincide casi exactamente con el paper original del dataset (Strack et al. 2014: 69 984). El reingreso `<30` pasó de 11.2 % a **9.0 %**.

### Problemas que siguen en el dataset limpio (y que condicionan todo)

- **Desbalance:** solo el 9 % reingresa en menos de 30 días.
- **Visitas previas cero-infladas:** entre 87 y 93 % de ceros, incluso después del `log1p`. En la práctica son casi binarias.
- **`patient_nbr` contiene información de tiempo:** tiene correlación 0.24 con `number_diagnoses`. **Nunca se usa como variable.**
- **Tope artificial en `number_diagnoses`:** el 44 % tiene exactamente 9, y el porcentaje sube en los registros más recientes. Es un cambio en el sistema de registro.
- **Señal débil:** la mejor variable sola tiene AUC 0.56 (`time_in_hospital`); la mejor categórica, V de Cramér 0.14 (destino al alta).
- **No hay grupos naturales visibles:** en la proyección PCA se ve una nube continua, y hacen falta 7 componentes para explicar el 80 % de la varianza.

---

## 3. KNN: decisiones estadísticas

### Configuración

| Decisión | Elección | Por qué |
|---|---|---|
| Variables | 8 numéricas + destino al alta agrupado (7 grupos) + `diag_1` | En KNN cada columna pesa en la distancia; conviene pocas y útiles |
| Preprocesamiento | `StandardScaler` + `OneHotEncoder` dentro de un `Pipeline` | Evita fuga: se ajusta solo con train |
| Separación de datos | 80/20 estratificado | Mantiene el 9 % de positivos en train y en test |
| Hiperparámetros | Validación cruzada de 5 folds optimizando **ROC-AUC** → k = 676, manhattan, uniform | El AUC mide qué tan bien se ordena a los pacientes, sin depender del umbral |
| Manejo del desbalance | **Mover el umbral** (sin SMOTE) | KNN no tiene `class_weight`. La literatura muestra que remuestrear empeora la calibración |
| Elección del umbral | Con predicciones *out-of-fold* de train, **nunca con test** | Evita sobreajustar el umbral al test |

### Cómo cambió la decisión del umbral

KNN no da una probabilidad propiamente tal: da la **proporción de vecinos que reingresaron**. Con k = 676, un paciente con 84 vecinos que reingresaron recibe 84 / 676 = 0.124. Casi nadie pasa de 0.5, así que el umbral por defecto no detecta a nadie.

| Versión | Criterio | Umbral | Reingresos detectados | **No detectados** (de 1 257) | Pacientes marcados |
|---|---|---|---|---|---|
| 1 | Umbral por defecto | 0.5 | 0 | 1 257 | 0 % |
| 2 | Máximo F1 | 0.098 | 608 (48 %) | 649 | 30 % |
| 3 | Recall ≥ 80 % (minimizar falsos negativos) | 0.065 | 994 (79 %) | 263 | 66 % |
| **4 (final)** | **Dos niveles por capacidad** (ver sección 5) | 0.124 / 0.096 | 49 % de los reingresos con algún contacto | — | 31 % |

**Cómo llegamos a la versión final:**
- **De F1 a recall 80 %:** un falso negativo (el paciente reingresa sin que lo hayamos marcado) es el error grave, porque se va a casa sin seguimiento.
- **De recall 80 % a dos niveles:** marcar al 66 % de los pacientes genera costos innecesarios. Cada reingreso extra detectado entre F1 y recall 80 % cuesta ~12 seguimientos innecesarios. Con ese umbral, el modelo apenas le gana a hacer seguimiento a todos (3.5 pacientes de cada 1 000).
- **Solución:** diferenciar la intervención. Lo caro (cita presencial) va solo al riesgo alto; lo barato (teleconsulta) puede ser más amplio.

### Mejoras que agregamos después de la revisión crítica

| Mejora | Resultado |
|---|---|
| **Intervalos de confianza** (bootstrap, 1 000 repeticiones) | AUC 0.639, IC 95 % [0.624, 0.656]. Las diferencias menores a ~0.01 son ruido |
| **Calibración** | El KNN subestimaba el riesgo más alto (14.5 % predicho contra 19.4 % real). Con recalibración isotónica ajustada en train: 18.5 % contra 19.1 % |
| **Modelos de referencia** | Con las mismas variables, todos empatan: regresión logística AUC 0.645 y gradient boosting 0.644 (diferencias no significativas). Gradient boosting con **todas** las variables: **0.653**, significativamente mejor (+0.014, IC [0.004, 0.024]), AP 0.187 contra 0.161. La ventaja viene de las variables, no del algoritmo |
| **Curva de decisión** (beneficio neto) | Con umbral 0.065 el modelo apenas supera a "seguimiento a todos". Con umbral 0.10 lo supera claramente (+14.7 contra −11.3 por cada 1 000 pacientes) |
| **Variables de la literatura en KNN** | Agregarlas **empeora** al KNN (AUC 0.640 → 0.634 con todas), aunque ayudan al gradient boosting |
| **Comparación con marcar al azar** | Con la misma carga (66 %), el KNN detecta 994 reingresos y el azar ~857 |

---

## 4. Clustering: decisiones, revisión crítica y resultados

Los dos modelos de clustering se hicieron en dos etapas: una **primera versión** y una **revisión crítica** basada en la literatura metodológica. En la revisión se mantuvo el mismo algoritmo (DBSCAN y GaussianMixture); lo que cambió fue cómo se valida.

### DBSCAN

**Primera versión:** 10 variables estandarizadas (incluye las visitas previas), muestra de 20 000, `eps` = 1.5, `min_samples` = 20 → 4 clusters + 14.5 % de ruido, silhouette 0.17.

**Revisión crítica:**

| Decisión | Por qué | Resultado |
|---|---|---|
| Validar con **DBCV** (Moulavi et al. 2014) en vez de silhouette | Silhouette y Davies-Bouldin favorecen clusters redondos; DBCV está hecho para clusters por densidad (de −1 a +1). Lo programamos y lo probamos con datos de juguete: +0.53 al clustering correcto, −0.60 al incorrecto | Primera versión: **−0.30**, casi igual que la regla trivial "tipo de visita previa" (−0.34) |
| Estabilidad por cluster con Jaccard en re-muestras (inspirado en Hennig 2007) | Medir si cada cluster se repite en otras muestras (≥ 0.75 estable) | 0.77–0.99: estables, **pero son capas triviales** que reproduce una regla simple |
| DBSCAN **sin las visitas previas** (7 variables; `min_samples` = 14 = 2 × dimensión) | Las visitas previas son casi binarias y dominaban la distancia | **DBCV negativo en todas las configuraciones** (−0.005 a −0.45): no hay clusters por densidad |
| **Prueba del jitter**: sumar ruido que no cambia ningún dato | Si los clusters son reales, deben sobrevivir | Los 4 clusters (`eps` = 1.0) eran exactamente "0, 1, 2 y 3 medicamentos" (pureza 100 %). Con jitter **se funden en uno solo** |
| Validación externa del ruido con IC 95 % contra reglas simples | Ver si DBSCAN aporta algo como detector de riesgo | Ruido de la primera versión: reingreso 13.0 % [11.8, 14.3], riesgo relativo 1.55. La regla "≥ 1 hospitalización previa" es mejor: 15.5 % [14.1, 17.0], riesgo relativo 1.89, marcando a menos pacientes |

**Conclusión DBSCAN:** este dataset **no tiene clusters por densidad**. DBSCAN encuentra los "escalones" de las variables discretas: en la primera versión, el tipo de visita previa (capas reales pero triviales, que reproduce una regla simple); sin las visitas previas, el número de medicamentos (un artefacto: desaparece al romper la grilla de enteros, aunque tenía Jaccard 0.85–0.94). Por eso, **estable no significa real**. Su valor en el proyecto es diagnóstico: muestra que los pacientes forman un continuo.

### Mezcla gaussiana (GMM)

**Primera versión:** 7 variables sin visitas previas (con ellas las componentes colapsan), jitter de ±0.5 para romper la discretez, K = 4 por el codo del BIC (que no tiene mínimo), silhouette 0.07.

**Revisión crítica:**

| Decisión | Por qué | Resultado |
|---|---|---|
| Medir la **estabilidad frente al jitter** | El jitter agrega azar que no se había evaluado | Con K = 4, cambiar solo el jitter da ARI **0.72** (mínimo 0.65): estabilidad moderada |
| Elegir K con **ICL** (Biernacki et al. 2000) | Penaliza componentes solapadas; está pensado para clustering | Sin K claro: el mínimo promedio está en K = 7, pero su ICL varía ±2 000 según el jitter y ese rango abarca a K = 3 y K = 4, que quedan empatados. K = 5 es algo más estable que K = 4 (ARI 0.73 vs 0.72). Se mantuvo **K = 4 por interpretabilidad** (grupo más chico: 17.6 % en el consenso) |
| **GMM de consenso**: 10 ajustes con distinto jitter, componentes alineadas (algoritmo húngaro) y probabilidades promediadas | Que el resultado no dependa de una realización del ruido | Estabilidad entre dos consensos independientes: ARI **0.88** (contra 0.71 individual); Jaccard por grupo 0.81–0.85 contra ajustes independientes |
| Validación externa con IC 95 % y **línea base** (cuartiles de una sola variable) | Saber si los grupos dicen algo del reingreso más allá de una variable | Reingreso de 7.0 % [6.6, 7.4] a 10.9 % [10.4, 11.4]. V de Cramér **0.053**, igual que los cuartiles de días de hospitalización (0.055) |

**Grupos (consenso):** mayores con muchos diagnósticos (33 %, reingreso 10.2 %), estancia corta sin procedimientos (28 %, 7.9 %), estancia corta con procedimiento (21 %, 7.0 %) y estancia larga e intensiva (18 %, 10.9 %). Son los mismos perfiles de la primera versión.

**Conclusión GMM:**
- Los 4 perfiles son **estables e interpretables**, pero son una forma útil de dividir un continuo, no grupos naturales. El 26 % de los pacientes queda en la frontera entre dos perfiles.
- Un grupo depende de un artefacto: el de "muchos diagnósticos" está formado por pacientes con exactamente 9 diagnósticos, el tope del sistema de registro.
- Como herramienta de riesgo no supera a una sola variable. Su valor es **descriptivo**.

---

## 5. Propuesta de uso del modelo: seguimiento en dos niveles

| Nivel | Para quién | Reingresa realmente (test) | % de los reingresos |
|---|---|---|---|
| **Cita presencial** | 10 % de mayor riesgo | **18.9 %** [17.0, 21.0] (1 de cada 5) | 23 % |
| **Teleconsulta** | Siguiente 20 % | 11.6 % [10.5, 12.9] | 27 % |
| Sin seguimiento | Resto | 6.6 % [6.1, 7.1] | 51 % |

Los cortes se calcularon en train y se aplicaron en test (en test se contacta al 31.4 %: 10.7 % + 20.8 %). Entre corchetes, IC 95 % de Wilson. La cobertura de reingresos es 49 % [47, 52].

**Análisis económico** (supuestos ilustrativos: reingreso US$15 000; teleconsulta US$50 que evita 5 % de los reingresos; cita presencial US$250 que evita 20 %):
- Ahorro de la propuesta: ~**US$41 700 por cada 1 000 pacientes** (IC 95 % bootstrap: 35 000–48 500). La mejor opción sin modelo (cita presencial a todos) ahorra ~US$19 400 y cuesta 7 veces más; la diferencia es positiva en el 95 % de las re-muestras (10 200–33 500).
- Asignar según el riesgo calibrado ahorra más (~US$59 500), pero exige citas presenciales para el 41 % de los pacientes.
- En el análisis de sensibilidad (12 escenarios de efectividad), la propuesta ahorra en **11 de 12**.
- **El supuesto que más pesa es la efectividad real de las intervenciones.** Este dataset no puede medirla; habría que estimarla con un piloto.
- Además se supone que la efectividad es **la misma para todos los pacientes**, sin importar su riesgo. Si en los más graves fuera menor (o mayor), el ahorro real sería menor (o mayor).

---

## 6. Conclusiones

1. **El modelo funciona dentro de lo que permiten los datos.** El KNN obtiene AUC 0.639 [0.624, 0.656]. La literatura, cuando evalúa bien (un paciente por fila), reporta 0.62–0.67 para este problema. Los trabajos con AUC > 0.9 tienen fuga de información, por ejemplo por aplicar SMOTE antes de separar train y test. **El techo lo ponen las variables disponibles, no el algoritmo.**
2. **Accuracy engaña.** Decir siempre "no reingresa" da 91 % de accuracy sin detectar a nadie. Hay que evaluar con recall, precision, AUC y average precision.
3. **El umbral es una decisión de costos, no estadística.** Pasar de F1 a recall 80 % evita 386 reingresos no detectados, pero cuesta ~12 seguimientos innecesarios por cada uno. La mejor solución fue diferenciar la intervención según el riesgo.
4. **Con las mismas variables, el algoritmo no importa.** La regresión logística (0.645) y el gradient boosting (0.644) empatan con el KNN. Solo el gradient boosting con **todas** las variables es significativamente mejor (0.653): la ventaja viene de las variables extra, que el KNN no puede aprovechar porque en su distancia cada columna pesa igual.
5. **Las probabilidades del KNN hay que recalibrarlas** antes de interpretarlas como riesgo. Con k = 676 se "aplastan" hacia el promedio.
6. **Las variables más asociadas al reingreso** son las hospitalizaciones previas, el destino al alta (los pacientes que van a rehabilitación reingresan un 26 %) y la duración de la estancia. Lo muestran el EDA (asociaciones univariadas) y la ablación del KNN (el destino al alta sube el AUC de 0.606 a 0.633), y coincide con la literatura. El KNN no mide importancia de variables.
7. **El clustering confirma que no hay grupos naturales.** Con la métrica adecuada (DBCV), DBSCAN no encuentra clusters por densidad: solo los escalones de las variables discretas, que desaparecen con jitter. La GMM de consenso da 4 perfiles estables y descriptivos, pero no dice más sobre el reingreso que los días de hospitalización. **El clustering sirve para describir; el KNN, para estimar el riesgo.**
8. **Estable no significa real.** Los clusters de DBSCAN sin visitas previas eran un artefacto de la discretez y aun así tenían Jaccard de 0.85 a 0.94; los de la primera versión (hasta 0.99) son capas reales pero triviales. Hay que combinar la estabilidad con una métrica adecuada (DBCV, ICL) y con validación externa.

### Limitaciones
- Datos de 1999–2008, administrativos, de una sola red de hospitales. Solo se ven los reingresos dentro de esa red.
- No hay laboratorio detallado, variables sociales ni validación externa.
- Hay artefactos de registro que cambian con el tiempo (por ejemplo, el tope de 9 diagnósticos).
- Los costos y la efectividad de la sección 5 son supuestos, no datos.
- Las asociaciones univariadas que orientaron la elección de variables se calcularon con todos los pacientes (test incluido); el sesgo esperado es mínimo.
- Algunas variables (`diag_1`, `number_diagnoses`) suelen codificarse al cierre administrativo del alta; habría que confirmar que están disponibles cuando se decide el seguimiento.
- La limpieza excluye fallecidos y hospicio antes de quedarse con el primer encuentro: 17 pacientes cuyo primer encuentro fue un alta a hospicio entran con un encuentro posterior (0.02 % de la cohorte; impacto nulo).

---

## 7. Qué hicimos bien y qué queda pendiente

**Bien:**
- La limpieza sigue al paper original.
- No hay fugas de información: `Pipeline`, test usado una sola vez y umbral elegido fuera de test.
- No usamos SMOTE.
- En KNN: comparamos contra el azar y contra modelos de referencia, y reportamos intervalos de confianza y calibración.
- En clustering: validamos con métricas adecuadas (DBCV, ICL), medimos la estabilidad, probamos si los clusters sobreviven al jitter y validamos contra el reingreso con intervalos de confianza y líneas base.

**Pendiente (opcional):**
- Reportar resultados por subgrupos (sexo, edad, raza) y por periodo de tiempo.
- Estimar la efectividad real de las intervenciones de seguimiento; el análisis de costos depende de ella.

---

## Referencias principales

- Strack B. et al. (2014). *Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records.* BioMed Research International. https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/
- Liu V.B., Sue L.Y., Wu Y. (2024). *Comparison of machine learning models for predicting 30-day readmission rates for patients with diabetes.* J Med Artif Intell. https://jmai.amegroups.org/article/view/9179/html
- Kansagara D. et al. (2011). *Risk prediction models for hospital readmission: a systematic review.* JAMA. https://pmc.ncbi.nlm.nih.gov/articles/PMC3603349/
- Vickers A.J., Elkin E.B. (2006). *Decision curve analysis.* Medical Decision Making. https://journals.sagepub.com/doi/10.1177/0272989X06295361
- Van Calster B. et al. (2019). *Calibration: the Achilles heel of predictive analytics.* BMC Medicine. https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7
- van den Goorbergh R. et al. (2022). *The harm of class imbalance corrections for risk prediction models.* JAMIA. https://academic.oup.com/jamia/article/29/9/1525/6605096
- Schubert E. et al. (2017). *DBSCAN Revisited, Revisited.* ACM TODS. https://doi.org/10.1145/3068335
- Moulavi D. et al. (2014). *Density-Based Clustering Validation.* SIAM SDM. https://epubs.siam.org/doi/10.1137/1.9781611973440.96
- Hennig C. (2007). *Cluster-wise assessment of cluster stability.* Computational Statistics & Data Analysis 52: 258–271. https://www.sciencedirect.com/science/article/abs/pii/S0167947306004622
- Biernacki C., Celeux G., Govaert G. (2000). *Assessing a mixture model for clustering with the integrated completed likelihood.* IEEE TPAMI 22(7): 719–725.
- Collins G.S. et al. (2024). *TRIPOD+AI.* BMJ. https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/
