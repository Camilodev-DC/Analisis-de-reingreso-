# Análisis de reingreso hospitalario en pacientes diabéticos

Predicción del **reingreso en menos de 30 días** y búsqueda de subgrupos de pacientes con **KNN, DBSCAN y mezclas gaussianas (GMM)**, sobre el dataset *Diabetes 130-US hospitals for years 1999–2008* (UCI).

El foco del proyecto es evaluar con honestidad: sin fuga de información, con intervalos de confianza, calibración, validación de los clusters con métricas adecuadas y una traducción del modelo a una decisión clínica concreta.

## Resultados principales

| Modelo | Pregunta | Resultado |
|---|---|---|
| **KNN** | ¿Qué paciente va a reingresar? | AUC **0.639** (IC 95 %: 0.624–0.656), dentro del rango publicado sin fuga (0.62–0.67). Con las mismas variables, regresión logística (0.645) y gradient boosting (0.644) rinden igual |
| **DBSCAN** | ¿Existen grupos naturales de pacientes? | **No.** Con DBCV (métrica para clusters por densidad) no hay estructura; los clusters aparentes son escalones de variables discretas que desaparecen con jitter |
| **GMM** | ¿Qué tipos de hospitalización hay? | 4 perfiles estables (consenso de 10 ajustes, ARI 0.88), con reingreso de 7.0 % a 10.9 %. Útiles para describir, no para predecir |

**Propuesta de uso:** seguimiento en dos niveles. Cita presencial para el 10 % de mayor riesgo (reingresa el **18.9 %**, el doble del promedio) y teleconsulta para el 20 % siguiente. Contactando al 31 % de los pacientes se cubre el 49 % de los reingresos. Bajo supuestos de costo ilustrativos, ahorra ~2 veces más que la mejor política sin modelo.

## Estructura

```
├── Data/
│   ├── diabetic_data.csv        dataset original (UCI)
│   ├── IDS_mapping.csv          significado de los códigos de admisión y alta
│   └── diabetic_clean.csv       dataset limpio (lo genera el EDA)
├── EDA/
│   └── EDA_diabetes.ipynb       exploración, problemas de calidad y limpieza
├── Modelos/
│   ├── KNN_diabetes.ipynb       clasificación, calibración, comparación, curva de decisión y propuesta de uso
│   ├── DBSCAN_diabetes.ipynb    clustering por densidad y revisión crítica (DBCV, jitter, estabilidad)
│   └── GMM_diabetes.ipynb       mezcla gaussiana, ICL, GMM de consenso y validación externa
├── Informes/
│   ├── Informe_ejecutivo_reingreso_diabetes (.docx / .pdf)   4 páginas, para toma de decisiones
│   └── Informe_academico_reingreso_diabetes (.docx / .pdf)   4 páginas, formato paper
└── Decisiones_y_conclusiones.md  todas las decisiones estadísticas, cambios y conclusiones
```

## Cómo ejecutarlo

1. Python 3.11 y las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar en este orden:
   1. `EDA/EDA_diabetes.ipynb` (genera `Data/diabetic_clean.csv`; ya viene incluido)
   2. Cualquiera de los notebooks de `Modelos/`

Los notebooks encuentran la carpeta `Data/` solos, tanto si se abren desde la raíz del repositorio como desde su propia carpeta. Tiempos aproximados: EDA 1 min, KNN 6 min, DBSCAN 4 min, GMM 8 min.

## Decisiones clave

- **Un encuentro por paciente** (101 766 → 69 987): el 46 % de las filas eran pacientes repetidos, lo que genera fuga de información entre entrenamiento y prueba.
- **Sin fallecidos ni altas a hospicio**, que no pueden reingresar.
- **Sin SMOTE:** el desbalance (9 % de reingresos) se maneja eligiendo el umbral con predicciones *out-of-fold* de entrenamiento; la prueba se usa una sola vez.
- **Probabilidades recalibradas** (regresión isotónica) para interpretarlas como riesgo real.
- **Clustering validado con métricas adecuadas:** DBCV para DBSCAN, ICL y estabilidad para la GMM, y validación externa contra el reingreso con intervalos de confianza y líneas base.

El detalle completo está en [Decisiones_y_conclusiones.md](Decisiones_y_conclusiones.md).

## Limitaciones

Datos administrativos de 1999–2008 de una sola red de hospitales, sin laboratorio detallado ni variables sociales, y sin validación externa. Los costos y la efectividad del análisis económico son supuestos ilustrativos, no datos.

## Datos

Strack B. et al. (2014). *Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records.* BioMed Research International. Dataset disponible en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (licencia CC BY 4.0).
