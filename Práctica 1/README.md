# Práctica 1 — Aprendizaje Automático a Gran Escala

**Predicción de la duración de actividades de atletas con PySpark - MLlib**

El objetivo de este proyecto es entrenar un modelo de **Regresión Lineal** que prediga la duración (en minutos) de las actividades realizadas por atletas, a partir del dataset `run_ww_2020_d.csv` (https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset/data), utilizando **PySpark**.

---

## Contenido necesario
- `main.py` → Script ejecutable del proyecto.  
- `run_ww_2020_d.csv` → Dataset.  
> ⚠️ Asegúrate de mantener el nombre del dataset (`run_ww_2020_d.csv`) y de no modificar su estructura original.
📥 [Descargar `run_ww_2020_d.csv`](https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset/data?select=run_ww_2020_d.csv)


---

## Requisitos
- **Python 3.9** o superior 
- **Java JDK** en versiones desde la 11 hasta la 22 (con variable de entorno `JAVA_HOME` configurada)  
- **PySpark** instalado:  
  ```bash
  pip install pyspark
  ```

---

## Estructura del código
El script `main.py` se organiza en las siguientes secciones:
1. **Carga y exploración del dataset.**
2. **Preprocesamiento de datos:** limpieza, filtrado de valores atípicos y creación de variables derivadas.
3. **Preparación para el entrenamiento:** transformación de variables categóricas, ensamblado de características y división del dataset en entrenamiento y test.
4. **Creación y entrenamiento del modelo:** regresión lineal con PySpark MLlib.
5. **Evaluación y resultados:** cálculo de métricas (RMSE, R²) y visualización de ejemplos de predicción.

---

## Cómo ejecutar
1. Ubicar tanto el archivo `main.py` como el `run_ww_2020_d.csv` en el mismo directorio.
2. Abrir una terminal en dicho directorio.
3. Ejecuta:
   ```bash
   python main.py
   ```
4. El programa mostrará:
   - Filas de ejemplo del dataset original.
   - Número de filas y variables del dataset. 
   - Los pasos que se van siguiendo. 
   - Hiperparámetros seleccionados para el modelo.
   - Resultados del modelo (RMSE y R²).  
   - Ejemplo de predicciones.  

---

## Resultados esperados
Al finalizar, el script mostrará métricas de evaluación del modelo:
- **RMSE (Root Mean Squared Error)** → error promedio en minutos.  
- **R² (Coeficiente de determinación)** → calidad del ajuste del modelo.  

Además, se mostrarán ejemplos de predicciones comparando la duración real y predicha junto al ritmo medio (min/km).

---

## Ejemplo de salida

Al ejecutar `main.py`, se mostrarán resultados similares a:

| athlete | distance | label | prediction | pace_real_min_km | pace_pred_min_km |
|---------|---------|-------|------------|-----------------|-----------------|
| 1580    | 11.54   | 63.0  | 59.31      | 5.46            | 5.14            |


Esto permite ver rápidamente cómo se comparan las **duraciones reales** (`label`) con las **predicciones** (`prediction`) y el ritmo medio estimado (`pace_pred_min_km`).

