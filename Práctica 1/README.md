# Práctica 1 - Aprendizaje Automático a Gran Escala  

**Predicción de la duración de actividades de atletas con PySpark**

El objetivo de este proyecto es entrenar un modelo de **Regresión Lineal** que prediga la duración (en minutos) de actividades de atletas a partir del dataset `run_ww_2020_d.csv` (https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset/data), utilizando **PySpark**.

---

## Contenido necesario
- `main.py` → Script ejecutable del proyecto.  
- `run_ww_2020_d.csv` → Dataset.  
> ⚠️ Asegúrate de mantener el nombre del dataset (`run_ww_2020_d.csv`) y de no modificar su estructura original.

---

## Requisitos
- **Python 3.9** o superior 
- **Java JDK** en versiones desde la 11 hasta la 22 (con variable de entorno `JAVA_HOME` configurada)  
- **PySpark** instalado:  
  ```bash
  pip install pyspark
  ```

---

## Cómo ejecutar
1. Ubicar tanto el archivo `main.py` como el `run_ww_2020_d.csv` en el mismo directorio.
2. Abrir una terminal en dicho directorio.
3. Ejecuta:
   ```bash
   python main.py
   ```
4. El programa mostrará:
   - Número de filas leídas.  
   - Resultados del modelo (RMSE y R²).  
   - Ejemplo de predicciones.  

---

## Resultados esperados
Al finalizar, el script mostrará métricas de evaluación del modelo:
- **RMSE (Root Mean Squared Error)** → error promedio en minutos.  
- **R² (Coeficiente de determinación)** → calidad del ajuste del modelo.  

Además, se mostrarán ejemplos de predicciones comparando la duración real y predicha junto al ritmo medio (min/km).
