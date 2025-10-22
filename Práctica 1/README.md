# Práctica 1 — Aprendizaje Automático a Gran Escala  

**Predicción de la duración de carreras con PySpark**

Este proyecto utiliza **PySpark** para predecir la duración (en minutos) de actividades de carrera a partir del dataset `run_ww_2020_d.csv`, aplicando un modelo de **Regresión Lineal** con **validación cruzada**.

---

## Contenido
- `Practica1_final.py` → Script ejecutable del proyecto.  
- `run_ww_2020_d.csv` → Dataset (debe estar en el mismo directorio).  

---

## Requisitos
- **Python 3.9 o superior**  
- **Java JDK 11–22** (con `JAVA_HOME` configurado)  
- **PySpark** instalado:  
  ```bash
  pip install pyspark
  ```

---

## Cómo ejecutar
1. Asegúrate de que el CSV está junto al script.  
2. Ejecuta:
   ```bash
   python Practica1_final.py
   ```
3. El programa mostrará:
   - Número de filas leídas.  
   - Resultados del modelo (RMSE y R²).  
   - Ejemplo de predicciones.  

---

## Resumen del pipeline
1. **Carga y limpieza** de datos (`distance`, `duration`, `athlete`, etc.).  
2. **Cálculo del ritmo** (`pace_min_km`) y eliminación de outliers.  
3. **Codificación** de variables categóricas (`gender`, `age_group`, `country`, `major`, `season`).  
4. **Separación** del conjunto en train/test por atleta.  
5. **Entrenamiento** con `LinearRegression` y `CrossValidator`.  
6. **Evaluación** final mediante **RMSE** y **R²**.

---

## Resultados esperados
Ejemplo de salida:
```
=== RESULTADOS ===
RMSE: 10.51
R²: 0.90
```

---

