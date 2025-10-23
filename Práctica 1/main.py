# ===============================================================
#  MODELO DE ENTRENAMIENTO PARA PREDICCIÓN DE LA DURACIÓN DE ACTIVIDADES DE ATLETAS
# ===============================================================
"""
Proyecto end-to-end: Predicción de duración de carreras con PySpark (Regresión Lineal)

Descripción:
  - Carga run_ww_2020_d.csv
  - Preprocesado de datos
  - Preparación de datos para entrenamiento
  - Entrenamiento de modelo de Regresión Lineal
  - Evaluación (RMSE y R²)
  - Visualización de resultados de predicciones
"""



# Imprimir título principal al iniciar el script.
print("\n" + "=" * 80) 
print("MODELO DE ENTRENAMIENTO PARA PREDICCIÓN DE LA DURACIÓN DE ACTIVIDADES DE ATLETAS".center(80)) 
print("=" * 80 + "\n")



# ---------------------------------------------------------------
# IMPORTACIONES NECESARIAS
# ---------------------------------------------------------------
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import month, when, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator



# ---------------------------------------------------------------
# CONFIGURACIÓN BÁSICA
# ---------------------------------------------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable



# ---------------------------------------------------------------
# INICIO SESIÓN SPARK
# ---------------------------------------------------------------
print("\n" + "-" * 60)
print(">>> INICIO SESIÓN SPARK")
print("-" * 60 + "\n")

spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Para evitar mensajes info y warning excesivos.
spark.sparkContext.setLogLevel("ERROR")



# ---------------------------------------------------------------
# CARGA Y EXPLORACIÓN INICIAL DEL DATASET
# ---------------------------------------------------------------
print("\n\n" + "-" * 60)
print(">>> CARGA DE DATOS Y EXPLORACIÓN INICIAL")
print("-" * 60 + "\n")

df = spark.read.csv("run_ww_2020_d.csv", header=True, inferSchema=True)
print("Muestra de las primeras cinco filas del dataset original:")
df.show(5, truncate=False)
print(f"Filas: {df.count()}, Variables: {len(df.columns)}")
print("\nEsquema del dataset original:")
df.printSchema()
print("\n--- FIN EXPLORACIÓN INICIAL ---\n\n")



# ---------------------------------------------------------------
# PREPROCESAMIENTO DEL DATASET
# ---------------------------------------------------------------
print("\n" + "-" * 60)
print(">>> INICIANDO PREPROCESAMIENTO")
print("-" * 60 + "\n")

# Eliminación de filas con valores de duración o distancia igual a 0.
df_filtered = df.filter((df.duration > 0) & (df.distance > 0))
print("Filtrado por duración y distancia > 0 completado.")

# Eliminación de filas con valores nulos.
df_filtered = df_filtered.dropna()
print("Eliminación de valores nulos completada.")

# Filtrar por número de atletas.
df_final = df_filtered.filter(df_filtered.athlete < 15000)
print("Filtrado por número de atletas < 15000 completado.")

# Creación de la variable 'season' a partir de la variable datetime.
df_final = df_final.withColumn(
    "season",
    when((month("datetime") >= 3) & (month("datetime") <= 5), "spring")
    .when((month("datetime") >= 6) & (month("datetime") <= 8), "summer")
    .when((month("datetime") >= 9) & (month("datetime") <= 11), "autumn")
    .otherwise("winter"))
print("Creación de la variable 'season' completada.")

# Selección de columnas relevantes.
df_final = df_final.select("athlete", "distance", "duration", "gender", "age_group", "country", "season")
print("Selección de columnas relevantes completada.")

# Eliminar valores atípicos.
# Filtramos primero por distancia. Eliminamos actividades con distancia < 0.5 km o > 100 km.
df_final = df_final.filter((col("distance") > 0.5) & (col("distance") < 100)) 
print("Filtrado de valores atípicos por distancia completado.")

# Luego filtramos por duración. Eliminamos actividades con duración < 5 min o > 300 min.
df_final = df_final.filter((col("duration") > 5) & (col("duration") < 300)) 
print("Filtrado de valores atípicos por duración completado.")

# Hacemos un filtrado adicional por ritmo medio.
# Primero añadimos la columna pace_min_km (min/km) como el resultado de la duración entre la distancia.
df_final = df_final.withColumn("pace_min_km", col("duration") / col("distance")) 

# Filtramos actividades con ritmo medio < 1 min/km o > 9 min/km.
df_final = df_final.filter((col("pace_min_km") >= 1.0) & (col("pace_min_km") <= 9.0))
print("Filtrado de valores atípicos por ritmo medio completado.")

print("\nMuestra de las primeras cinco filas del dataset final tras el preprocesamiento:")
df_final.show(5, truncate=False)
print(f"Filas: {df_final.count()}, Variables: {len(df_final.columns)}")
print("\nEsquema del dataset final tras el preprocesamiento:")
df_final.printSchema()
print("\n--- PREPROCESAMIENTO COMPLETADO ---\n\n")



# ---------------------------------------------------------------
# PREPARACIÓN DE DATOS PARA EL ENTRENAMIENTO
# ---------------------------------------------------------------
print("\n" + "-" * 60)
print(">>> PREPARANDO DATOS PARA ENTRENAMIENTO")
print("-" * 60 + "\n")

# Convertimos las columnas de tipo categórico en índices numéricos. 
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
age_indexer = StringIndexer(inputCol="age_group", outputCol="age_index")
country_indexer = StringIndexer(inputCol="country", outputCol="country_index")
season_indexer = StringIndexer(inputCol="season", outputCol="season_index")

# Convertimos los índices numéricos del paso anterior en vectores one-hot.
encoder = OneHotEncoder(inputCols=["gender_index", "age_index", "country_index", "season_index"],
                        outputCols=["gender_vec", "age_vec", "country_vec", "season_vec"])

# Creamos un vector features que combine todas las variables numéricas.
assembler = VectorAssembler(inputCols=["distance", "gender_vec", "age_vec", "country_vec", "season_vec"],
                            outputCol="features")

# Creamos el Pipeline con todas las etapas.
pipeline = Pipeline(stages=[gender_indexer, age_indexer, country_indexer, season_indexer, encoder, assembler])

# Ajustamos el pipeline y transformamos los datos.
df_prepared = pipeline.fit(df_final).transform(df_final)
print("Pipeline aplicado y datos transformados.")

# División de los datos en entrenamiento y test.
# Obtenemos los atletas.
athletes = df_final.select("athlete").distinct()

# Asignamos aleatoriamente el 80% de atletas a train y el 20% a test.
train_athletes, test_athletes = athletes.randomSplit([0.8, 0.2], seed=42)

# Filtramos filas según atletas asignados.
train_df = df_prepared.join(train_athletes, on="athlete", how="inner")
test_df = df_prepared.join(test_athletes, on="athlete", how="inner")
print("\nDataset separado en entrenamiento y test según atletas.")

print("\nMostramos el número de filas en train y test:")
print(f"Train: {train_df.count()} filas, Test: {test_df.count()} filas")

# Definición de la variable objetivo a predecir.
# Renombramos la columna 'duration' a 'label', ya que es la variable a predecir.
train_df = train_df.withColumnRenamed("duration", "label")  
test_df = test_df.withColumnRenamed("duration", "label")
print("\nVariable objetivo 'duration' renombrada a 'label'.\n")
print("\n--- PREPARACIÓN DE DATOS PARA ENTRENAMIENTO COMPLETADA ---\n\n")



# ---------------------------------------------------------------
# MODELO: REGRESIÓN LINEAL
# ---------------------------------------------------------------
print("\n" + "-" * 60)
print(">>> ENTRENAMIENTO DEL MODELO")
print("-" * 60 + "\n")

# Definir el modelo base.
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.001, elasticNetParam=1.0)  

# Entrenar el modelo.
print("Entrenando modelo de Regresión Lineal...")
print("Hiperparámetros: regParam=0.001, elasticNetParam=1.0, maxIter=50")
lr_model = lr.fit(train_df)
print("\n--- ENTRENAMIENTO COMPLETADO ---\n\n")



# ---------------------------------------------------------------
# EVALUACIÓN
# ---------------------------------------------------------------
print("\n" + "-" * 60)
print(">>> EVALUANDO MODELO")
print("-" * 60 + "\n")

# Preparar evaluadores.
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

# Predecir en el conjunto de test.
lr_preds = lr_model.transform(test_df)

# Evaluar el modelo.
lr_rmse = evaluator_rmse.evaluate(lr_preds)
lr_r2   = evaluator_r2.evaluate(lr_preds)

# Mostrar resultados de la evaluación.
print(f"Modelo de Regresión Lineal")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R²: {lr_r2:.4f}")
print("\n--- EVALUACIÓN COMPLETADA ---\n\n")



# ---------------------------------------------------------------
# MUESTRA DE RESULTADOS
# ---------------------------------------------------------------
# Función para añadir columnas de ritmo (min/km) al DataFrame de predicciones.
def add_paces(pred_df, distance_col="distance", y_true_col="label", y_pred_col="prediction"):
  
    return (pred_df
            .withColumn("pace_real_min_km",  col(y_true_col) / col(distance_col))  
            .withColumn("pace_pred_min_km",  col(y_pred_col) / col(distance_col))) 

# Mostrar resultados de predicciones.
# Añadir columnas de ritmo (min/km)
lr_preds = add_paces(lr_preds)

print("\n" + "-" * 60)
print(">>> MOSTRANDO RESULTADOS")
print("-" * 60 + "\n")

print("Se muestran los resultados de predicción de duración junto al ritmo medio real y predicho:")
# Mostrar algunas predicciones con ritmos.
lr_preds.select("athlete", "distance", "label", "prediction", 
                "pace_real_min_km", "pace_pred_min_km").show(10, truncate=False)



print("\n" + "-" * 60)
print(">>>  PROCESO COMPLETADO EXITOSAMENTE  <<<")
print("-" * 60 + "\n")
