
"""
Proyecto end-to-end: Predicción de duración de carreras con PySpark (Regresión Lineal)

Descripción:
  - Carga run_ww_2020_d.csv
  - Limpieza de datos y filtrado de outliers
  - Creación de variable 'season'
  - Codificación categórica (OneHotEncoder)
  - División en train/test por atleta
  - Entrenamiento de modelo de Regresión Lineal con CrossValidation
  - Evaluación (RMSE y R²)
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, when, to_timestamp, lit, round as spark_round
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# ========================
# CONFIGURACIÓN BÁSICA
# ========================
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

INPUT_PATH = "run_ww_2020_d.csv"   # Debe estar en la misma carpeta
ATHLETE_MAX = 15000
SEED = 42

# ========================
# SESIÓN SPARK
# ========================
spark = (SparkSession.builder
         .appName("MarathonLinearRegression")
         .config("spark.driver.memory", "8g")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# ========================
# CARGA Y EXPLORACIÓN
# ========================
print("=== CARGANDO DATOS ===")
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
print(f"Filas totales: {df.count()}, Columnas: {len(df.columns)}")

# ========================
# LIMPIEZA Y FILTRADO
# ========================
df = df.filter((col("duration") > 0) & (col("distance") > 0)).dropna()
df = df.filter(col("athlete") < ATHLETE_MAX)

# Crear columna 'season' a partir de datetime (si existe)
if "datetime" in df.columns:
    df = df.withColumn("dt_ts", to_timestamp(col("datetime")))
    df = df.withColumn(
        "season",
        when((month("dt_ts") >= 3) & (month("dt_ts") <= 5), "spring")
        .when((month("dt_ts") >= 6) & (month("dt_ts") <= 8), "summer")
        .when((month("dt_ts") >= 9) & (month("dt_ts") <= 11), "autumn")
        .otherwise("winter")
    )
else:
    df = df.withColumn("season", lit("unknown"))

# Añadir columnas faltantes si no existen
for c in ["gender", "age_group", "country"]:
    if c not in df.columns:
        df = df.withColumn(c, lit(None))

# Selección de columnas relevantes
df = df.select("athlete", "distance", "duration", "gender", "age_group", "country", "season")

# Eliminar valores atípicos
df = df.filter((col("distance") > 0.5) & (col("distance") < 100))
df = df.filter((col("duration") > 5) & (col("duration") < 300))
df = df.withColumn("pace_min_km", col("duration") / col("distance")) \
       .filter((col("pace_min_km") >= 1.0) & (col("pace_min_km") <= 9.0))

print(f"Filas tras limpieza: {df.count()}")

# ========================
# PREPARACIÓN DE FEATURES
# ========================
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
age_indexer = StringIndexer(inputCol="age_group", outputCol="age_index", handleInvalid="keep")
country_indexer = StringIndexer(inputCol="country", outputCol="country_index", handleInvalid="keep")
season_indexer = StringIndexer(inputCol="season", outputCol="season_index", handleInvalid="keep")

encoder = OneHotEncoder(
    inputCols=["gender_index", "age_index", "country_index", "season_index"],
    outputCols=["gender_vec", "age_vec", "country_vec", "season_vec"]
)

assembler = VectorAssembler(
    inputCols=["distance", "gender_vec", "age_vec", "country_vec", "season_vec"],
    outputCol="features"
)

pipeline = Pipeline(stages=[gender_indexer, age_indexer, country_indexer, season_indexer, encoder, assembler])
df_prepared = pipeline.fit(df).transform(df)

# ========================
# DIVISIÓN TRAIN / TEST
# ========================
athletes = df_prepared.select("athlete").distinct()
train_athletes, test_athletes = athletes.randomSplit([0.8, 0.2], seed=SEED)

train_df = df_prepared.join(train_athletes, on="athlete", how="inner").withColumnRenamed("duration", "label")
test_df = df_prepared.join(test_athletes, on="athlete", how="inner").withColumnRenamed("duration", "label")

print(f"Train: {train_df.count()} filas, Test: {test_df.count()} filas")

# ========================
# MODELO: REGRESIÓN LINEAL CON CROSS-VALIDATION
# ========================
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=50)

param_grid = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.0, 0.01, 0.1])
              .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
              .build())

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_r2,
    numFolds=3,
    seed=SEED
)

print("Entrenando modelo de Regresión Lineal (con CrossValidation)...")
cv_model = cv.fit(train_df)

# ========================
# EVALUACIÓN
# ========================
predictions = cv_model.transform(test_df)
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n=== RESULTADOS ===")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Muestra de predicciones
predictions = predictions.withColumn("pace_real_min_km", spark_round(col("label") / col("distance"), 3)) \
                         .withColumn("pace_pred_min_km", spark_round(col("prediction") / col("distance"), 3))

predictions.select("athlete", "distance", "label", "prediction", 
                   "pace_real_min_km", "pace_pred_min_km").show(10, truncate=False)

spark.stop()
print("=== PROCESO COMPLETADO ===")
