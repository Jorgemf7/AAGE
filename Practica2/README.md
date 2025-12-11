# Práctica 2 — Aprendizaje Automático a Gran Escala

**Aprendizaje Federado (Flower) y Aprendizaje Continuo (River)**

Esta práctica se divide en dos partes:
1. **Aprendizaje Federado** utilizando Flower con las estrategias FedAvg y FedProx sobre Fashion-MNIST.  
2. **Aprendizaje Continuo** utilizando la librería River, evaluando modelos incrementales en flujo de datos.

---

## 1. Instalación

El proyecto incluye un `requirements.txt` generado directamente desde el entorno de desarrollo, por lo que contiene todas las dependencias necesarias.

Para recrear el entorno:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## 2. Estructura del proyecto

- `task.py`: descarga Fashion-MNIST, calcula normalización real, genera particiones Non-IID y histogramas.
- `client_app.py`: lógica del cliente Flower (entrenamiento local y evaluación).
- `server_app.py`: lógica del servidor, estrategia y guardado de resultados.
- `plot_results.py`: genera gráficas comparativas de loss y accuracy.
- `river.ipynb`: Parte II de aprendizaje continuo con River.

Carpetas generadas automáticamente:

- `results/`: CSV con métricas por ronda.
- `plots/`: gráficas generadas.
- `histograms/`: histogramas Non-IID.

---

## 3. Parte 1 - Aprendizaje Federado (Flower)

### 1. Ejecución

#### 1.1 Configurar el experimento

Editar `pyproject.toml`:

```toml
model-type = "mlp"       # o "cnn"
strategy = "fedavg"      # o "fedprox"
proximal-mu = 0.01       # solo para FedProx
num-server-rounds = 10
```

#### 1.2 Generar particiones e histogramas (solo una vez)

```bash
python task.py
```

#### 1.3 Ejecutar la simulación federada

```bash
flwr run .
```

Los resultados aparecerán en la carpeta `results/`.

#### 1.4 Generar gráficas comparativas

```bash
python plots_results.py
```

Las gráficas aparecerán en `plots/`.

---

### 2. Resultados generados

- `results/*.csv`: métricas por ronda (accuracy y loss).
- `plots/*.png`: gráficas de resultados FedAvg vs FedProx (MLP y CNN) y comparativa global.
- `histograms/*.png`: hisotgramas de distribución Non-IID por cliente.

---

## 4. Parte II — Aprendizaje Continuo (River)

La segunda parte se desarrolla en el notebook `river.ipynb`, donde se implementa aprendizaje incremental sobre flujos de datos.

### 1 Contenido del notebook

Incluye:
- Simulación de flujo de datos.
- Entrenamiento incremental (online learning).
- Evaluación prequential (test-then-train).
- Modelos: NaiveBayes, LogisticRegression incremental, HoeffdingTreeClassifier, AdaptiveRandomForestClassifier.
- Comparación batch vs incremental.
- Gráficas de evolución temporal del rendimiento.

### 2. Ejecución

```bash
jupyter notebook river.ipynb
```

### 3 Resultados esperados

- Curvas de accuracy incremental.
- Evaluación prequential.
- Comparación entre modelos incrementales.
- Análisis del concept drift y adaptación de los modelos.

---

## 5. Requisitos de entrega

### Parte I (Flower)
- CSV generados.
- Gráficas comparativas.
- Discusión FedAvg vs FedProx.
- Comparación MLP vs CNN.
- Histogramas Non-IID.

### Parte II (River)
- Notebook ejecutado.
- Gráficas de aprendizaje incremental.
- Conclusiones sobre concept drift.

