# Práctica 2 — Aprendizaje Automático a Gran Escala

**Aprendizaje Federado (Flower) y Aprendizaje Continuo (River)**

Esta práctica se divide en dos partes:
1. **Aprendizaje Federado** utilizando Flower con las estrategias FedAvg y FedProx sobre Fashion-MNIST.  
2. **Aprendizaje Continuo** utilizando la librería River, evaluando modelos incrementales en flujo de datos sobre Electricity.

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

```text
Practica2/
│
├── Parte 1/                    # Aprendizaje Federado con Flower
│   ├── histograms/             # (Generado) Imágenes de distribución de datos (Non-IID)
│   ├── plots/                  # (Generado) Gráficas comparativas de loss y accuracy
│   ├── results/                # (Generado) Archivos CSV con métricas por ronda
│   ├── client_app.py           # Lógica del cliente Flower (entrenamiento/evaluación)
│   ├── plots_results.py        # Generación de gráficas desde los CSV
│   ├── pyproject.toml          # Configuración del proyecto y dependencias
│   ├── server_app.py           # Lógica del servidor y estrategia de agregación
│   └── task.py                 # Descarga, normalización y particionado de datos
│
├── Parte 2/                    # Aprendizaje Continuo con River
│   ├── river.ipynb             # Notebook principal de la Parte II
│   └── *.png                   # Gráficas resultantes (Batch vs Streaming, Drifts, etc.)
│
├── Memoria Práctica 2.pdf   
│ 
└── README.md
```

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
fraction-fit = 1.0          
min-available-clients = 10  
local-epochs = 10            
batch-size = 64
learning-rate = 0.01
```

#### 1.2 Generar particiones e histogramas (solo una vez)

```bash
python task.py
```

Las gráficas aparecerán en `histograms/`.

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

### 2. Resultados generados

- `results/*.csv`: métricas por ronda (accuracy y loss).
- `plots/*.png`: gráficas de resultados FedAvg vs FedProx (MLP y CNN) y comparativa global.
- `histograms/*.png`: hisotgramas de distribución Non-IID por cliente.

---

## 4. Parte II — Aprendizaje Continuo (River)

La segunda parte se desarrolla en el notebook `river.ipynb`, donde se implementa aprendizaje incremental y estrategias adaptativas sobre el conjunto de datos de Electricity (Elec2).

### 1. Contenido del notebook

- Comparación Batch vs. Streaming: Contrastación de rendimiento entre un enfoque estático (GaussianNB de Scikit-learn) y uno incremental (GaussianNB de River).
- Manejo de Concept Drift: Implementación del detector ADWIN para identificar cambios en la distribución de datos y reiniciar el modelo automáticamente.
- Modelos Adaptativos Avanzados: Implementación y evaluación de árboles y ensambles diseñados para flujos cambiantes:
    - `HoeffdingAdaptiveTreeClassifier` (HAT).
    - `AdaptiveRandomForestClassifier` (ARF).
- Evaluación Progresiva: Uso de la métrica accuracy actualizada instancia a instancia (test-then-train).


### 2. Ejecución

```bash
jupyter notebook river.ipynb
```

### 3 Resultados esperados

- Gráfica comparativa: Precisión fija (Batch) vs. Precisión evolutiva (Streaming).
- Visualización de los momentos de drift detectados por ADWIN.
- Comparativa de rendimiento entre modelos de árboles adaptativos (HAT vs. ARF).
