"""Fashion-MNIST: Flower server app with FedAvg and FedProx (logging enabled).

Este archivo define el servidor federado para Flower usando PyTorch.
Contiene:
1. CSVLogger: para guardar métricas por ronda automáticamente.
2. Función weighted_average: para agregar métricas de los clientes.
3. Función server_fn: construye la estrategia federada (FedAvg o FedProx) 
   y configura el servidor.
4. Objeto ServerApp para ejecutar la simulación con Flower.
"""

from typing import Dict, List, Tuple
import csv
import os

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from task import create_model, get_model_parameters


# =========================================================
# CSV LOGGER
# =========================================================
class CSVLogger:
    """
    Guarda métricas (loss y val_accuracy) de cada ronda en un CSV.
    """

    def __init__(self, filename: str):
        self.filename = filename
        # Crear fichero con cabecera si no existe
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "loss", "val_accuracy"])

    def log(self, rnd: int, loss: float, val_acc: float):
        """Añade una fila al CSV con métricas de la ronda."""
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rnd, loss, val_acc])


# =========================================================
# MÉTRICA AGREGADA
# =========================================================
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """
    Agrega métricas de los clientes usando media ponderada por número de ejemplos.
    """
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}

    results: Dict[str, float] = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (float, int)):
                results[k] = results.get(k, 0) + v * num_examples / total
    return results


# =========================================================
# FUNCION SERVER_FN
# =========================================================
def server_fn(context: Context) -> ServerAppComponents:
    """
    Construye los componentes del servidor federado para Flower.
    - Selecciona estrategia (FedAvg o FedProx) según config.
    - Inicializa modelo global.
    - Configura logging de métricas.
    """
    cfg = context.run_config

    # ----------------------------
    # LEER CONFIGURACIÓN
    # ----------------------------
    model_type = cfg.get("model-type", "mlp")
    strategy_name = cfg.get("strategy", "fedavg").lower()
    num_rounds = int(cfg.get("num-server-rounds", 10))
    fraction_fit = float(cfg.get("fraction-fit", 0.5))
    min_available_clients = int(cfg.get("min-available-clients", 10))
    local_epochs = int(cfg.get("local-epochs", 1))
    proximal_mu = float(cfg.get("proximal-mu", 0.1))
    lr = float(cfg.get("learning-rate", 0.005))

    # ----------------------------
    # LOGGER CSV
    # ----------------------------
    log_file = f"results_{strategy_name}({model_type}).csv"
    logger = CSVLogger(log_file)

    # ----------------------------
    # MODELO GLOBAL
    # ----------------------------
    model = create_model(model_type)
    init_params = ndarrays_to_parameters(get_model_parameters(model))

    # ----------------------------
    # CONFIGURACIONES DE CLIENTE
    # ----------------------------
    def fit_config(rnd: int):
        """Hiperparámetros enviados a los clientes durante el entrenamiento."""
        return {
            "local_epochs": local_epochs,
            "proximal_mu": proximal_mu if strategy_name == "fedprox" else 0.0,
            "learning_rate": lr,
        }

    def eval_config(rnd: int):
        """Hiperparámetros enviados a los clientes durante la evaluación."""
        return {}

    # ----------------------------
    # ELEGIR ESTRATEGIA
    # ----------------------------
    if strategy_name == "fedprox":
        strategy = FedProx(
            proximal_mu=proximal_mu,
            fraction_fit=fraction_fit,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=init_params,
        )
    else:
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            min_available_clients=min_available_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=init_params,
        )

    # ----------------------------
    # WRAPPEAR MÉTRICAS PARA LOGGING
    # ----------------------------
    old_aggregate_evaluate = strategy.aggregate_evaluate

    def aggregate_evaluate_with_logging(
        rnd: int, results, failures
    ) -> Tuple[float, Dict[str, Scalar]]:
        """
        Agrega métricas y guarda automáticamente en CSV.
        """
        loss, metrics = old_aggregate_evaluate(rnd, results, failures)
        val_acc = metrics.get("val_accuracy", None)
        if loss is not None and val_acc is not None:
            logger.log(rnd, float(loss), float(val_acc))
        return loss, metrics

    strategy.aggregate_evaluate = aggregate_evaluate_with_logging

    # ----------------------------
    # CONFIGURACIÓN DEL SERVIDOR
    # ----------------------------
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# =========================================================
# SERVER APP
# =========================================================
# Se utiliza para lanzar la simulación con Flower
app = ServerApp(server_fn=server_fn)