"""fmnist_example: Servidor Flower con estrategias FedAvg/FedProx/Scaffold."""

from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx, Scaffold

from sklearn_example.task import create_model, get_model_parameters


# --- Agregador de métricas personalizado (media ponderada por nº muestras) ---

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """
    Promedia las métricas de los clientes ponderando por su tamaño de dataset.
    metrics: lista de (num_samples, {"metric": valor, ...})
    """
    results: Dict[str, float] = {}
    total = sum(num_samples for num_samples, _ in metrics)

    if total == 0:
        return results

    for num_samples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (float, int)):
                if k not in results:
                    results[k] = 0.0
                results[k] += v * num_samples / total

    return results


def server_fn(context: Context) -> ServerAppComponents:
    """Crea los componentes del servidor (estrategia + configuración)."""

    model_type = context.run_config["model-type"]         # "mlp" o "cnn"
    fraction_fit = context.run_config["fraction-fit"]     # proporción de clientes por ronda
    min_available_clients = context.run_config["min-available-clients"]
    num_rounds = context.run_config["num-server-rounds"]

    # Estrategia: "fedavg", "fedprox" o "scaffold"
    strategy_name = context.run_config.get("strategy", "fedavg").lower()
    proximal_mu = float(context.run_config.get("proximal-mu", 0.1))

    # Modelo inicial
    model = create_model(model_type)
    ndarrays = get_model_parameters(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Elegir estrategia
    if strategy_name == "fedprox":
        strategy = FedProx(
            proximal_mu=proximal_mu,
            fraction_fit=fraction_fit,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=global_model_init,
        )
    elif strategy_name == "scaffold":
        strategy = Scaffold(
            fraction_fit=fraction_fit,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=global_model_init,
        )
    else:  # FedAvg por defecto
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=global_model_init,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
