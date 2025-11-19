from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from task import create_model, get_model_parameters  # o sklearn_example.task, según tu estructura


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    total = sum(num_samples for num_samples, _ in metrics)
    if total == 0:
        return {}
    results: Dict[str, float] = {}
    for num_samples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (float, int)):
                results[k] = results.get(k, 0.0) + v * num_samples / total
    return results


def server_fn(context: Context) -> ServerAppComponents:
    model_type = context.run_config["model-type"]      # "mlp" / "cnn" / lo que uses
    fraction_fit = context.run_config["fraction-fit"]
    min_available = context.run_config["min-available-clients"]
    num_rounds = context.run_config["num-server-rounds"]

    strategy_name = context.run_config.get("strategy", "fedavg").lower()
    proximal_mu = float(context.run_config.get("proximal-mu", 0.1))

    # Modelo inicial
    model = create_model(model_type)
    ndarrays = get_model_parameters(model)
    initial_params = ndarrays_to_parameters(ndarrays)

    if strategy_name == "fedprox":
        strategy = FedProx(
            proximal_mu=proximal_mu,
            fraction_fit=fraction_fit,
            min_available_clients=min_available,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_params,
        )
    else:  # FedAvg por defecto
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            min_available_clients=min_available,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_params,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
