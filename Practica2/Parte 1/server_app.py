"""Fashion-MNIST: Flower ServerApp."""

import csv
import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from task import create_model, load_centralized_dataset, test

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Entry point del ServerApp."""

    cfg = context.run_config
    model_type = cfg.get("model-type", "mlp")
    strategy_name = cfg.get("strategy", "fedavg").lower()
    num_rounds = int(cfg.get("num-server-rounds", 10))
    
    # Parámetros Strategy
    fraction_train = float(cfg.get("fraction-fit", 1.0)) 
    fraction_eval = float(cfg.get("fraction-evaluate", 1.0))
    min_nodes = int(cfg.get("min-available-clients", 2))

    # Parámetros Entrenamiento (config para cliente)
    lr = float(cfg.get("learning-rate", 0.01))
    local_epochs = int(cfg.get("local-epochs", 1))
    proximal_mu = float(cfg.get("proximal-mu", 0.0))

    print(f"--- Arrancando ServerApp ---")
    print(f"Estrategia: {strategy_name.upper()} | Modelo: {model_type.upper()}")

    # --- CONFIGURACIÓN CSV ---
    # Nombre de la carpeta de resultados
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Nombre dinámico del archivo con ruta incluida
    filename = f"results_{strategy_name}({model_type}).csv"
    csv_filename = os.path.join(results_dir, filename)
    
    # Inicializar el archivo CSV con los encabezados
    # 'w' sobreescribe el archivo si ya existe, asegurando una ejecución limpia
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy", "Loss"])
    
    print(f"Los resultados se guardarán en: {csv_filename}")

    # Modelo global inicial
    global_model = create_model(model_type)
    arrays = ArrayRecord(global_model.state_dict())

    # Configuración de Estrategia
    strategy_args = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
        min_available_nodes=min_nodes,
    )

    if strategy_name == "fedprox":
        strategy = FedProx(proximal_mu=proximal_mu, **strategy_args)
    else:
        strategy = FedAvg(**strategy_args)

    train_config = ConfigRecord({
        "lr": lr,
        "local_epochs": local_epochs,
        "proximal_mu": proximal_mu if strategy_name == "fedprox" else 0.0,
        "model_type": model_type 
    })

    # Wrapper para pasar el nombre del archivo CSV a la función de evaluación
    def evaluate_fn(server_round: int, arrays: ArrayRecord):
        return global_evaluate(server_round, arrays, model_type, csv_filename)

    # Iniciar estrategia
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn, 
    )

    print("\nSimulación finalizada.")


def global_evaluate(server_round: int, arrays: ArrayRecord, model_type: str, csv_filename: str) -> MetricRecord:
    """Evaluación centralizada en el servidor."""
    model = create_model(model_type)
    model.load_state_dict(arrays.to_torch_state_dict())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = load_centralized_dataset()

    test_loss, test_acc = test(model, test_loader, device)
    
    print(f"Ronda {server_round} | Global Acc: {test_acc:.4f} | Loss: {test_loss:.4f}")

    # --- GUARDAR EN CSV ---
    # Guardamos siempre, incluso la ronda 0 (inicial) si se desea ver el punto de partida
    with open(csv_filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([server_round, test_acc, test_loss])

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})