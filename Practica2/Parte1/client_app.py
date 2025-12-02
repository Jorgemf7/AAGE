"""fmnist_example: Cliente Flower con PyTorch y Fashion-MNIST.

Este archivo define el cliente federado para Flower usando PyTorch.
Contiene:
1. Clase FlowerClient que hereda de NumPyClient.
2. Función client_fn que construye el cliente según partición.
3. Objeto ClientApp para ejecución con Flower.
"""

import warnings
import torch
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from task import (
    create_model,
    get_model_parameters,
    load_data,
    set_model_params,
    train_one_round,
    test,
)

# =========================================================
# CLASE DEL CLIENTE
# =========================================================
class FlowerClient(NumPyClient):
    """Implementación de un cliente federado PyTorch para Flower."""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    # --------- Métodos requeridos por NumPyClient ---------

    def get_parameters(self, config):
        """Devuelve los parámetros actuales del modelo local."""
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Entrenamiento local en el cliente."""
        # Cargar pesos globales enviados por el servidor
        set_model_params(self.model, parameters)

        # Hiperparámetros específicos de FL
        local_epochs = int(config.get("local_epochs", 1))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Entrenamiento local (FedAvg si proximal_mu = 0, FedProx si > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignorar warnings de PyTorch
            train_one_round(
                self.model,
                self.train_loader,
                device=self.device,
                epochs=local_epochs,
                proximal_mu=proximal_mu,
                global_params=parameters,
            )

        # Métricas de entrenamiento
        train_loss, train_acc = test(self.model, self.train_loader, self.device)
        num_examples = len(self.train_loader.dataset)

        return (
            get_model_parameters(self.model),
            num_examples,
            {"train_loss": train_loss, "train_accuracy": train_acc},
        )

    def evaluate(self, parameters, config):
        """Evaluación local (validación) en el cliente."""
        set_model_params(self.model, parameters)
        val_loss, val_acc = test(self.model, self.val_loader, self.device)
        num_examples = len(self.val_loader.dataset)

        return val_loss, num_examples, {"val_accuracy": val_acc}


# =========================================================
# FUNCION CLIENT_FN
# =========================================================
def client_fn(context: Context):
    """
    Construye el cliente federado con la partición de datos correspondiente.
    - Obtiene el ID de cliente y número total de particiones.
    - Carga datos (train y val) usando task.load_data.
    - Crea modelo PyTorch y selecciona dispositivo.
    - Devuelve un cliente Flower listo para entrenar/evaluar.
    """
    # Info de partición
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Hiperparámetros
    model_type = context.run_config["model-type"]  # "mlp" o "cnn"
    batch_size = int(context.run_config.get("batch-size", 32))

    # Cargar datos
    train_loader, val_loader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    # Crear modelo y seleccionar dispositivo
    model = create_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Devolver cliente federado
    return FlowerClient(model, train_loader, val_loader, device).to_client()


# =========================================================
# CLIENTE APP
# =========================================================
# Se utiliza para lanzar la simulación con Flower
app = ClientApp(client_fn=client_fn)