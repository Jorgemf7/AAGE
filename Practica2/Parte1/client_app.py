"""fmnist_example: Cliente Flower con PyTorch y Fashion-MNIST."""

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


class FlowerClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    # --------- métodos requeridos por NumPyClient ---------

    def get_parameters(self, config):
        """Devuelve los parámetros actuales del modelo local."""
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Entrenamiento local en el cliente."""
        # Cargar pesos globales
        set_model_params(self.model, parameters)

        # Hiperparámetros específicos de FL (config dict viene del servidor)
        local_epochs = int(config.get("local_epochs", 1))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Entrenamiento local (FedAvg si proximal_mu = 0, FedProx si > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_one_round(
                self.model,
                self.train_loader,
                device=self.device,
                epochs=local_epochs,
                proximal_mu=proximal_mu,
                global_params=parameters,
            )

        # Métrica de entrenamiento (accuracy local)
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


def client_fn(context: Context):
    """Construye el cliente con su partición de datos."""

    # Info de partición (nº de cliente y nº total de clientes)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Hiperparámetros de ejecución
    model_type = context.run_config["model-type"]  # "mlp" o "cnn"
    batch_size = int(context.run_config.get("batch-size", 32))

    # Cargar datos para este cliente
    train_loader, val_loader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    # Crear modelo y seleccionar dispositivo
    model = create_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return FlowerClient(model, train_loader, val_loader, device).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
