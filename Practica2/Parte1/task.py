import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from flwr.common import NDArrays

# Para que los experimentos sean reproducibles
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

NUM_CLASSES = 10  # Fashion-MNIST
IMAGE_SHAPE = (1, 28, 28)

# Evitar warnings raros de algunos entornos
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Cachés globales
_train_dataset = None
_test_dataset = None
_partitions = None  # lista de índices por cliente


# =============== CARGA DEL DATASET ===============

def get_datasets():
    """Carga Fashion-MNIST (train y test) con transformaciones estándar."""
    global _train_dataset, _test_dataset

    if _train_dataset is None or _test_dataset is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        _train_dataset = datasets.FashionMNIST(
            root="data", train=True, download=True, transform=transform
        )
        _test_dataset = datasets.FashionMNIST(
            root="data", train=False, download=True, transform=transform
        )

    return _train_dataset, _test_dataset


# =============== PARTICIONADO NO IID (2–3 CLASES/CLIENTE) ===============

def create_partitions(
    num_partitions: int,
    min_classes_per_client: int = 2,
    max_classes_per_client: int = 3,
):
    """
    Crea un particionado no IID:
      - Cada cliente ve solo 2 o 3 clases.
      - Las muestras de cada clase se reparten entre los clientes que la tienen.
    """
    global _partitions
    if _partitions is not None:
        return _partitions

    train_ds, _ = get_datasets()
    labels = np.array(train_ds.targets)
    idxs = np.arange(len(labels))

    # Índices por clase
    idxs_per_class = [idxs[labels == c] for c in range(NUM_CLASSES)]

    rng = np.random.default_rng(SEED)

    # 1) Elegir qué clases ve cada cliente (2 o 3)
    client_classes: List[List[int]] = []
    for _cid in range(num_partitions):
        base = rng.choice(NUM_CLASSES, size=min_classes_per_client, replace=False)
        classes = set(base.tolist())
        # ~50% de clientes tendrán una tercera clase
        if max_classes_per_client > min_classes_per_client and rng.random() < 0.5:
            extra = rng.integers(0, NUM_CLASSES)
            while extra in classes:
                extra = rng.integers(0, NUM_CLASSES)
            classes.add(int(extra))
        client_classes.append(sorted(classes))

    # 2) Asegurar que todas las clases aparecen al menos en un cliente
    for c in range(NUM_CLASSES):
        if not any(c in cc for cc in client_classes):
            # Buscar cliente con hueco (menos de max_classes_per_client)
            candidates = [i for i, cc in enumerate(client_classes) if len(cc) < max_classes_per_client]
            if not candidates:
                candidates = list(range(num_partitions))
            i = rng.choice(candidates)
            client_classes[i].append(c)

    # 3) Repartir muestras de cada clase entre los clientes que la tienen
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]
    for c in range(NUM_CLASSES):
        clients_with_c = [i for i, cc in enumerate(client_classes) if c in cc]
        idxs_c = idxs_per_class[c].copy()
        rng.shuffle(idxs_c)
        splits = np.array_split(idxs_c, len(clients_with_c))
        for cid, split in zip(clients_with_c, splits):
            partitions[cid].extend(split.tolist())

    # Mezclar dentro de cada cliente
    for cid in range(num_partitions):
        rng.shuffle(partitions[cid])

    _partitions = partitions
    return _partitions


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Devuelve DataLoader de train y val para el cliente partition_id."""

    global _partitions
    train_ds, _ = get_datasets()

    if _partitions is None:
        _partitions = create_partitions(num_partitions)

    indices = _partitions[partition_id]
    subset = Subset(train_ds, indices)

    # 80% train / 20% validación por cliente
    n_total = len(subset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(SEED + partition_id)

    train_subset, val_subset = random_split(subset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# =============== MODELOS ===============

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SmallCNN(nn.Module):
    """CNN pequeña para la extensión opcional."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(model_type: str):
    """Crea el modelo PyTorch según el tipo indicado."""
    if model_type == "mlp":
        return MLP()
    elif model_type == "cnn":
        return SmallCNN()
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type} (usa 'mlp' o 'cnn').")


# =============== AYUDAS PARA FL (parámetros y entrenamiento) ===============

def get_model_parameters(model: nn.Module) -> NDArrays:
    """Convierte el state_dict del modelo en lista de ndarrays para Flower."""
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: nn.Module, params: NDArrays) -> nn.Module:
    """Carga parámetros (lista de ndarrays) en un modelo PyTorch."""
    state_dict = model.state_dict()
    new_state_dict = {}
    for (name, old_val), p in zip(state_dict.items(), params):
        new_state_dict[name] = torch.tensor(p, dtype=old_val.dtype)
    model.load_state_dict(new_state_dict, strict=True)
    return model


def train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    proximal_mu: float = 0.0,
    global_params: NDArrays = None,
):
    """Entrenamiento local (soporta FedProx si proximal_mu > 0)."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    global_tensors = None
    if proximal_mu > 0.0 and global_params is not None:
        # Convertir parámetros globales a tensores para término proximal (FedProx)
        state_vals = list(model.state_dict().values())
        global_tensors = [
            torch.tensor(p, dtype=state_vals[i].dtype, device=device)
            for i, p in enumerate(global_params)
        ]

    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            # Término proximal de FedProx
            if proximal_mu > 0.0 and global_tensors is not None:
                prox = 0.0
                for w, w0 in zip(model.state_dict().values(), global_tensors):
                    prox = prox + torch.sum((w - w0) ** 2)
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            optimizer.step()


def test(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evalúa el modelo: devuelve (loss_media, accuracy)."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =============== HISTOGRAMAS POR CLIENTE (para la memoria) ===============

if __name__ == "__main__":
    """
    Ejecuta:
        python task.py
    y se generarán hist_client_0.png, hist_client_1.png, ..., uno por cliente.
    """
    import matplotlib.pyplot as plt

    train_ds, _ = get_datasets()
    labels = np.array(train_ds.targets)

    num_clients = 10
    parts = create_partitions(num_clients)

    for cid, idxs in enumerate(parts):
        y = labels[idxs]
        counts = [np.sum(y == c) for c in range(NUM_CLASSES)]

        plt.figure()
        plt.bar(range(NUM_CLASSES), counts)
        plt.xlabel("Clase")
        plt.ylabel("Número de muestras")
        plt.title(f"Distribución de clases - Cliente {cid}")
        plt.tight_layout()
        plt.savefig(f"hist_client_{cid}.png")
        plt.close()
