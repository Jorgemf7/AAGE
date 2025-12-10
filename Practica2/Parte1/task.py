import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from flwr.common import NDArrays

# =========================================================
# CONFIGURACIÓN GLOBAL Y SEMILLAS
# =========================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cachés globales
_train_dataset = None
_test_dataset = None
_partitions = None

# =========================================================
# FUNCIONES DE CARGA DE DATOS
# =========================================================

def _calculate_stats():
    """
    Carga el dataset de entrenamiento temporalmente para calcular
    la media y desviación típica reales de los píxeles.
    """
    print("Calculando estadísticas de normalización (Media y Std)...")
    # Cargamos datos sin normalizar (solo ToTensor)
    temp_transform = transforms.Compose([transforms.ToTensor()])
    temp_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=temp_transform)
    
    # Cargamos todo el dataset en un solo batch para cálculo vectorizado rápido
    # num_workers=0 es importante en Windows
    loader = DataLoader(temp_dataset, batch_size=len(temp_dataset), shuffle=False, num_workers=0)
    
    data, _ = next(iter(loader))
    
    mean = data.mean().item()
    std = data.std().item()
    
    print(f"Estadísticas calculadas -> Mean: {mean:.4f}, Std: {std:.4f}")
    return (mean,), (std,)

def get_datasets():
    """Carga Fashion-MNIST con normalización calculada dinámicamente."""
    global _train_dataset, _test_dataset
    if _train_dataset is None or _test_dataset is None:
        
        # 1. Calculamos los valores reales antes de definir la transformación final
        mean_val, std_val = _calculate_stats()

        # 2. Usamos esos valores en la normalización
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_val, std_val)
        ])
        
        _train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        _test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return _train_dataset, _test_dataset

def create_partitions(num_partitions: int):
    """Crea particionado NO IID (2-3 clases por cliente)."""
    global _partitions
    if _partitions is not None:
        return _partitions

    train_ds, _ = get_datasets()
    
    # --- CORRECCIÓN DEL ERROR DE NUMPY ---
    if isinstance(train_ds.targets, torch.Tensor):
        labels = train_ds.targets.numpy()
    else:
        labels = np.array(train_ds.targets)
    
    idxs = np.arange(len(labels))

    idxs_per_class = [idxs[labels == c] for c in range(NUM_CLASSES)]
    rng = np.random.default_rng(SEED)

    client_classes: List[List[int]] = []
    for _ in range(num_partitions):
        base = rng.choice(NUM_CLASSES, size=2, replace=False)
        classes = set(base.tolist())
        if rng.random() < 0.5:
            extra = rng.integers(0, NUM_CLASSES)
            while extra in classes:
                extra = rng.integers(0, NUM_CLASSES)
            classes.add(int(extra))
        client_classes.append(sorted(classes))

    for c in range(NUM_CLASSES):
        if not any(c in cc for cc in client_classes):
            candidates = [i for i, cc in enumerate(client_classes) if len(cc) < 3]
            if not candidates:
                candidates = list(range(num_partitions))
            i = rng.choice(candidates)
            client_classes[i].append(c)

    partitions: List[List[int]] = [[] for _ in range(num_partitions)]
    for c in range(NUM_CLASSES):
        clients_with_c = [i for i, cc in enumerate(client_classes) if c in cc]
        idxs_c = idxs_per_class[c].copy()
        rng.shuffle(idxs_c)
        splits = np.array_split(idxs_c, len(clients_with_c))
        for cid, split in zip(clients_with_c, splits):
            partitions[cid].extend(split.tolist())

    for cid in range(num_partitions):
        rng.shuffle(partitions[cid])

    _partitions = partitions
    return _partitions

def load_data(partition_id: int, num_partitions: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    global _partitions
    train_ds, _ = get_datasets()
    if _partitions is None:
        _partitions = create_partitions(num_partitions)

    indices = _partitions[partition_id]
    subset = Subset(train_ds, indices)
    n_total = len(subset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    gen = torch.Generator().manual_seed(SEED + partition_id)
    train_subset, val_subset = random_split(subset, [n_train, n_val], generator=gen)

    # num_workers=0 es vital en Windows para evitar errores de Pickle en Ray
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def load_centralized_dataset() -> DataLoader:
    _, test_ds = get_datasets()
    return DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

# =========================================================
# MODELOS
# =========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        
        # Corregido: tenías dos definiciones de self.dropout
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_model(model_type: str):
    if model_type.lower() == "mlp":
        return MLP()
    elif model_type.lower() == "cnn":
        return SmallCNN()
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def get_model_parameters(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_params(model: nn.Module, params: NDArrays) -> nn.Module:
    state_dict = model.state_dict()
    new_state_dict = {}
    for (name, old_val), p in zip(state_dict.items(), params):
        new_state_dict[name] = torch.tensor(p, dtype=old_val.dtype)
    model.load_state_dict(new_state_dict, strict=True)
    return model

# =========================================================
# ENTRENAMIENTO Y TEST
# =========================================================
def train_one_round(model: nn.Module, train_loader: DataLoader, device: torch.device,
                    epochs: int = 1, proximal_mu: float = 0.0, global_params: NDArrays = None, lr: float = 0.01):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_tensors = None
    if proximal_mu > 0.0 and global_params is not None:
        state_vals = list(model.state_dict().values())
        global_tensors = [torch.tensor(p, dtype=state_vals[i].dtype, device=device)
                          for i, p in enumerate(global_params)]

    running_loss = 0.0
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            if proximal_mu > 0.0 and global_tensors is not None:
                prox_loss = 0.0
                for param, g_param in zip(model.state_dict().values(), global_tensors):
                    prox_loss += ((param - g_param) ** 2).sum()
                loss += (proximal_mu / 2.0) * prox_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    return running_loss / len(train_loader)

def test(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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
            total_loss += criterion(outputs, target).item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
            
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Generando particiones y guardando histogramas...")
    
    parts = create_partitions(num_partitions=10)
    train_ds, _ = get_datasets()
    
    # Mismo fix aquí para el main
    if isinstance(train_ds.targets, torch.Tensor):
        labels = train_ds.targets.numpy()
    else:
        labels = np.array(train_ds.targets)
    
    if not os.path.exists("histograms"):
        os.makedirs("histograms")

    for cid, idxs in enumerate(parts):
        y_client = labels[idxs]
        counts = [np.sum(y_client == c) for c in range(NUM_CLASSES)]
        plt.figure()
        plt.bar(range(NUM_CLASSES), counts)
        plt.title(f"Cliente {cid}")
        plt.savefig(f"histograms/hist_client_{cid}.png")
        plt.close()
    
    print("Histogramas guardados.")