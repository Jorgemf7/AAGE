"""fmnist_example: Cliente Flower con PyTorch."""

import warnings
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import (
    create_model,
    load_data,
    train_one_round,
    test,
    get_model_parameters 
)

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    run_cfg = context.run_config
    model_type = run_cfg.get("model-type", "mlp")
    batch_size = int(run_cfg.get("batch-size", 32))
    lr = float(run_cfg.get("learning-rate", 0.01))
    local_epochs = int(run_cfg.get("local-epochs", 1))
    proximal_mu = float(run_cfg.get("proximal-mu", 0.0))

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, _ = load_data(partition_id, num_partitions, batch_size)

    # Cargar modelo con pesos del servidor
    model = create_model(model_type)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Params para FedProx
    global_params = [val.cpu().numpy() for _, val in model.state_dict().items()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_loss = train_one_round(
            model,
            train_loader,
            device=device,
            epochs=local_epochs,
            proximal_mu=proximal_mu,
            global_params=global_params,
            lr=lr,
        )

    # Devolver pesos actualizados
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(train_loader.dataset),
    }
    return Message(content=RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    run_cfg = context.run_config
    model_type = run_cfg.get("model-type", "mlp")
    batch_size = int(run_cfg.get("batch-size", 32))
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, val_loader = load_data(partition_id, num_partitions, batch_size)

    model = create_model(model_type)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss, acc = test(model, val_loader, device)

    metrics = {
        "eval_loss": loss,
        "eval_acc": acc,
        "num-examples": len(val_loader.dataset),
    }
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)