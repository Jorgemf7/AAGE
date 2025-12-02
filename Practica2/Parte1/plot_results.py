import pandas as pd
import matplotlib.pyplot as plt

# Leer archivos
df_fedavg_mlp = pd.read_csv("results_fedavg(mlp).csv")
df_fedavg_cnn = pd.read_csv("results_fedavg(cnn).csv")
df_fedprox_mlp = pd.read_csv("results_fedprox(mlp).csv")
df_fedprox_cnn = pd.read_csv("results_fedprox(cnn).csv")

# ---------------- MLP -----------------
# Loss
plt.figure()
plt.plot(df_fedavg_mlp["round"], df_fedavg_mlp["loss"], marker="o", label="FedAvg MLP")
plt.plot(df_fedprox_mlp["round"], df_fedprox_mlp["loss"], marker="o", label="FedProx MLP")
plt.xlabel("Ronda")
plt.ylabel("Pérdida")
plt.title("MLP: Pérdida por ronda")
plt.grid(True)
plt.legend()
plt.savefig("mlp_loss_comparison.png")

# Accuracy
plt.figure()
plt.plot(df_fedavg_mlp["round"], df_fedavg_mlp["val_accuracy"], marker="o", label="FedAvg MLP")
plt.plot(df_fedprox_mlp["round"], df_fedprox_mlp["val_accuracy"], marker="o", label="FedProx MLP")
plt.xlabel("Ronda")
plt.ylabel("Accuracy")
plt.title("MLP: Accuracy por ronda")
plt.grid(True)
plt.legend()
plt.savefig("mlp_accuracy_comparison.png")

# ---------------- CNN -----------------
# Loss
plt.figure()
plt.plot(df_fedavg_cnn["round"], df_fedavg_cnn["loss"], marker="o", label="FedAvg CNN")
plt.plot(df_fedprox_cnn["round"], df_fedprox_cnn["loss"], marker="o", label="FedProx CNN")
plt.xlabel("Ronda")
plt.ylabel("Pérdida")
plt.title("CNN: Pérdida por ronda")
plt.grid(True)
plt.legend()
plt.savefig("cnn_loss_comparison.png")

# Accuracy
plt.figure()
plt.plot(df_fedavg_cnn["round"], df_fedavg_cnn["val_accuracy"], marker="o", label="FedAvg CNN")
plt.plot(df_fedprox_cnn["round"], df_fedprox_cnn["val_accuracy"], marker="o", label="FedProx CNN")
plt.xlabel("Ronda")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy por ronda")
plt.grid(True)
plt.legend()
plt.savefig("cnn_accuracy_comparison.png")

print("Gráficas generadas para MLP y CNN.")