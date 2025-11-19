import pandas as pd
import matplotlib.pyplot as plt

df_avg = pd.read_csv("results_fedavg.csv")
df_prox = pd.read_csv("results_fedprox.csv")

# ---------------- PÉRDIDA -----------------
plt.figure()
plt.plot(df_avg["round"], df_avg["loss"], marker="o", label="FedAvg")
plt.plot(df_prox["round"], df_prox["loss"], marker="o", label="FedProx")
plt.xlabel("Ronda")
plt.ylabel("Pérdida")
plt.title("Pérdida global por ronda")
plt.grid(True)
plt.legend()
plt.savefig("loss_comparison.png")

# ---------------- ACCURACY -----------------
plt.figure()
plt.plot(df_avg["round"], df_avg["val_accuracy"], marker="o", label="FedAvg")
plt.plot(df_prox["round"], df_prox["val_accuracy"], marker="o", label="FedProx")
plt.xlabel("Ronda")
plt.ylabel("Accuracy global")
plt.title("Accuracy global por ronda")
plt.grid(True)
plt.legend()
plt.savefig("accuracy_comparison.png")

print("Gráficas generadas.")
    