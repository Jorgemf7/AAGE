import pandas as pd
import matplotlib.pyplot as plt
import os

# Carpetas de entrada (csv) y salida (imágenes)
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Crear carpeta de plots si no existe
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Función auxiliar para graficar si los datos existen
def plot_comparison(df_avg, df_prox, title_suffix, filename_suffix):
    # Rutas de salida
    loss_path = os.path.join(PLOTS_DIR, f"{filename_suffix}_loss_comparison.png")
    acc_path = os.path.join(PLOTS_DIR, f"{filename_suffix}_accuracy_comparison.png")

    # --- Loss Plot ---
    plt.figure()
    if df_avg is not None:
        plt.plot(df_avg["Round"], df_avg["Loss"], marker="o", label=f"FedAvg {title_suffix}")
    if df_prox is not None:
        plt.plot(df_prox["Round"], df_prox["Loss"], marker="o", label=f"FedProx {title_suffix}")
    
    plt.xlabel("Ronda")
    plt.ylabel("Pérdida")
    plt.title(f"{title_suffix}: Pérdida por ronda")
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_path)
    print(f"Gráfica guardada en: {loss_path}")

    # --- Accuracy Plot ---
    plt.figure()
    if df_avg is not None:
        plt.plot(df_avg["Round"], df_avg["Accuracy"], marker="o", label=f"FedAvg {title_suffix}")
    if df_prox is not None:
        plt.plot(df_prox["Round"], df_prox["Accuracy"], marker="o", label=f"FedProx {title_suffix}")

    plt.xlabel("Ronda")
    plt.ylabel("Accuracy")
    plt.title(f"{title_suffix}: Accuracy por ronda")
    plt.grid(True)
    plt.legend()
    plt.savefig(acc_path)
    print(f"Gráfica guardada en: {acc_path}")

def load_csv(filename):
    # Construir ruta completa hacia la carpeta results
    full_path = os.path.join(RESULTS_DIR, filename)
    
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        print(f"Advertencia: No se encontró {full_path}, saltando...")
        return None

# --- CARGA DE DATOS ---
print("--- Cargando resultados MLP ---")
df_fedavg_mlp = load_csv("results_fedavg(mlp).csv")
df_fedprox_mlp = load_csv("results_fedprox(mlp).csv")

print("--- Cargando resultados CNN ---\n")
df_fedavg_cnn = load_csv("results_fedavg(cnn).csv")
df_fedprox_cnn = load_csv("results_fedprox(cnn).csv")

# --- GENERACIÓN DE GRÁFICAS ---
if df_fedavg_mlp is not None or df_fedprox_mlp is not None:
    plot_comparison(df_fedavg_mlp, df_fedprox_mlp, "MLP", "mlp")

if df_fedavg_cnn is not None or df_fedprox_cnn is not None:
    plot_comparison(df_fedavg_cnn, df_fedprox_cnn, "CNN", "cnn")

print("\nProceso finalizado.")