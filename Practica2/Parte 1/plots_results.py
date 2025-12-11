import pandas as pd
import matplotlib.pyplot as plt
import os

# Carpetas
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def load_data(filename):
    """Carga el CSV si existe, sino avisa."""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"Advertencia: No se encontró {filename}")
        return None

def plot_experiment(experiment_name, files_dict):
    """
    Genera UNA imagen con dos gráficas (Accuracy y Loss) para un conjunto de archivos.
    
    Args:
        experiment_name (str): Título general y nombre del archivo de salida.
        files_dict (dict): Diccionario { "Etiqueta para Leyenda": "nombre_archivo.csv" }
    """
    print(f"--- Generando gráfica combinada para: {experiment_name} ---")
    
    # Preparamos 1 figura con 2 subplots (1 fila, 2 columnas)
    # figsize=(16, 6) hace la imagen más ancha para que quepan bien las dos gráficas
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))
    
    data_found = False

    for label, filename in files_dict.items():
        df = load_data(filename)
        if df is not None:
            data_found = True
            # Plot Accuracy en el primer eje (ax_acc)
            ax_acc.plot(df["Round"], df["Accuracy"], marker="o", linestyle="-", label=label)
            # Plot Loss en el segundo eje (ax_loss)
            ax_loss.plot(df["Round"], df["Loss"], marker="o", linestyle="-", label=label)

    if not data_found:
        print("No hay datos para graficar en este experimento.\n")
        plt.close(fig)
        return

    # --- Configuración Gráfica Accuracy (Izquierda) ---
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Ronda")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, linestyle='--', alpha=0.7)
    ax_acc.legend()

    # --- Configuración Gráfica Loss (Derecha) ---
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Ronda")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    ax_loss.legend()

    # Título global de la figura
    fig.suptitle(f"{experiment_name}", fontsize=16)
    
    # Ajustar espacios para que no se monten los textos
    plt.tight_layout()

    # Guardar archivo único
    output_filename = f"{experiment_name.replace(' ', '_')}_combined.png"
    save_path = os.path.join(PLOTS_DIR, output_filename)
    
    fig.savefig(save_path)
    print(f"Guardado: {save_path}")
    plt.close(fig)
    print("")


# EXPERIMENTOS

# 1. IMPACTO DE LOCAL EPOCHS
# Comparamos local epochs 1 vs 5 vs 10 epcohs 
experiments_le = {
    "local-epochs=1": "results_fedavg(mlp)_le1_frac1.0.csv",
    "local-epochs=5": "results_fedavg(mlp)_le5_frac1.0.csv",
    "local-epochs=10": "results_fedavg(mlp)_le10_frac1.0.csv",
}

# 2. IMPACTO DE FRACTION-FIT
# Comparamos fraction-fit 0.1 vs 0.4 vs 1.0 manteniendo local-epochs = 10
experiments_frac = {
    "fraction-fit=0.1 (10%)":   "results_fedavg(mlp)_le10_frac0.1.csv",
    "fraction-fit=0.4 (40%)":   "results_fedavg(mlp)_le10_frac0.4.csv",
    "fraction-fit=1.0 (Todos)": "results_fedavg(mlp)_le10_frac1.0.csv",
}

# 3. IMPACTO DE MU
# Comparamos mu 0.0 vs 0.01 vs 0.1 manteniendo local-epochs = 10 y fraction-fit = 1.0
experiments_prox = {
    "FedAvg (Baseline mu=0.0)": "results_fedavg(mlp)_le10_frac1.0.csv",
    "FedProx (mu=0.01)": "results_fedprox(mlp)_le10_frac1.0_mu0.01.csv",
    "FedProx (mu=0.1)":  "results_fedprox(mlp)_le10_frac1.0_mu0.1.csv",
}

# 4. COMPARATIVA DE MODELOS: MLP vs CNN
# Comparamos MLP con FedAvg(CNN) y FedProx(CNN) manteniendo local-epochs = 10 y fraction-fit = 1.0
experiments_models = {
    "FedAvg MLP": "results_fedavg(mlp)_le10_frac1.0.csv",
    "FedAvg CNN": "results_fedavg(cnn)_le10_frac1.0.csv",
    "FedProx (mu=0.01) CNN": "results_fedprox(cnn)_le10_frac1.0_mu0.01.csv",
}

# 5. COMPARATIVA GENERAL DE TODOS LOS MODELOS 
# Comparamos todos los modelos con local-epochs = 10 y fraction-fit = 1.0, que son los mejores hiperparámetros
experiments_general = {
    "FedAvg MLP": "results_fedavg(mlp)_le10_frac1.0.csv",
    "FedProx (mu=0.01) MLP": "results_fedprox(mlp)_le10_frac1.0_mu0.01.csv",
    "FedAvg CNN": "results_fedavg(cnn)_le10_frac1.0.csv",
    "FedProx (mu=0.01) CNN": "results_fedprox(cnn)_le10_frac1.0_mu0.01.csv",
}

if __name__ == "__main__":
    # Ejecutar gráficas
    plot_experiment("Impacto Local Epochs", experiments_le)
    plot_experiment("Impacto Fraction-Fit (Local-Epochs = 10)", experiments_frac)
    plot_experiment("Impacto Mu (Local-Epochs = 10 y Fraction-Fit = 1.0)", experiments_prox)
    plot_experiment("MLP vs CNN (Local-Epochs = 10 y Fraction-Fit = 1.0)", experiments_models)
    plot_experiment("Comparativa General de Modelos (Local-Epochs = 10 y Fraction-Fit = 1.0)", experiments_general)