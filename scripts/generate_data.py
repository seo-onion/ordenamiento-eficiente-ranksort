import os
import numpy as np

def generate_binary_dataset(N, filename, distribution="uniform"):
    print(f"Generando dataset '{distribution}' con N = {N}...")
    if distribution == "uniform":
        data = np.random.uniform(0.0, 1000.0, N).astype(np.float64)
    elif distribution == "sorted":
        data = np.sort(np.random.uniform(0.0, 1000.0, N).astype(np.float64))
    elif distribution == "reverse":
        data = np.sort(np.random.uniform(0.0, 1000.0, N).astype(np.float64))[::-1]
    elif distribution == "gaussian":
        data = np.random.normal(500.0, 150.0, N).astype(np.float64)
    else:
        raise ValueError("Distribución no soportada")

    # Guardar en formato binario puro (double = float64, 8 bytes por elemento)
    data.tofile(filename)
    print(f"Guardado en {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Tamaños de prueba
    sizes = {
        "10k": 10_000,
        "100k": 100_000,
        "1m": 1_000_000,
        "10m": 10_000_000, 
        "100m": 100_000_000
    }
    
    distributions = ["uniform", "sorted", "reverse", "gaussian"]
    
    for label, N in sizes.items():
        os.makedirs(f"data/{label}", exist_ok=True)
        for dist in distributions:
            filename = f"data/{label}/{dist}.bin"
            generate_binary_dataset(N, filename, dist)
            
    print("\n¡Datasets generados con éxito en la carpeta 'data/'!")
