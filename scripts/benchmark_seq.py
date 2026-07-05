import os
import subprocess
import re
import csv

def parse_output(output):
    correct = "NO"
    t_comm = 0.0
    t_comp = 0.0
    t_total = 0.0
    flops = 0
    gflops = 0.0

    m_correct = re.search(r"Ordenamiento correcto:\s+(SÍ|NO)", output)
    if m_correct:
        correct = m_correct.group(1)

    m_comm = re.search(r"Tiempo de Comunicación.*:\s+([0-9.e-]+)\s*s", output)
    if m_comm:
        t_comm = float(m_comm.group(1))

    m_comp = re.search(r"Tiempo de Cómputo Local.*:\s+([0-9.e-]+)\s*s", output)
    if m_comp:
        t_comp = float(m_comp.group(1))

    m_total = re.search(r"Tiempo Total del Proceso Root:\s+([0-9.e-]+)\s*s", output)
    if m_total:
        t_total = float(m_total.group(1))

    m_flops = re.search(r"Cómputo Total Realizado.*:\s+(\d+)\s+FLOPs\s+\(([0-9.e-]+)\s+GFLOP/s\)", output)
    if m_flops:
        flops = int(m_flops.group(1))
        gflops = float(m_flops.group(2))
    return correct, t_comm, t_comp, t_total, flops, gflops

def run_experiment(algorithm, size_label, N, distribution, iterations=3):
    bin_path = f"./bin/{algorithm}"
    data_file = f"data/{size_label}/{distribution}.bin"
    
    cmd = [bin_path, data_file, str(N)]
    
    print(f"Ejecutando ({iterations} iteraciones): {bin_path} {data_file} {N}")
    t_comps = []
    last_flops = 0
    correct_all = "SÍ"
    any_error = False
    
    for i in range(iterations):
        try:
            timeout_val = 1800 if N >= 1000000 and "ranksort" in algorithm else 120
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_val)
            if res.returncode != 0:
                print(f"  -> Intento {i+1} falló: {res.stderr}")
                any_error = True
                break
            correct, t_comm, t_comp, t_total, flops, gflops = parse_output(res.stdout)
            if correct != "SÍ":
                correct_all = "NO"
            t_comps.append(t_comp)
            last_flops = flops
        except subprocess.TimeoutExpired:
            print(f"  -> Intento {i+1} superó el tiempo límite de {timeout_val}s")
            any_error = True
            break
        except Exception as e:
            print(f"  -> Intento {i+1} generó excepción: {str(e)}")
            any_error = True
            break
            
    if any_error or not t_comps:
        return "ERROR", 0.0, 0.0, 0.0, 0, 0.0
        
    avg_t_comp = sum(t_comps) / len(t_comps)
    avg_gflops = last_flops / (avg_t_comp * 1e9) if avg_t_comp > 0 else 0.0
    
    print(f"  -> Tiempo promedio: {avg_t_comp:.6f} s, GFLOP/s promedio: {avg_gflops:.4f}")
    return correct_all, 0.0, avg_t_comp, 0.0, last_flops, avg_gflops
 
def main():
    os.makedirs("results", exist_ok=True)
    csv_file = "results/benchmark_seq_results.csv"
    
    headers = ["Algorithm", "Distribution", "N", "Correct", "T_Comp", "FLOPs", "GFLOPs"]
    results = []
    
    algorithms = ["seq_ranksort", "seq_quicksort"]
    distributions = ["uniform", "sorted", "reverse", "gaussian"]
    
    sizes = {
        "10k": 10000,
        "100k": 100000,
        "1m": 1000000,
        "10m": 10000000,
        "100m": 100000000
    }
    
    for alg in algorithms:
        for size_label, N in sizes.items():
            if alg == "seq_ranksort" and N > 1000000:
                continue
            for dist in distributions:
                # ranksort 1M es O(N^2), pero se requiere obligatoriamente
                iters = 3
                
                correct, t_comm, t_comp, t_total, flops, gflops = run_experiment(
                    alg, size_label, N, dist, iterations=iters
                )

                results.append({
                    "Algorithm": alg,
                    "Distribution": dist,
                    "N": N,
                    "Correct": correct,
                    "T_Comp": t_comp,
                    "FLOPs": flops,
                    "GFLOPs": gflops
                })
                
                # Guardar progresivamente
                with open(csv_file, mode="w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(results)
                    
    print(f"\n¡Pruebas secuenciales completadas! Resultados guardados en {csv_file}")

if __name__ == "__main__":
    main()
