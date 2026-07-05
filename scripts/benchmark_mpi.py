import os
import subprocess
import re
import csv
import argparse

def parse_output(output):
    correct = "NO"
    t_comm = 0.0
    t_comp = 0.0
    t_total = 0.0
    flops = 0
    gflops = 0.0

    m_correct = re.search(r"Ordenamiento correcto:\s+(SI|NO)", output)
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
    else:
        m_flops2 = re.search(r"Cómputo Total Realizado:\s+(\d+)\s+FLOPs", output)
        if m_flops2:
            flops = int(m_flops2.group(1))
        m_gflops2 = re.search(r"\(([0-9.e-]+)\s+GFLOP/s\)", output)
        if m_gflops2:
            gflops = float(m_gflops2.group(1))

    return correct, t_comm, t_comp, t_total, flops, gflops

def run_experiment(algorithm, size_label, N, distribution, mpi_ranks, omp_threads, iterations=3):
    bin_path = f"./bin/{algorithm}"
    data_file = f"data/{size_label}/{distribution}.bin"
    
    cmd = ["mpirun", "-np", str(mpi_ranks), bin_path, data_file, str(N)]
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    
    print(f"Ejecutando ({iterations} iteraciones): OMP_NUM_THREADS={omp_threads} {' '.join(cmd)}")
    
    t_comms = []
    t_comps = []
    t_totals = []
    last_flops = 0
    correct_all = "SÍ"
    any_error = False

    for i in range(iterations):
        try:
            # Usamos timeout grande para 1M RankSort puro MPI
            timeout_val = 1800 if N >= 1000000 and "ranksort" in algorithm else 300
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_val)
            if res.returncode != 0:
                print(f"  -> Intento {i+1} falló: {res.stderr}")
                any_error = True
                break
            correct, t_comm, t_comp, t_total, flops, gflops = parse_output(res.stdout)
            if correct != "SÍ":
                correct_all = "NO"
            t_comms.append(t_comm)
            t_comps.append(t_comp)
            t_totals.append(t_total)
            last_flops = flops
        except subprocess.TimeoutExpired:
            print(f"  -> Intento {i+1} superó el tiempo límite")
            any_error = True
            break
        except Exception as e:
            print(f"  -> Intento {i+1} generó excepción: {str(e)}")
            any_error = True
            break

    if any_error or not t_comps:
        return "ERROR", 0.0, 0.0, 0.0, 0, 0.0

    avg_t_comm = sum(t_comms) / len(t_comms)
    avg_t_comp = sum(t_comps) / len(t_comps)
    avg_t_total = sum(t_totals) / len(t_totals)
    avg_gflops = last_flops / (avg_t_comp * 1e9) if avg_t_comp > 0 else 0.0

    print(f"  -> T_Comp promedio: {avg_t_comp:.6f} s, GFLOP/s promedio: {avg_gflops:.4f}")
    return correct_all, avg_t_comm, avg_t_comp, avg_t_total, last_flops, avg_gflops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kipu", action="store_true", help="Run with Kipu HPC configs")
    args = parser.parse_args()

    # Binarios ya compilados en el nodo de acceso (los de cómputo no tienen headers de C).
    os.makedirs("results", exist_ok=True)
    csv_file = "results/benchmark_mpi_results_kipu.csv" if args.kipu else "results/benchmark_mpi_results.csv"
    
    headers = [
        "Algorithm", "Distribution", "N", "MPI_Ranks", "OMP_Threads", "Total_Cores",
        "Correct", "T_Comm", "T_Comp", "T_Total", "FLOPs", "GFLOPs"
    ]
    
    if args.kipu:
        configs = [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1)]
    else:
        configs = [(1, 1), (2, 1), (4, 1)]
    
    sizes = {
        "10k": 10000,
        "100k": 100000,
        "1m": 1000000,
        "10m": 10000000,
        "100m": 100000000
    }
    
    algorithms = ["ranksort", "quicksort"]
    distributions = ["uniform", "sorted", "reverse", "gaussian"]
    
    results = []
    
    for alg in algorithms:
        for size_label, N in sizes.items():
            if alg == "ranksort" and N > 1000000:
                continue
            for dist in distributions:
                for mpi, omp in configs:
                    # En MPI puro, omitir si requiere más cores de los que hay (Kipu 32, local 8)
                    max_cores = 32 if args.kipu else 8
                    if mpi > max_cores:
                        continue
                        
                    correct, t_comm, t_comp, t_total, flops, gflops = run_experiment(
                        alg, size_label, N, dist, mpi, omp
                    )
                    
                    results.append({
                        "Algorithm": alg,
                        "Distribution": dist,
                        "N": N,
                        "MPI_Ranks": mpi,
                        "OMP_Threads": omp,
                        "Total_Cores": mpi * omp,
                        "Correct": correct,
                        "T_Comm": t_comm,
                        "T_Comp": t_comp,
                        "T_Total": t_total,
                        "FLOPs": flops,
                        "GFLOPs": gflops
                    })
                    
                    with open(csv_file, mode="w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(results)
                        
    print(f"\n¡Pruebas MPI completadas! Resultados guardados en {csv_file}")

if __name__ == "__main__":
    main()
