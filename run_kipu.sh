#!/bin/bash
#SBATCH --job-name=sort_scaling
#SBATCH --output=sort_scaling_%j.out
#SBATCH --error=sort_scaling_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --partition=standard

# Un solo nodo: -march=native del Makefile exige compilar y ejecutar en la misma CPU.
# 32 cores = tope de cuenta (pregrado/tesis) y tope interno de los scripts (max_cores=32).

set -e  # si un benchmark falla, el job termina en error (no imprime "Listo" en falso)

# Módulos de Khipu (OpenHPC): openmpi4 provee mpirun; python3 corre los scripts.
# NO se compila aquí: los nodos de cómputo no tienen headers de C. Compilar en el
# nodo de acceso con `make ARCH=skylake-avx512` ANTES del sbatch.
module load gnu12/12.4.0 openmpi4/4.1.6 python3/3.11.11

# Default OMP; benchmark_mpi/hybrid lo sobreescriben por corrida.
export OMP_NUM_THREADS=1

# Baseline de speedup = config (1,1) = mpirun -np 1 (medido en Khipu, mismo hardware).
echo "Ejecutando Benchmarks MPI en Khipu..."
python3 scripts/benchmark_mpi.py --kipu

echo "Ejecutando Benchmarks Hibridos en Khipu..."
python3 scripts/benchmark_hybrid.py --kipu

echo "Listo. Resultados en results/"
