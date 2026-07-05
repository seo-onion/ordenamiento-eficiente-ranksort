#!/bin/bash
# Script para ejecutar las pruebas en local
echo "Iniciando compilación en local..."
make clean && make

echo "Ejecutando Benchmarks Secuenciales (N = 10k, 100k, 1M)..."
python3 scripts/benchmark_seq.py

echo "Ejecutando Benchmarks MPI (N = 10k, 100k, 1M)..."
python3 scripts/benchmark_mpi.py

echo "Ejecutando Benchmarks Híbridos (N = 10k, 100k, 1M)..."
python3 scripts/benchmark_hybrid.py

echo "¡Pruebas completadas con éxito localmente!"
