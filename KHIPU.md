# Despliegue en Khipu — Runbook

Todo el flujo para correr el proyecto en el clúster HPC Khipu y recuperar resultados.
Usuario: `sebastian.hernandez` · Ruta local: `~/Workspace/Utec/Paralelas/Proyecto` · Ruta remota: `~/Proyecto`

> Los comandos empiezan con `!` para poder ejecutarlos desde la sesión de Claude Code.
> Si los corres en tu terminal normal, quita el `!`.

---

## 1. Subir SOLO lo esencial (código, sin data/bin/results)

`data/` se genera en el clúster (más abajo); `bin/` no es portable (`-march=native`); `results/` lo produce el job.

```
! rsync -azP \
    --exclude bin --exclude results --exclude data \
    --exclude recursos --exclude .claude --exclude '.git' \
    ~/Workspace/Utec/Paralelas/Proyecto \
    sebastian.hernandez@khipu.utec.edu.pe:~/
```

Sube: `Makefile`, `run_kipu.sh`, `src/`, `scripts/`. Pesa unos pocos KB.

---

## 2. Generar los datasets EN Khipu (nodo de acceso, tiene numpy)

Crea `data/{10k,100k,1m,10m,100m}/{uniform,sorted,reverse,gaussian}.bin` — que es justo lo que leen los scripts.

El módulo `python3` NO trae `numpy`; se instala una vez en tu espacio de usuario (`--user`):

```
! ssh sebastian.hernandez@khipu.utec.edu.pe \
    "cd ~/Proyecto && module load python3/3.11.11 && python3 -m pip install --user numpy && python3 scripts/generate_data.py"
```

> Si `pip` no está: `python3 -m ensurepip --user` y reintenta.

> Genera ~3.5 GB (incluye `100m`). Si NO quieres `100m`, edita `sizes` en
> `scripts/generate_data.py` y en `benchmark_mpi.py`/`benchmark_hybrid.py` antes de generar.

---

## 3. Compilar EN EL NODO DE ACCESO (obligatorio)

Los nodos de cómputo no tienen headers de C (`features.h`), así que el job NO compila:
hay que compilar aquí, con arch portable a los nodos `standard` (Skylake).

```
! ssh sebastian.hernandez@khipu.utec.edu.pe \
    "cd ~/Proyecto && module load gnu12/12.4.0 openmpi4/4.1.6 && make clean && make ARCH=skylake-avx512"
```

Debe compilar los 6 binarios sin errores. Si algún nodo diera `Illegal instruction`,
recompila con el arch ultra-seguro `make ARCH=x86-64-v3`.

---

## 4. Enviar el job

```
! ssh sebastian.hernandez@khipu.utec.edu.pe "cd ~/Proyecto && sbatch run_kipu.sh"
```

Devuelve algo como `Submitted batch job 12345`. Guarda ese número (JOBID).

---

## 5. Monitorear

```
! ssh sebastian.hernandez@khipu.utec.edu.pe "squeue --me"
```

Ver el log en vivo (reemplaza JOBID):

```
! ssh sebastian.hernandez@khipu.utec.edu.pe "tail -f ~/Proyecto/sort_scaling_JOBID.out"
```

Cancelar si hace falta:

```
! ssh sebastian.hernandez@khipu.utec.edu.pe "scancel JOBID"
```

---

## 6. Recibir los CSV (y logs) de vuelta en tu máquina

Al terminar el job, el clúster deja en `~/Proyecto/results/`:
- `benchmark_mpi_results_kipu.csv`
- `benchmark_hybrid_results_kipu.csv`

Descarga resultados + logs SLURM a tu carpeta local:

```
! rsync -azP \
    sebastian.hernandez@khipu.utec.edu.pe:'~/Proyecto/results/*_kipu.csv' \
    ~/Workspace/Utec/Paralelas/Proyecto/results/

! rsync -azP \
    sebastian.hernandez@khipu.utec.edu.pe:'~/Proyecto/sort_scaling_*.out' \
    sebastian.hernandez@khipu.utec.edu.pe:'~/Proyecto/sort_scaling_*.err' \
    ~/Workspace/Utec/Paralelas/Proyecto/results/
```

---

## Notas
- **Baseline de speedup:** la config `(1,1)` del sweep MPI (`np=1`, medida en Khipu). No se corre el secuencial en el clúster.
- **Cuenta:** verifica tus límites con `ssh ... "myaccount"`. `run_kipu.sh` pide 32 cores / 1 nodo / 7.5 h (tope de pregrado/tesis). Si tienes más, se puede subir.
- **Limpieza (opcional):** `ssh ... "rm ~/Proyecto/data/data_*.bin"` borra los binarios planos viejos que no usa ningún script.
