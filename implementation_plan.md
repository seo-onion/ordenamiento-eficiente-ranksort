# Plan de Implementación: Comparación Progresiva de Escalabilidad (Secuencial vs. Pure MPI vs. Híbrido)

Este plan establece una metodología rigurosa para evaluar la escalabilidad y cuellos de botella de dos algoritmos de ordenamiento (**Rank Sort** y **Quicksort PSRS**) en el clúster HPC Kipu. La evaluación se realizará de forma estrictamente incremental.

---

## User Review Required

> [!IMPORTANT]
> **Fases de Desarrollo y Restricción de Compilación**
> 1. **Fase Secuencial:** Implementación de versiones secuenciales puras para establecer el baseline.
> 2. **Fase Pure MPI:** Ejecución distribuida con múltiples procesos MPI desactivando OpenMP (`OMP_NUM_THREADS=1`).
> 3. **Fase Híbrida (MPI + OpenMP):** Activación de OpenMP (`OMP_NUM_THREADS > 1`) para explotar hilos por nodo.
> 4. **RESTRICCIÓN:** No se compilará ni ejecutará código de la fase MPI hasta que la fase secuencial esté completamente validada.

---

## Estrategia de Datos y Carga Computacional

Para analizar de manera robusta la escalabilidad y la relación cómputo/comunicación sin introducir cuellos de botella ajenos al paralelismo, se establece lo siguiente:

1. **Uso Unificado de `Complex`:** 
   Se usará la estructura `Complex` (que almacena una parte real `r` y una parte imaginaria `i` como `double`) como el tipo de dato base para todas las pruebas. Esta estructura representa de forma contigua en memoria un vector de dos flotantes de doble precisión.
2. **Evaluación de Escenarios de Carga:**
   * **Baja carga computacional (Flotantes/Enteros):** Se evalúa considerando únicamente la parte real de los números generados, lo que minimiza el costo aritmético de comparación local y expone los límites impuestos por el ancho de banda y la latencia de la red de comunicación (MPI).
   * **Carga computacional media/alta (Complejos):** Se evalúa comparando la magnitud al cuadrado ($r^2 + i^2$), lo que añade un costo aritmético de **6 FLOPs por comparación** y resalta la aceleración lograda por la paralelización del cómputo local.
3. **Exclusión de Matrices Dinámicas:**
   Se descarta explícitamente ordenar colecciones de matrices de tamaño variable para evitar:
   * **Pérdida de Localidad de Caché:** La indirección de memoria de matrices dinámicas degrada el rendimiento de los procesadores modernos al impedir la prebúsqueda de datos (*hardware prefetching*).
   * **Saturación del Ancho de Banda de I/O:** Leer archivos binarios masivos de matrices desde el almacenamiento distribuido del clúster (Kipu) generaría un cuello de botella de lectura/escritura que enmascararía el comportamiento de la red paralela.

---

## Fases Propuestas

### Fase 1: Baseline Secuencial
Implementar versiones de un solo hilo y un solo proceso para ambos algoritmos, compartiendo la estructura de datos `Complex` basada en magnitudes al cuadrado.

- **Rank Sort Secuencial:** Complejidad $O(N^2)$.
- **Quicksort Secuencial:** Complejidad $O(N \log N)$ usando pivoteo *median-of-three* para evitar el peor caso.

### Fase 2: Paralelismo de Memoria Distribuida (Pure MPI con Comunicaciones No Bloqueantes)
Ejecutar los algoritmos usando únicamente paso de mensajes punto a punto no bloqueantes con OpenMP inactivo (`OMP_NUM_THREADS=1`). Se eliminarán todas las funciones colectivas de MPI y se reemplazarán de la siguiente manera:

- **Rank Sort MPI (Comunicaciones No Bloqueantes):**
  1. **Broadcast manual no bloqueante:** El proceso 0 envía el arreglo original `A` a todos los demás procesos usando `MPI_Isend`. Los procesos $1 \dots P-1$ lo reciben con `MPI_Irecv`. Se sincroniza con `MPI_Waitall`/`MPI_Wait`.
  2. **Cómputo local:** Cada proceso calcula los rangos para su subrango asignado de tamaño aproximado $N/P$ (sin pragmas de OpenMP).
  3. **Gatherv manual no bloqueante:** Los procesos envían sus rangos locales a través de `MPI_Isend` y el root los recibe directamente en las posiciones correspondientes de `R` con `MPI_Irecv`, sincronizando mediante `MPI_Waitall`/`MPI_Wait`.
  4. **Reduce manual no bloqueante:** Los procesos envían sus contadores de FLOPs al root usando `MPI_Isend`/`MPI_Irecv` no bloqueantes.

- **Quicksort PSRS MPI (Comunicaciones No Bloqueantes):**
  1. **Scatterv manual no bloqueante:** El root distribuye los datos iniciales utilizando `MPI_Isend` individuales a cada proceso, que los reciben con `MPI_Irecv`.
  2. **Ordenamiento local secuencial:** Cada proceso ordena su porción usando `quicksort_seq` secuencial puro (se remueve todo uso de OpenMP).
  3. **Muestreo regular y Gather no bloqueante:** Los procesos seleccionan sus muestras locales y las envían al root con `MPI_Isend`/`MPI_Irecv`.
  4. **Selección y Broadcast no bloqueante de pivotes:** El root ordena las muestras, elige los pivotes y los transmite a todos los procesos usando `MPI_Isend`/`MPI_Irecv`.
  5. **Alltoall manual no bloqueante (intercambio de tamaños):** Cada proceso envía a todos los demás los tamaños de sus particiones usando `MPI_Isend`/`MPI_Irecv`.
  6. **Alltoallv manual no bloqueante (redistribución de datos):** Cada proceso redistribuye los bloques de datos a sus respectivos destinos usando `MPI_Isend` e `MPI_Irecv` asíncronos y espera con `MPI_Waitall`.
  7. **Mezcla multi-way local:** Fusión de las particiones recibidas usando un min-heap.
  8. **Gatherv manual no bloqueante final:** El root recopila las porciones ordenadas finales de cada proceso en el buffer de salida mediante `MPI_Isend`/`MPI_Irecv`.
  9. **Reduce manual no bloqueante:** Suma de FLOPs de comparaciones al root mediante envíos no bloqueantes.

### Fase 3: Paralelismo Híbrido (MPI No Bloqueante + OpenMP)
Activar el multi-threading local (`OMP_NUM_THREADS > 1`) sobre la base distribuida no bloqueante para estudiar la ganancia y el overhead de sincronización en memoria compartida.
- **Rank Sort Híbrido:** Paralelizar el bucle externo con `#pragma omp parallel for` y el interno con `#pragma omp simd` para forzar la vectorización.
- **Quicksort PSRS Híbrido:** Paralelizar el ordenamiento local mediante tareas OpenMP (`#pragma omp task`) y la mezcla multi-way.

---

## Proposed Changes

### [Componente Cómputo Secuencial]

#### [NEW] [seq_ranksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/seq_ranksort.cpp)
Implementación secuencial de Rank Sort para números complejos.

#### [NEW] [seq_quicksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/seq_quicksort.cpp)
Implementación secuencial de Quicksort (Median of Three) para números complejos.

### [Componente de Automatización y Medición]

#### [MODIFY] [Makefile](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/Makefile)
Añadir reglas para `seq_ranksort` y `seq_quicksort`.

#### [NEW] [benchmark_scalability.py](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/scripts/benchmark_scalability.py)
Script en Python para orquestar las corridas en sus tres fases y generar curvas de Speedup y Eficiencia.

### [Componente de Cómputo MPI (Fase 2 - Pure MPI)]

#### [MODIFY] [mpi_ranksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/mpi_ranksort.cpp)
Refactorizar para eliminar OpenMP (remover directivas `#pragma omp` e incluye de `omp.h`) y reescribir las comunicaciones colectivas (`MPI_Bcast`, `MPI_Gatherv`, `MPI_Reduce`) a sus equivalentes manuales no bloqueantes empleando `MPI_Isend`, `MPI_Irecv` y `MPI_Waitall`/`MPI_Wait`.

#### [MODIFY] [mpi_quicksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/mpi_quicksort.cpp)
Refactorizar para eliminar OpenMP (reemplazar `quicksort_tasks` por `quicksort_seq` y remover `#pragma omp`) y reescribir todas las comunicaciones colectivas (`MPI_Scatterv`, `MPI_Gather`, `MPI_Bcast`, `MPI_Alltoall`, `MPI_Alltoallv`, `MPI_Reduce`) con primitivas punto a punto no bloqueantes (`MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`).

### [Optimización de Entrada/Salida y Rendimiento]

#### [MODIFY] [seq_ranksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/seq_ranksort.cpp)
#### [MODIFY] [seq_quicksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/seq_quicksort.cpp)
#### [MODIFY] [mpi_ranksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/mpi_ranksort.cpp)
#### [MODIFY] [mpi_quicksort.cpp](file:///home/seo-onion/Workspace/Utec/Paralelas/Proyecto/src/mpi_quicksort.cpp)

Modificaciones propuestas para mitigar el ruido en mediciones de rendimiento:
1. Desactivar la sincronización entre streams de C y C++ en `main()` usando `std::ios_base::sync_with_stdio(false); std::cin.tie(NULL);`.
2. Reemplazar todos los usos de `std::endl` por el caracter `'\n'` para evitar vaciados de buffer innecesarios y costosos.

---

## Plan de Verificación y Métricas

### Métricas de Evaluación
1. **Speedup ($S_p$):** $T_{sec} / T_{par}$
2. **Eficiencia ($E_p$):** $S_p / (P \times T)$
3. **Rendimiento Computacional:** GFLOP/s calculados a partir de las comparaciones e instrucciones aritméticas efectivas.
4. **Overhead de Red:** Porcentaje de tiempo consumido en funciones de comunicación MPI frente al tiempo de cómputo local.
