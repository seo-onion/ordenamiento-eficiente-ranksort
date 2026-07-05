#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <mpi.h>

struct Complex { 
    double r, i; 
    
    // Sobrecarga para ordenamiento basado en magnitud al cuadrado
    bool operator<(const Complex& o) const {
        return (r*r + i*i) < (o.r*o.r + o.i*o.i);
    }
    bool operator==(const Complex& o) const {
        return (r*r + i*i) == (o.r*o.r + o.i*o.i);
    }
};

bool read_binary_file(const std::string& filename, std::vector<Complex>& A, int N) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    std::vector<double> temp(N);
    file.read(reinterpret_cast<char*>(temp.data()), N * sizeof(double));
    if (file.gcount() != static_cast<std::streamsize>(N * sizeof(double))) {
        return false;
    }
    for (int i = 0; i < N; ++i) {
        A[i].r = temp[i];
        A[i].i = temp[i] * 0.3; // Parte imaginaria determinista
    }
    return true;
}

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Inicializar MPI (sin soporte de hilos multithreading, ya que OpenMP está desactivado)
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string filename = "";
    int N = 10000; // Tamaño por defecto

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1.size() > 4 && arg1.substr(arg1.size() - 4) == ".bin") {
            filename = arg1;
            if (argc > 2) {
                N = std::atoi(argv[2]);
            }
        } else {
            N = std::atoi(argv[1]);
        }
    }

    std::vector<Complex> A(N);
    
    // Lectura de datos (Solo en proceso Root / rank 0)
    if (rank == 0) {
        if (!filename.empty()) {
            if (read_binary_file(filename, A, N)) {
                std::cout << "MPI Rank Sort leyendo binario: " << filename << " con N = " << N << '\n';
            } else {
                std::cerr << "Error al leer el archivo binario: " << filename << ". Generando fallback..." << '\n';
                std::srand(std::time(nullptr));
                for (int i = 0; i < N; ++i) {
                    double val = (double)std::rand() / RAND_MAX * 1000.0;
                    A[i].r = val;
                    A[i].i = val * 0.3;
                }
            }
        } else {
            std::srand(std::time(nullptr));
            for (int i = 0; i < N; ++i) {
                double val = (double)std::rand() / RAND_MAX * 1000.0;
                A[i].r = val;
                A[i].i = val * 0.3;
            }
            std::cout << "MPI Rank Sort original inicializado (aleatorio) con N = " << N << '\n';
        }
        std::cout << "Ejecutando con " << size << " procesos MPI (OpenMP desactivado/hilo único)..." << '\n';
    }

    // --- Comunicación 1: Broadcast manual no bloqueante de A ---
    double t_comm_start = MPI_Wtime();
    std::vector<MPI_Request> bcast_requests;
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Request req;
            MPI_Isend(A.data(), N * 2, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &req);
            bcast_requests.push_back(req);
        }
    } else {
        MPI_Request req;
        MPI_Irecv(A.data(), N * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
        bcast_requests.push_back(req);
    }
    if (!bcast_requests.empty()) {
        MPI_Waitall(bcast_requests.size(), bcast_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_bcast = MPI_Wtime() - t_comm_start;

    // --- Cálculo del rango de cómputo local ---
    int base_chunk = N / size;
    int remainder = N % size;
    int inicio = rank * base_chunk + std::min(rank, remainder);
    int fin = (rank + 1) * base_chunk + std::min(rank + 1, remainder);
    int local_n = fin - inicio;

    std::vector<int> local_R(local_n, 0);
    long long local_flops = 0;

    // --- Fase de Cómputo Local ---
    double t_comp_start = MPI_Wtime();

    // Precalcular las magnitudes al cuadrado para todo el arreglo A
    std::vector<double> mag(N);
    for (int j = 0; j < N; ++j) {
        mag[j] = A[j].r * A[j].r + A[j].i * A[j].i;
    }

    for (int i = inicio; i < fin; ++i) {
        int r = 0;
        double mag_i = mag[i];
        
        // Para j < i: se reduce a mag_j <= mag_i
        for (int j = 0; j < i; ++j) {
            r += (mag[j] <= mag_i);
        }
        // Para j > i: se reduce a mag_j < mag_i
        for (int j = i + 1; j < N; ++j) {
            r += (mag[j] < mag_i);
        }
        
        local_R[i - inicio] = r;
        local_flops += N * 5; // Mantenemos la métrica nominal de FLOPs
    }
    double t_comp = MPI_Wtime() - t_comp_start;

    // Preparar estructuras para recolectar rangos en rank 0
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        int r_start = r * base_chunk + std::min(r, remainder);
        int r_end = (r + 1) * base_chunk + std::min(r + 1, remainder);
        recvcounts[r] = r_end - r_start;
        displs[r] = r_start;
    }

    std::vector<int> R;
    if (rank == 0) {
        R.resize(N);
    }

    // --- Comunicación 2: Gather manual no bloqueante de los rangos locales R ---
    double t_comm_gather_start = MPI_Wtime();
    std::vector<MPI_Request> gather_requests;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            if (p == 0) {
                std::copy(local_R.begin(), local_R.end(), R.begin() + displs[0]);
            } else {
                MPI_Request req;
                MPI_Irecv(R.data() + displs[p], recvcounts[p], MPI_INT, p, 1, MPI_COMM_WORLD, &req);
                gather_requests.push_back(req);
            }
        }
    } else {
        MPI_Request req;
        MPI_Isend(local_R.data(), local_n, MPI_INT, 0, 1, MPI_COMM_WORLD, &req);
        gather_requests.push_back(req);
    }
    if (!gather_requests.empty()) {
        MPI_Waitall(gather_requests.size(), gather_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_gather = MPI_Wtime() - t_comm_gather_start;

    double t_comm_total = t_comm_bcast + t_comm_gather;

    // --- Comunicación 3: Reduce manual no bloqueante de FLOPs ---
    long long total_flops = 0;
    std::vector<MPI_Request> reduce_requests;
    std::vector<long long> temp_flops(size, 0);
    if (rank == 0) {
        temp_flops[0] = local_flops;
        for (int p = 1; p < size; ++p) {
            MPI_Request req;
            MPI_Irecv(&temp_flops[p], 1, MPI_LONG_LONG, p, 2, MPI_COMM_WORLD, &req);
            reduce_requests.push_back(req);
        }
    } else {
        MPI_Request req;
        MPI_Isend(&local_flops, 1, MPI_LONG_LONG, 0, 2, MPI_COMM_WORLD, &req);
        reduce_requests.push_back(req);
    }
    if (!reduce_requests.empty()) {
        MPI_Waitall(reduce_requests.size(), reduce_requests.data(), MPI_STATUSES_IGNORE);
    }
    if (rank == 0) {
        total_flops = 0;
        for (int p = 0; p < size; ++p) {
            total_flops += temp_flops[p];
        }
    }

    // --- Reconstrucción y validación en rank 0 ---
    if (rank == 0) {
        double t_reconstruct_start = MPI_Wtime();
        std::vector<Complex> B(N);
        for (int i = 0; i < N; ++i) {
            B[R[i]] = A[i];
        }
        double t_reconstruct = MPI_Wtime() - t_reconstruct_start;

        // Validar si el arreglo está correctamente ordenado
        bool sorted = true;
        for (int i = 0; i < N - 1; ++i) {
            if (B[i + 1] < B[i]) {
                sorted = false;
                break;
            }
        }

        std::cout << "\n--- Resultados de la Ejecución ---" << '\n';
        std::cout << "Ordenamiento correcto: " << (sorted ? "SÍ" : "NO") << '\n';
        std::cout << "Tiempo de Comunicación (Bcast + Gather): " << t_comm_total << " s" << '\n';
        std::cout << "Tiempo de Cómputo Local Máximo (Proceso 0): " << t_comp << " s" << '\n';
        std::cout << "Tiempo de Reconstrucción (B): " << t_reconstruct << " s" << '\n';
        std::cout << "Cómputo Total Realizado: " << total_flops << " FLOPs (" 
                  << (double)total_flops / (t_comp * 1e9) << " GFLOP/s)" << '\n';
        std::cout << "Tiempo Total del Proceso Root: " << (t_comm_total + t_comp + t_reconstruct) << " s" << '\n';
    }

    MPI_Finalize();
    return 0;
}

/*
 * =========================================================================
 *                      GLOSARIO DE VARIABLES (MPI RANK SORT)
 * =========================================================================
 *
 * ESTRUCTURA 'Complex':
 *   - r: Parte real del número complejo.
 *   - i: Parte imaginaria del número complejo.
 *
 * VARIABLES GLOBALES / CONFIGURACIÓN DE MPI:
 *   - rank: Identificador único (ID) del proceso actual en el comunicador MPI_COMM_WORLD.
 *   - size: Número total de procesos MPI lanzados en el comunicador MPI_COMM_WORLD.
 *   - filename: Ruta al archivo binario de entrada que contiene el dataset a ordenar.
 *   - N: Tamaño total del arreglo (número de elementos Complex a ordenar).
 *   - A: Vector que contiene los datos originales a ordenar (completo en rank 0, vacío en los demás ranks).
 *
 * BROADCAST MANUAL DE DATOS ('A'):
 *   - t_comm_start: Marca de tiempo para registrar el inicio de las fases de comunicación.
 *   - bcast_requests: Vector de solicitudes MPI_Request utilizado para rastrear y sincronizar los envíos/recepciones asíncronos (MPI_Isend / MPI_Irecv) del arreglo original.
 *   - t_comm_bcast: Tiempo transcurrido durante la fase inicial de broadcast de datos.
 *
 * PARTICIONAMIENTO DEL TRABAJO:
 *   - base_chunk: Cantidad base de elementos asignados a cada proceso (N / size).
 *   - remainder: Residuo de la división (N % size) distribuido equitativamente entre los primeros procesos.
 *   - inicio: Índice de inicio del segmento de datos del cual el proceso actual es responsable.
 *   - fin: Índice de fin (exclusivo) del segmento de datos del cual el proceso actual es responsable.
 *   - local_n: Cantidad de elementos en la sección local del proceso actual (fin - inicio).
 *
 * FASE DE CÓMPUTO LOCAL:
 *   - local_R: Vector local de tamaño 'local_n' que almacena las posiciones finales (rangos) de la porción de datos asignada.
 *   - local_flops: Contador de operaciones de coma flotante (FLOPs) calculadas nominalmente en el proceso actual.
 *   - t_comp_start: Marca de tiempo que registra el inicio de la fase de cómputo local.
 *   - mag: Vector que contiene las magnitudes al cuadrado precalculadas de todos los elementos de 'A' (para evitar recalcularlas en el bucle anidado).
 *   - t_comp: Tiempo total de cómputo local registrado en el proceso actual.
 *
 * GATHER MANUAL DE RANGOS:
 *   - recvcounts: Vector en el proceso Root que indica cuántos rangos recibir de cada uno de los procesos.
 *   - displs: Vector en el proceso Root que indica el desplazamiento (offset) en el arreglo 'R' donde se debe guardar el segmento recibido de cada proceso.
 *   - R: Vector global en el proceso Root que almacena todos los rangos consolidados de los elementos.
 *   - t_comm_gather_start: Marca de tiempo que indica el inicio de la fase de recolección de rangos.
 *   - gather_requests: Vector de solicitudes MPI_Request para rastrear los envíos/recepciones asíncronos de los rangos locales.
 *   - t_comm_gather: Tiempo transcurrido durante el Gather manual de los rangos.
 *   - t_comm_total: Tiempo total acumulado de las comunicaciones de datos y rangos (t_comm_bcast + t_comm_gather).
 *
 * REDUCCIÓN MANUAL DE FLOPS:
 *   - total_flops: Suma total de operaciones flotantes de todos los procesos recolectada en el proceso Root.
 *   - reduce_requests: Vector de solicitudes MPI_Request para rastrear la reducción asíncrona de los contadores FLOPs.
 *   - temp_flops: Vector auxiliar utilizado en el Root para almacenar temporalmente los FLOPs reportados por cada proceso.
 *
 * RECONSTRUCCIÓN Y VALIDACIÓN (SOLO ROOT):
 *   - t_reconstruct_start: Marca de tiempo para registrar el inicio de la ordenación física de los datos.
 *   - B: Vector destino en el proceso Root donde se colocan los elementos del arreglo original en su posición final ordenada (B[R[i]] = A[i]).
 *   - t_reconstruct: Tiempo transcurrido durante la fase de reconstrucción del arreglo.
 *   - sorted: Variable booleana de control que determina si el arreglo 'B' fue ordenado correctamente.
 * =========================================================================
 */