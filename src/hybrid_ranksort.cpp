#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <mpi.h>
#include <omp.h>

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

    // Inicializar MPI con soporte para múltiples hilos
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Advertencia: El nivel de soporte de hilos MPI no es suficiente." << '\n';
    }

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
                std::cout << "Hybrid Rank Sort leyendo binario: " << filename << " con N = " << N << '\n';
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
            std::cout << "Hybrid Rank Sort original inicializado (aleatorio) con N = " << N << '\n';
        }
        
        int max_threads = omp_get_max_threads();
        std::cout << "Ejecutando con " << size << " procesos MPI y " << max_threads << " hilos OpenMP por proceso..." << '\n';
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

    // --- Fase de Cómputo Local (Paralelizado con OpenMP) ---
    double t_comp_start = MPI_Wtime();

    // Precalcular las magnitudes al cuadrado para todo el arreglo A
    std::vector<double> mag(N);
    
    #pragma omp parallel for simd
    for (int j = 0; j < N; ++j) {
        mag[j] = A[j].r * A[j].r + A[j].i * A[j].i;
    }

    // Paralelización del bucle principal
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = inicio; i < fin; ++i) {
        double mag_i = mag[i];
        
        int local_r1 = 0;
        // Para j < i: se reduce a mag_j <= mag_i
        #pragma omp simd reduction(+:local_r1)
        for (int j = 0; j < i; ++j) {
            local_r1 += (mag[j] <= mag_i);
        }
        
        int local_r2 = 0;
        // Para j > i: se reduce a mag_j < mag_i
        #pragma omp simd reduction(+:local_r2)
        for (int j = i + 1; j < N; ++j) {
            local_r2 += (mag[j] < mag_i);
        }
        
        local_R[i - inicio] = local_r1 + local_r2;
    }
    
    long long local_flops = (long long)local_n * N * 5; // Mantenemos la métrica nominal de FLOPs
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
        
        #pragma omp parallel for
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
