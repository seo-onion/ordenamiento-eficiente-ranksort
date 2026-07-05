#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <queue>
#include <cmath>
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

// Estructura para el nodo del Min-Heap (fusión multi-way)
struct HeapNode {
    Complex val;
    int list_idx;
    int element_idx;
    
    bool operator>(const HeapNode& o) const {
        return o.val < val; // Mayor que para simular min-heap en std::priority_queue
    }
};

// Contador de comparaciones para medir FLOPs
long long local_compare_count = 0;

inline bool compare_complex(const Complex& a, const Complex& b) {
    local_compare_count++;
    return a < b;
}

inline void swap(Complex& a, Complex& b) {
    Complex t = a;
    a = b;
    b = t;
}

int partition(Complex* A, int low, int high) {
    Complex pivot = A[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (compare_complex(A[j], pivot)) {
            i++;
            swap(A[i], A[j]);
        }
    }
    swap(A[i + 1], A[high]);
    return (i + 1);
}

int median_of_three(Complex* A, int low, int high) {
    int mid = low + (high - low) / 2;
    if (compare_complex(A[mid], A[low]))
        swap(A[mid], A[low]);
    if (compare_complex(A[high], A[low]))
        swap(A[high], A[low]);
    if (compare_complex(A[high], A[mid]))
        swap(A[high], A[mid]);
    swap(A[mid], A[high]);
    return partition(A, low, high);
}

void quicksort_seq(Complex* A, int low, int high) {
    if (low < high) {
        int p = median_of_three(A, low, high);
        quicksort_seq(A, low, p - 1);
        quicksort_seq(A, p + 1, high);
    }
}

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

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

    std::vector<Complex> A;
    if (rank == 0) {
        A.resize(N);
        if (!filename.empty()) {
            if (read_binary_file(filename, A, N)) {
                std::cout << "MPI Quicksort PSRS leyendo binario: " << filename << " con N = " << N << '\n';
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
            std::cout << "MPI Quicksort PSRS inicializado (aleatorio) con N = " << N << '\n';
        }
        std::cout << "Ejecutando con " << size << " procesos MPI (OpenMP desactivado/hilo único)..." << '\n';
    }

    // --- Distribución de datos inicial (Fase 1 Scatterv manual) ---
    int base_chunk = N / size;
    int remainder = N % size;
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = base_chunk + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : (displs[r - 1] + sendcounts[r - 1]);
    }
    int local_n = sendcounts[rank];
    std::vector<Complex> local_A(local_n);

    std::vector<int> sendcounts_dbl(size);
    std::vector<int> displs_dbl(size);
    for (int r = 0; r < size; ++r) {
        sendcounts_dbl[r] = sendcounts[r] * 2;
        displs_dbl[r] = displs[r] * 2;
    }
    
    double t_comm_scatter_start = MPI_Wtime();
    std::vector<MPI_Request> scatter_requests;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            if (p == 0) {
                std::copy(A.begin() + displs[0], A.begin() + displs[0] + sendcounts[0], local_A.begin());
            } else {
                MPI_Request req;
                MPI_Isend(A.data() + displs[p], sendcounts_dbl[p], MPI_DOUBLE, p, 10, MPI_COMM_WORLD, &req);
                scatter_requests.push_back(req);
            }
        }
    } else {
        MPI_Request req;
        MPI_Irecv(local_A.data(), local_n * 2, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &req);
        scatter_requests.push_back(req);
    }
    if (!scatter_requests.empty()) {
        MPI_Waitall(scatter_requests.size(), scatter_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_scatter = MPI_Wtime() - t_comm_scatter_start;

    // Caso especial para 1 solo proceso
    if (size == 1) {
        double t_comp_start = MPI_Wtime();
        local_compare_count = 0;
        quicksort_seq(local_A.data(), 0, local_n - 1);
        long long local_flops = local_compare_count * 6;
        double t_local_sort = MPI_Wtime() - t_comp_start;

        bool sorted = true;
        for (int i = 0; i < N - 1; ++i) {
            if (local_A[i + 1] < local_A[i]) {
                sorted = false;
                break;
            }
        }
        
        std::cout << "\n--- Resultados de la Ejecución (Quicksort PSRS - Secuencial/Single Process) ---" << '\n';
        std::cout << "Ordenamiento correcto: " << (sorted ? "SÍ" : "NO") << '\n';
        std::cout << "Tiempo de Comunicación Total: 0 s" << '\n';
        std::cout << "Tiempo de Cómputo Local (Sort, Proceso 0): " << t_local_sort << " s" << '\n';
        std::cout << "Cómputo Total Realizado: " << local_flops << " FLOPs (" 
                  << (double)local_flops / (t_local_sort * 1e9) << " GFLOP/s)" << '\n';
        std::cout << "Tiempo Total del Proceso Root: " << t_local_sort << " s" << '\n';
        
        MPI_Finalize();
        return 0;
    }

    // --- Fase 1: Ordenamiento Local Secuencial ---
    double t_comp_sort_start = MPI_Wtime();
    local_compare_count = 0;
    quicksort_seq(local_A.data(), 0, local_n - 1);
    long long local_flops = local_compare_count * 6;
    double t_local_sort = MPI_Wtime() - t_comp_sort_start;

    // --- Fase 1: Muestreo Regular ---
    std::vector<Complex> local_samples(size - 1);
    int step = local_n / size;
    for (int j = 1; j < size; ++j) {
        int idx = std::min(j * step, local_n - 1);
        local_samples[j - 1] = local_A[idx];
    }

    // --- Fase 2: Recolección de muestras (Gather manual no bloqueante) ---
    std::vector<Complex> all_samples;
    if (rank == 0) {
        all_samples.resize(size * (size - 1));
    }
    
    double t_comm_gather_samples_start = MPI_Wtime();
    std::vector<MPI_Request> gather_samples_requests;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            if (p == 0) {
                std::copy(local_samples.begin(), local_samples.end(), all_samples.begin());
            } else {
                MPI_Request req;
                MPI_Irecv(all_samples.data() + p * (size - 1), (size - 1) * 2, MPI_DOUBLE, p, 11, MPI_COMM_WORLD, &req);
                gather_samples_requests.push_back(req);
            }
        }
    } else {
        MPI_Request req;
        MPI_Isend(local_samples.data(), (size - 1) * 2, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, &req);
        gather_samples_requests.push_back(req);
    }
    if (!gather_samples_requests.empty()) {
        MPI_Waitall(gather_samples_requests.size(), gather_samples_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_gather_samples = MPI_Wtime() - t_comm_gather_samples_start;

    std::vector<Complex> pivots(size - 1);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 0; i < size - 1; ++i) {
            int idx = (i + 1) * (size - 1) - 1; // Muestreo regular estándar
            idx = std::max(0, std::min(idx, (int)all_samples.size() - 1));
            pivots[i] = all_samples[idx];
        }
    }

    // --- Fase 2: Transmisión de pivotes (Broadcast manual no bloqueante) ---
    double t_comm_bcast_pivots_start = MPI_Wtime();
    std::vector<MPI_Request> bcast_pivots_requests;
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Request req;
            MPI_Isend(pivots.data(), (size - 1) * 2, MPI_DOUBLE, p, 12, MPI_COMM_WORLD, &req);
            bcast_pivots_requests.push_back(req);
        }
    } else {
        MPI_Request req;
        MPI_Irecv(pivots.data(), (size - 1) * 2, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD, &req);
        bcast_pivots_requests.push_back(req);
    }
    if (!bcast_pivots_requests.empty()) {
        MPI_Waitall(bcast_pivots_requests.size(), bcast_pivots_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_bcast_pivots = MPI_Wtime() - t_comm_bcast_pivots_start;

    // --- Fase 3: Particionado local ---
    std::vector<int> partition_borders(size - 1);
    for (int i = 0; i < size - 1; ++i) {
        auto it = std::lower_bound(local_A.begin(), local_A.end(), pivots[i]);
        partition_borders[i] = std::distance(local_A.begin(), it);
    }

    std::vector<int> send_counts_p(size, 0);
    send_counts_p[0] = partition_borders[0];
    for (int i = 1; i < size - 1; ++i) {
        send_counts_p[i] = partition_borders[i] - partition_borders[i - 1];
    }
    send_counts_p[size - 1] = local_n - partition_borders[size - 2];

    // Intercambiar tamaños de partición (Alltoall manual no bloqueante)
    std::vector<int> recv_counts_p(size, 0);
    std::vector<MPI_Request> alltoall_size_requests;
    for (int p = 0; p < size; ++p) {
        if (p == rank) {
            recv_counts_p[p] = send_counts_p[p];
        } else {
            MPI_Request req_send, req_recv;
            MPI_Irecv(&recv_counts_p[p], 1, MPI_INT, p, 13, MPI_COMM_WORLD, &req_recv);
            MPI_Isend(&send_counts_p[p], 1, MPI_INT, p, 13, MPI_COMM_WORLD, &req_send);
            alltoall_size_requests.push_back(req_recv);
            alltoall_size_requests.push_back(req_send);
        }
    }
    if (!alltoall_size_requests.empty()) {
        MPI_Waitall(alltoall_size_requests.size(), alltoall_size_requests.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<int> send_displs_p(size, 0);
    std::vector<int> recv_displs_p(size, 0);
    int total_send = 0;
    int total_recv = 0;
    for (int i = 0; i < size; ++i) {
        send_displs_p[i] = total_send;
        total_send += send_counts_p[i];
        
        recv_displs_p[i] = total_recv;
        total_recv += recv_counts_p[i];
    }

    std::vector<Complex> recv_buf(total_recv);

    // Escalar counts y displs para doubles (Complex = 2 doubles)
    std::vector<int> send_counts_p_dbl(size);
    std::vector<int> send_displs_p_dbl(size);
    std::vector<int> recv_counts_p_dbl(size);
    std::vector<int> recv_displs_p_dbl(size);
    for (int i = 0; i < size; ++i) {
        send_counts_p_dbl[i] = send_counts_p[i] * 2;
        send_displs_p_dbl[i] = send_displs_p[i] * 2;
        recv_counts_p_dbl[i] = recv_counts_p[i] * 2;
        recv_displs_p_dbl[i] = recv_displs_p[i] * 2;
    }

    // --- Fase 3: Comunicación Global (Alltoallv manual no bloqueante) ---
    double t_comm_alltoall_start = MPI_Wtime();
    std::vector<MPI_Request> alltoallv_requests;
    for (int p = 0; p < size; ++p) {
        if (p == rank) {
            std::copy(local_A.begin() + send_displs_p[p], local_A.begin() + send_displs_p[p] + send_counts_p[p], recv_buf.begin() + recv_displs_p[p]);
        } else {
            if (recv_counts_p_dbl[p] > 0) {
                MPI_Request req_recv;
                MPI_Irecv(reinterpret_cast<double*>(recv_buf.data()) + recv_displs_p_dbl[p], recv_counts_p_dbl[p], MPI_DOUBLE, p, 14, MPI_COMM_WORLD, &req_recv);
                alltoallv_requests.push_back(req_recv);
            }
            if (send_counts_p_dbl[p] > 0) {
                MPI_Request req_send;
                MPI_Isend(reinterpret_cast<const double*>(local_A.data()) + send_displs_p_dbl[p], send_counts_p_dbl[p], MPI_DOUBLE, p, 14, MPI_COMM_WORLD, &req_send);
                alltoallv_requests.push_back(req_send);
            }
        }
    }
    if (!alltoallv_requests.empty()) {
        MPI_Waitall(alltoallv_requests.size(), alltoallv_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_alltoall = MPI_Wtime() - t_comm_alltoall_start;

    // --- Fase 4: Mezcla Final Multi-way con Min-Heap ---
    std::vector<Complex> sorted_local;
    sorted_local.reserve(total_recv);
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> pq;
    
    for (int i = 0; i < size; ++i) {
        if (recv_counts_p[i] > 0) {
            int start_idx = recv_displs_p[i];
            pq.push({recv_buf[start_idx], i, 0});
        }
    }
    
    double t_merge_start = MPI_Wtime();
    long long merge_comparisons = 0;
    while (!pq.empty()) {
        HeapNode node = pq.top();
        pq.pop();
        sorted_local.push_back(node.val);
        
        int next_elem_idx = node.element_idx + 1;
        if (next_elem_idx < recv_counts_p[node.list_idx]) {
            int start_idx = recv_displs_p[node.list_idx];
            pq.push({recv_buf[start_idx + next_elem_idx], node.list_idx, next_elem_idx});
        }
        merge_comparisons += (pq.size() > 0 ? (int)(std::log2(pq.size())) + 1 : 1);
    }
    double t_merge = MPI_Wtime() - t_merge_start;
    local_flops += merge_comparisons * 6; // Cada comparación de complejos = 6 FLOPs

    // --- Fase 4: Recolección final de tamaños (Gather manual no bloqueante) ---
    std::vector<int> final_sizes(size);
    std::vector<MPI_Request> final_sizes_requests;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            if (p == 0) {
                final_sizes[0] = total_recv;
            } else {
                MPI_Request req;
                MPI_Irecv(&final_sizes[p], 1, MPI_INT, p, 15, MPI_COMM_WORLD, &req);
                final_sizes_requests.push_back(req);
            }
        }
    } else {
        MPI_Request req;
        MPI_Isend(&total_recv, 1, MPI_INT, 0, 15, MPI_COMM_WORLD, &req);
        final_sizes_requests.push_back(req);
    }
    if (!final_sizes_requests.empty()) {
        MPI_Waitall(final_sizes_requests.size(), final_sizes_requests.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<int> final_displs(size, 0);
    std::vector<Complex> B;
    if (rank == 0) {
        int total_size = 0;
        for (int i = 0; i < size; ++i) {
            final_displs[i] = total_size;
            total_size += final_sizes[i];
        }
        B.resize(total_size);
    }

    std::vector<int> final_sizes_dbl(size);
    std::vector<int> final_displs_dbl(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            final_sizes_dbl[i] = final_sizes[i] * 2;
            final_displs_dbl[i] = final_displs[i] * 2;
        }
    }

    // --- Fase 4: Recolección final de datos (Gatherv manual no bloqueante) ---
    double t_comm_gather_start = MPI_Wtime();
    std::vector<MPI_Request> final_gather_requests;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            if (p == 0) {
                std::copy(sorted_local.begin(), sorted_local.end(), B.begin());
            } else {
                if (final_sizes_dbl[p] > 0) {
                    MPI_Request req;
                    MPI_Irecv(reinterpret_cast<double*>(B.data()) + final_displs_dbl[p], final_sizes_dbl[p], MPI_DOUBLE, p, 16, MPI_COMM_WORLD, &req);
                    final_gather_requests.push_back(req);
                }
            }
        }
    } else {
        if (total_recv > 0) {
            MPI_Request req;
            MPI_Isend(reinterpret_cast<const double*>(sorted_local.data()), total_recv * 2, MPI_DOUBLE, 0, 16, MPI_COMM_WORLD, &req);
            final_gather_requests.push_back(req);
        }
    }
    if (!final_gather_requests.empty()) {
        MPI_Waitall(final_gather_requests.size(), final_gather_requests.data(), MPI_STATUSES_IGNORE);
    }
    double t_comm_gather = MPI_Wtime() - t_comm_gather_start;

    double t_comm_total = t_comm_scatter + t_comm_gather_samples + t_comm_bcast_pivots + t_comm_alltoall + t_comm_gather;

    // --- Recolección total de FLOPs (Reduce manual no bloqueante) ---
    long long total_flops = 0;
    std::vector<MPI_Request> reduce_requests;
    std::vector<long long> temp_flops(size, 0);
    if (rank == 0) {
        temp_flops[0] = local_flops;
        for (int p = 1; p < size; ++p) {
            MPI_Request req;
            MPI_Irecv(&temp_flops[p], 1, MPI_LONG_LONG, p, 17, MPI_COMM_WORLD, &req);
            reduce_requests.push_back(req);
        }
    } else {
        MPI_Request req;
        MPI_Isend(&local_flops, 1, MPI_LONG_LONG, 0, 17, MPI_COMM_WORLD, &req);
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

    if (rank == 0) {
        // Validar si el arreglo está correctamente ordenado
        bool sorted = true;
        for (int i = 0; i < N - 1; ++i) {
            if (B[i + 1] < B[i]) {
                sorted = false;
                break;
            }
        }

        std::cout << "\n--- Resultados de la Ejecución (Quicksort PSRS) ---" << '\n';
        std::cout << "Ordenamiento correcto: " << (sorted ? "SÍ" : "NO") << '\n';
        std::cout << "Tiempo de Comunicación Total: " << t_comm_total << " s" << '\n';
        std::cout << "  - Scatter inicial: " << t_comm_scatter << " s" << '\n';
        std::cout << "  - Gather de muestras: " << t_comm_gather_samples << " s" << '\n';
        std::cout << "  - Bcast de pivotes: " << t_comm_bcast_pivots << " s" << '\n';
        std::cout << "  - Alltoallv (redistribución): " << t_comm_alltoall << " s" << '\n';
        std::cout << "  - Gather final: " << t_comm_gather << " s" << '\n';
        std::cout << "Tiempo de Cómputo Local (Sort + Merge, Proceso 0): " << (t_local_sort + t_merge) << " s" << '\n';
        std::cout << "  - Quicksort local: " << t_local_sort << " s" << '\n';
        std::cout << "  - Mezcla multi-way: " << t_merge << " s" << '\n';
        std::cout << "Cómputo Total Realizado (Comparaciones * 6): " << total_flops << " FLOPs (" 
                  << (double)total_flops / ((t_local_sort + t_merge) * 1e9) << " GFLOP/s)" << '\n';
        std::cout << "Tiempo Total del Proceso Root: " << (t_comm_total + t_local_sort + t_merge) << " s" << '\n';
    }

    MPI_Finalize();
    return 0;
}
