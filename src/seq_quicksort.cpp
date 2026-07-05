#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <chrono>

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

long long compare_count = 0;

inline bool compare_complex(const Complex& a, const Complex& b) {
    compare_count++;
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
    
    if (!filename.empty()) {
        if (read_binary_file(filename, A, N)) {
            std::cout << "Sequential Quicksort leyendo binario: " << filename << " con N = " << N << '\n';
        } else {
            std::cerr << "Error al leer el archivo binario: " << filename << ". Generando fallback..." << '\n';
            std::srand(1337); // Semilla fija para consistencia en secuencial
            for (int i = 0; i < N; ++i) {
                double val = (double)std::rand() / RAND_MAX * 1000.0;
                A[i].r = val;
                A[i].i = val * 0.3;
            }
        }
    } else {
        std::srand(1337);
        for (int i = 0; i < N; ++i) {
            double val = (double)std::rand() / RAND_MAX * 1000.0;
            A[i].r = val;
            A[i].i = val * 0.3;
        }
        std::cout << "Sequential Quicksort original inicializado (aleatorio) con N = " << N << '\n';
    }

    compare_count = 0;

    // --- Fase de Cómputo ---
    auto t_comp_start = std::chrono::high_resolution_clock::now();
    
    quicksort_seq(A.data(), 0, N - 1);

    auto t_comp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> comp_diff = t_comp_end - t_comp_start;
    double t_comp = comp_diff.count();

    long long total_flops = compare_count * 6;

    // Validar si el arreglo está correctamente ordenado
    bool sorted = true;
    for (int i = 0; i < N - 1; ++i) {
        if (A[i + 1] < A[i]) {
            sorted = false;
            break;
        }
    }

    double t_comm_total = 0.0;

    std::cout << "\n--- Resultados de la Ejecución (Quicksort - Secuencial) ---\n";
    std::cout << "Ordenamiento correcto: " << (sorted ? "SÍ" : "NO") << '\n';
    std::cout << "Tiempo de Comunicación Total: " << t_comm_total << " s" << '\n';
    std::cout << "Tiempo de Cómputo Local (Sort, Proceso 0): " << t_comp << " s" << '\n';
    std::cout << "Cómputo Total Realizado: " << total_flops << " FLOPs (" 
              << (double)total_flops / (t_comp * 1e9) << " GFLOP/s)" << '\n';
    std::cout << "Tiempo Total del Proceso Root: " << t_comp << " s" << '\n';

    return 0;
}
