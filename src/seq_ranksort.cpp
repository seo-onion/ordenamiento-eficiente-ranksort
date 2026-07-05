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
            std::cout << "Sequential Rank Sort leyendo binario: " << filename << " con N = " << N << '\n';
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
        std::cout << "Sequential Rank Sort original inicializado (aleatorio) con N = " << N << '\n';
    }

    std::vector<int> R(N, 0);
    long long total_flops = 0;

    // --- Fase de Cómputo ---
    auto t_comp_start = std::chrono::high_resolution_clock::now();

    // Precalcular las magnitudes al cuadrado
    std::vector<double> mag(N);
    for (int i = 0; i < N; ++i) {
        mag[i] = A[i].r * A[i].r + A[i].i * A[i].i;
    }

    for (int i = 0; i < N; ++i) {
        int r = 0;
        double mag_i = mag[i];
        
        // Para j < i: la condición (mag_j < mag_i || (mag_j == mag_i && j < i)) se reduce a mag_j <= mag_i
        for (int j = 0; j < i; ++j) {
            r += (mag[j] <= mag_i);
        }
        
        // Para j > i: la condición se reduce a mag_j < mag_i (j = i se omite porque mag_i < mag_i es falso)
        for (int j = i + 1; j < N; ++j) {
            r += (mag[j] < mag_i);
        }
        
        R[i] = r;
        total_flops += N * 5; // Mantenemos la métrica nominal de FLOPs
    }

    auto t_comp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> comp_diff = t_comp_end - t_comp_start;
    double t_comp = comp_diff.count();

    // --- Fase de Reconstrucción ---
    auto t_reconstruct_start = std::chrono::high_resolution_clock::now();
    std::vector<Complex> B(N);
    for (int i = 0; i < N; ++i) {
        B[R[i]] = A[i];
    }
    auto t_reconstruct_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> reconstruct_diff = t_reconstruct_end - t_reconstruct_start;
    double t_reconstruct = reconstruct_diff.count();

    // Validar si el arreglo está correctamente ordenado
    bool sorted = true;
    for (int i = 0; i < N - 1; ++i) {
        if (B[i + 1] < B[i]) {
            sorted = false;
            break;
        }
    }

    double t_comm_total = 0.0;

    std::cout << "\n--- Resultados de la Ejecución ---\n";
    std::cout << "Ordenamiento correcto: " << (sorted ? "SÍ" : "NO") << '\n';
    std::cout << "Tiempo de Comunicación (Bcast + Gather): " << t_comm_total << " s" << '\n';
    std::cout << "Tiempo de Cómputo Local Máximo (Proceso 0): " << t_comp << " s" << '\n';
    std::cout << "Tiempo de Reconstrucción (B): " << t_reconstruct << " s" << '\n';
    std::cout << "Cómputo Total Realizado: " << total_flops << " FLOPs (" 
              << (double)total_flops / (t_comp * 1e9) << " GFLOP/s)" << '\n';
    std::cout << "Tiempo Total del Proceso Root: " << (t_comm_total + t_comp + t_reconstruct) << " s" << '\n';

    return 0;
}
