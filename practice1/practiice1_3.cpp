// task3.cpp
// Динамическая память, указатели и параллельное вычисление среднего
// Программа:
// 1) Создаёт динамический массив с помощью указателя (new)
// 2) Заполняет массив случайными числами
// 3) Вычисляет среднее значение элементов массива (последовательно)
// 4) Вычисляет среднее значение элементов массива параллельно с OpenMP (reduction)
// 5) Выводит количество ядер и реально использованных потоков
// 6) Сравнивает время выполнения
// 7) Освобождает динамическую память

#include <iostream>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

double average_sequential(const int* arr, size_t N) {
    long long sum = 0; // чтобы избежать переполнения
    for (size_t i = 0; i < N; ++i) sum += arr[i];
    return static_cast<double>(sum) / N;
}

double average_parallel(const int* arr, size_t N) {
    long long sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>(N); ++i) sum += arr[i];
#else
    for (size_t i = 0; i < N; ++i) sum += arr[i];
#endif
    return static_cast<double>(sum) / N;
}

int main() {
    using namespace std;

    cout << "Enter N (array size): ";
    size_t N;
    if (!(cin >> N) || N == 0) {
        cerr << "Invalid array size\n";
        return 1;
    }

    // Динамический массив
    int* arr = new int[N];

    // Заполнение случайными числами
    constexpr int RAND_MIN_VAL = 1;
    constexpr int RAND_MAX_VAL = 100;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);
    for (size_t i = 0; i < N; ++i) arr[i] = dist(gen);

    // Последовательное вычисление
    auto t1 = chrono::high_resolution_clock::now();
    double avg_seq = average_sequential(arr, N);
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_seq = t2 - t1;

    // Параллельное вычисление
    auto t3 = chrono::high_resolution_clock::now();
    double avg_par = average_parallel(arr, N);
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_par = t4 - t3;

    cout << "Sequential average = " << avg_seq
        << ", time = " << dur_seq.count() << " ms\n";
    cout << "Parallel average   = " << avg_par
        << ", time = " << dur_par.count() << " ms\n";

    if (abs(avg_seq - avg_par) > 1e-9) cerr << "Warning: results differ!\n";

#ifdef _OPENMP
    // Количество логических ядер на машине
    int num_cores = omp_get_num_procs();
    cout << "Number of logical cores on this machine: " << num_cores << '\n';

    // Количество потоков реально использованных в параллельном блоке
#pragma omp parallel
    {
#pragma omp single
        {
            cout << "Number of threads used in parallel average: "
                << omp_get_num_threads() << '\n';
        }
    }
#else
    cout << "OpenMP not available (compiled without -fopenmp)\n";
#endif

    delete[] arr; // освобождение памяти
    return 0;
}
