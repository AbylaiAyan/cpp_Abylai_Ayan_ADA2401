// lab2_3_selection_sort.cpp
// Параллельная сортировка выбором с OpenMP
//
// Программа:
// 1) Создаёт массивы размером 1000 и 10000 элементов
// 2) Заполняет их случайными числами
// 3) Сортирует массивы:
//    - последовательный алгоритм сортировки выбором
//    - параллельный алгоритм сортировки выбором с OpenMP
// 4) Замеряет и сравнивает время выполнения
//
// Сборка:
// g++ -fopenmp -O2 -std=c++17 lab3_selection_sort.cpp -o lab3.exe
//

#include <iostream>
#include <vector>
#include <cstdlib>    // rand()
#include <omp.h>      // OpenMP

// ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА ВЫБОРОМ
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex])
                minIndex = j;
        }
        std::swap(arr[i], arr[minIndex]);
    }
}

// ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА ВЫБОРОМ
// Простой вариант: параллельный поиск минимума на каждом шаге
void parallelSelectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

        // Параллельный поиск минимума на подмассиве arr[i..n-1]
#pragma omp parallel
        {
            int localMin = minIndex;

#pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[localMin])
                    localMin = j;
            }

            // Критическая секция для обновления глобального минимума
#pragma omp critical
            {
                if (arr[localMin] < arr[minIndex])
                    minIndex = localMin;
            }
        }

        std::swap(arr[i], arr[minIndex]);
    }
}

int main() {
    // Размеры массивов
    std::vector<int> sizes = { 1000, 10000 };

    for (int sz : sizes) {
        std::vector<int> arr(sz);

        // Заполняем массив случайными числами
        for (int i = 0; i < sz; i++)
            arr[i] = rand() % 100000;

        // ------------------ Последовательная сортировка ------------------
        std::vector<int> arr_seq = arr; // копия для последовательного варианта

        double start = omp_get_wtime();
        selectionSort(arr_seq);
        double end = omp_get_wtime();
        std::cout << "Size " << sz << " - Sequential sort time: "
            << (end - start) * 1000 << " ms" << std::endl;

        // ------------------ Параллельная сортировка ------------------
        std::vector<int> arr_par = arr; // копия для параллельного варианта

        start = omp_get_wtime();
        parallelSelectionSort(arr_par);
        end = omp_get_wtime();
        std::cout << "Size " << sz << " - Parallel sort time:   "
            << (end - start) * 1000 << " ms" << std::endl;

        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}
