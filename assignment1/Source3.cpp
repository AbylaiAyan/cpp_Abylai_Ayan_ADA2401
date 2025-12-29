// task3.cpp
// Задание 3
// Последовательный и параллельный поиск
// минимального и максимального элементов массива
// с использованием OpenMP

#include <iostream>
#include <random>
#include <chrono>
#include <climits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

int main() {

    // Размер массива
    int size = 1000000;

    // Динамически выделяем память под массив
    int* arr = new int[size];

    // Генератор случайных чисел от 1 до 100
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    // Заполняем массив случайными числами
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    /* ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК MIN И MAX */

       // Начинаем замер времени
    auto start_seq = high_resolution_clock::now();

    int min_seq = arr[0];
    int max_seq = arr[0];

    // Обычный цикл без OpenMP
    for (int i = 1; i < size; i++) {
        if (arr[i] < min_seq)
            min_seq = arr[i];
        if (arr[i] > max_seq)
            max_seq = arr[i];
    }

    // Заканчиваем замер времени
    auto end_seq = high_resolution_clock::now();
    auto time_seq = duration_cast<milliseconds>(end_seq - start_seq);

    cout << "Sequential version:" << endl;
    cout << "Min = " << min_seq << ", Max = " << max_seq << endl;
    cout << "Time = " << time_seq.count() << " ms\n\n";


    /* ПАРАЛЛЕЛЬНЫЙ ПОИСК MIN И MAX (OpenMP) */

#ifdef _OPENMP

    int min_par = arr[0];
    int max_par = arr[0];

    // Начинаем замер времени
    auto start_par = high_resolution_clock::now();

#pragma omp parallel
    {
        // Локальные минимум и максимум для каждого потока
        int local_min = INT_MAX;
        int local_max = INT_MIN;

        // Каждый поток обрабатывает свою часть массива
#pragma omp for
        for (int i = 0; i < size; i++) {
            if (arr[i] < local_min)
                local_min = arr[i];
            if (arr[i] > local_max)
                local_max = arr[i];
        }

        // Обновляем общий минимум и максимум
        // critical нужен, чтобы не было конфликтов между потоками
#pragma omp critical
        {
            if (local_min < min_par)
                min_par = local_min;
            if (local_max > max_par)
                max_par = local_max;
        }
    }

    // Заканчиваем замер времени
    auto end_par = high_resolution_clock::now();
    auto time_par = duration_cast<milliseconds>(end_par - start_par);

    cout << "Parallel version (OpenMP):" << endl;
    cout << "Min = " << min_par << ", Max = " << max_par << endl;
    cout << "Time = " << time_par.count() << " ms" << endl;

#else
    cout << "OpenMP is not enabled" << endl;
#endif

    // Освобождаем выделенную память
    delete[] arr;

    return 0;
}


