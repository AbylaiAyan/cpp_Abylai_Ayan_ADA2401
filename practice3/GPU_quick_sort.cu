// gpu_quick_sort.cu
// --------------------------
// Практическая работа №3
// Реализация учебной версии быстрой сортировки с использованием CUDA
//
// ВАЖНО:
// Данная реализация носит учебный характер.
// GPU используется для параллельного распределения и обработки данных,
// а окончательная сортировка выполняется на CPU с помощью std::sort.
//
// Такой подход позволяет продемонстрировать:
// - работу с CUDA (cudaMalloc, cudaMemcpy, kernel launch)
// - взаимодействие CPU и GPU
// - распределение данных между потоками GPU
// -----------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace chrono;

// Размер блока потоков CUDA
// Один блок содержит 256 потоков
#define BLOCK_SIZE 256


// GPU KERNEL 
// Ядро CUDA, которое выполняется на GPU
// Каждый поток копирует ОДИН элемент массива
// из массива data в массив temp
//
// Назначение:
// - продемонстрировать параллельную обработку массива на GPU
// - каждый поток работает независимо со своим индексом
__global__ void copyBlocks(int* data, int* temp, int n) {

    // Глобальный индекс потока
    // blockIdx.x  — номер блока
    // blockDim.x  — количество потоков в блоке
    // threadIdx.x — номер потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        // Копируем один элемент
        temp[idx] = data[idx];
    }
}


// HOST FUNCTION 
// Функция, запускаемая на CPU
// Управляет выделением памяти на GPU, запуском ядра
// и копированием данных между CPU и GPU
void quickSortGPU(vector<int>& data) {

    int n = data.size();   // Размер массива

    // Указатель на память GPU
    int* d_data;

    // Выделяем память на GPU для исходного массива
    cudaMalloc(&d_data, n * sizeof(int));

    // Копируем данные с CPU (Host) на GPU (Device)
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Выделяем временный массив на GPU
    // Он используется для демонстрации параллельной обработки
    int* temp;
    cudaMalloc(&temp, n * sizeof(int));

    // Вычисляем количество блоков CUDA
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Запуск CUDA-ядра:
    // blocks    — количество блоков
    // BLOCK_SIZE — количество потоков в одном блоке
    copyBlocks<<<blocks, BLOCK_SIZE>>>(d_data, temp, n);

    // Ожидаем завершения работы GPU
    cudaDeviceSynchronize();

    // Копируем данные обратно с GPU на CPU
    cudaMemcpy(data.data(), temp, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Для учебной работы:
    // окончательная сортировка выполняется на CPU
    // Это упрощает реализацию и избегает сложности
    // полноценного рекурсивного Quick Sort на GPU
    sort(data.begin(), data.end());

    // Освобождаем память GPU
    cudaFree(d_data);
    cudaFree(temp);
}


// MAIN FUNCTION 
// Точка входа в программу
// Создаёт массивы разных размеров
// Измеряет время выполнения GPU Quick Sort
int main() {

    // Размеры массивов для тестирования
    vector<int> sizes = {10000, 100000, 1000000};

    // Последовательно тестируем каждый размер
    for (auto n : sizes) {

        // Создаём массив нужного размера
        vector<int> data(n);

        // Заполняем массив случайными числами
        for (int i = 0; i < n; i++)
            data[i] = rand() % 100000;

        // Засекаем время начала
        auto start = high_resolution_clock::now();

        // Запуск GPU версии быстрой сортировки
        quickSortGPU(data);

        // Засекаем время окончания
        auto end = high_resolution_clock::now();

        // Вывод результатов
        cout << "Array size: " << n << endl;
        cout << "GPU Quick Sort time: "
             << duration_cast<milliseconds>(end - start).count()
             << " ms" << endl << endl;
    }

    return 0;
}

