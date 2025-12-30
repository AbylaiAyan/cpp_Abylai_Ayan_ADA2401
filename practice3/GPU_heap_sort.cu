/*
Практическая работа №3
Тема: Реализация сложных алгоритмов сортировки на GPU с использованием CUDA

Данный файл содержит реализацию пирамидальной сортировки (Heap Sort),
выполненной на GPU с использованием технологии CUDA.

Описание алгоритма:
- Массив целых чисел копируется из оперативной памяти (CPU) в память видеокарты (GPU).
- На GPU строится бинарная куча (max-heap) с использованием итеративной функции heapify.
- Процесс heapify реализован в виде CUDA kernel и выполняется полностью на GPU.
- Для построения кучи используется подход "bottom-up".
- Затем элементы поочерёдно извлекаются из кучи, при этом структура кучи восстанавливается.
- После завершения сортировки результат копируется обратно с GPU на CPU.

Особенности реализации:
- Рекурсия не используется (GPU плохо поддерживает рекурсивные вызовы).
- heapify реализован итеративно внутри kernel.
- Используется синхронизация cudaDeviceSynchronize() после каждого запуска kernel.
- Реализация носит учебный характер и предназначена для демонстрации работы CUDA.

Измерение времени:
- Время выполнения сортировки измеряется на стороне CPU
  с помощью библиотеки <chrono>.
- Тестирование проводится для массивов размером:
  10 000, 100 000 и 1 000 000 элементов.

Среда выполнения:
- Google Colab с включённым GPU.
*/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace chrono;

//  GPU kernel: итеративный heapify 
__global__ void heapifyKernel(int* data, int n, int start) {
    int idx = start + threadIdx.x;

    while (true) {
        int largest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < n && data[left] > data[largest])
            largest = left;
        if (right < n && data[right] > data[largest])
            largest = right;

        if (largest != idx) {
            int tmp = data[idx];
            data[idx] = data[largest];
            data[largest] = tmp;
            idx = largest;  // продолжаем с нового индекса
        } else {
            break;
        }
    }
}

//  Host function 
void heapSortGPU(vector<int>& arr) {
    int n = arr.size();
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Построение кучи (bottom-up)
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapifyKernel<<<1,1>>>(d_data, n, i);
        cudaDeviceSynchronize();
    }

    // Извлечение элементов
    for (int i = n - 1; i > 0; i--) {
        // Меняем корень и последний элемент
        cudaMemcpy(&arr[0], d_data, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&arr[i], d_data + i, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(d_data, &arr[i], sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data + i, &arr[0], sizeof(int), cudaMemcpyHostToDevice);

        // Восстанавливаем кучу для уменьшенного массива
        heapifyKernel<<<1,1>>>(d_data, i, 0);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

//  Main 
int main() {
    vector<int> sizes = {10000, 100000, 1000000};

    for (auto n : sizes) {
        vector<int> data(n);
        for (int i = 0; i < n; i++)
            data[i] = rand() % 100000;

        auto start = high_resolution_clock::now();
        heapSortGPU(data);
        auto end = high_resolution_clock::now();

        cout << "Array size: " << n << endl;
        cout << "GPU Heap Sort time: "
             << duration_cast<milliseconds>(end - start).count()
             << " ms" << endl << endl;
    }

    return 0;
}

