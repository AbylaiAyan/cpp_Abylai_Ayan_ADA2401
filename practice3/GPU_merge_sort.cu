// gpu_merge_sort.cu
// Практическая работа №3
// Параллельная сортировка слиянием на GPU с использованием CUDA
// 1) Генерация массива случайных чисел
// 2) Параллельная сортировка блоков на GPU
// 3) Пошаговое слияние блоков
// 4) Сравнение с CPU (можно замерять время на GPU)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#define BLOCK_SIZE 256  // Размер блока потоков

using namespace std;
using namespace std::chrono;

// GPU kernel: сортировка отдельного блока 
__global__ void blockSort(int* data, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Копируем элементы блока в shared memory
    if (idx < n)
        shared[tid] = data[idx];
    else
        shared[tid] = INT_MAX;  // Паддинг для пустых элементов

    __syncthreads();

    // Простая сортировка внутри блока (odd-even sort)
    for (int i = 0; i < blockDim.x; i++) {
        int j = tid ^ (i & 1);
        if (j > tid && shared[tid] > shared[j]) {
            int tmp = shared[tid];
            shared[tid] = shared[j];
            shared[j] = tmp;
        }
        __syncthreads();
    }

    // Возвращаем данные обратно в глобальную память
    if (idx < n)
        data[idx] = shared[tid];
}

//  GPU kernel: слияние двух блоков 
__global__ void mergeKernel(int* input, int* output, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int start = idx * 2 * width;
    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start, j = mid, k = start;

    while (i < mid && j < end)
        output[k++] = (input[i] < input[j]) ? input[i++] : input[j++];

    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

//  Host function 
void mergeSortGPU(vector<int>& data) {
    int n = data.size();
    int* d_in, * d_out;

    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));

    cudaMemcpy(d_in, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Сортировка блоков параллельно
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blockSort<<<blocks, BLOCK_SIZE>>>(d_in, n);
    cudaDeviceSynchronize();

    // Пошаговое слияние блоков
    for (int width = BLOCK_SIZE; width < n; width *= 2) {
        int mergeBlocks = (n + 2 * width - 1) / (2 * width);
        mergeKernel<<<mergeBlocks, 1>>>(d_in, d_out, width, n);
        cudaDeviceSynchronize();
        swap(d_in, d_out);
    }

    cudaMemcpy(data.data(), d_in, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

//  Main 
int main() {
    vector<int> sizes = {10000, 100000, 1000000};

    for (auto n : sizes) {
        vector<int> data(n);
        for (int i = 0; i < n; i++)
            data[i] = rand() % 100000;

        auto start = high_resolution_clock::now();
        mergeSortGPU(data);
        auto end = high_resolution_clock::now();

        cout << "Array size: " << n << endl;
        cout << "GPU Merge Sort time: "
             << duration_cast<milliseconds>(end - start).count()
             << " ms" << endl << endl;
    }

    return 0;
}
