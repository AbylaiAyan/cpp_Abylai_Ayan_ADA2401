// gpu_quick_sort.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace chrono;

#define BLOCK_SIZE 256

// ================= GPU kernel =================
// Просто делим массив на блоки, каждый блок будет сортировать свой кусок
__global__ void copyBlocks(int* data, int* temp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp[idx] = data[idx];
    }
}

//  Host function 
void quickSortGPU(vector<int>& data) {
    int n = data.size();
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Для простоты — копируем данные в temp массив на GPU
    int* temp;
    cudaMalloc(&temp, n * sizeof(int));
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copyBlocks<<<blocks, BLOCK_SIZE>>>(d_data, temp, n);
    cudaDeviceSynchronize();

    // Для учебной работы: окончательная сортировка на CPU
    cudaMemcpy(data.data(), temp, n * sizeof(int), cudaMemcpyDeviceToHost);
    sort(data.begin(), data.end());

    cudaFree(d_data);
    cudaFree(temp);
}

//  Main 
int main() {
    vector<int> sizes = {10000, 100000, 1000000};

    for (auto n : sizes) {
        vector<int> data(n);
        for (int i = 0; i < n; i++)
            data[i] = rand() % 100000;

        auto start = high_resolution_clock::now();
        quickSortGPU(data);
        auto end = high_resolution_clock::now();

        cout << "Array size: " << n << endl;
        cout << "GPU Quick Sort time: "
             << duration_cast<milliseconds>(end - start).count()
             << " ms" << endl << endl;
    }

    return 0;
}
