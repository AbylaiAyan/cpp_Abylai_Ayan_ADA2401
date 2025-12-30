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
