#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256  // размер блока потоков для GPU

// ======================
// Задание 1 — генерация данных
// ======================
void generateArray(int* arr, int n) {
    // Заполняем массив случайными числами от 0 до 99
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 100;
}

// ======================
// Задание 2.1 — редукция с использованием только глобальной памяти
// ======================
__global__ void reduceGlobal(int* input, int* result, int n) {
    // Каждый поток работает с отдельным элементом
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        // Атомарно суммируем в глобальную память
        atomicAdd(result, input[idx]);
}

// ======================
// Задание 2.2 — редукция с использованием глобальной + разделяемой памяти
// ======================
__global__ void reduceShared(int* input, int* result, int n) {
    // Разделяемая память для блока потоков
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x; // индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + tid; // глобальный индекс

    // Загружаем элемент из глобальной памяти в разделяемую память
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads(); // ждем, пока все потоки блока загрузят данные

    // Суммирование в рамках блока (редукция)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads(); // синхронизация потоков после каждого шага
    }

    // Первый поток блока добавляет результат блока в глобальную память
    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

// ======================
// Задание 3 — сортировка пузырьком с использованием локальной памяти
// ======================
__global__ void bubbleSortLocal(int* data, int n, int chunk) {
    int start = blockIdx.x * chunk; // начало подмассива
    if (start >= n) return; // если блок выходит за границы массива, выходим

    int local[64]; // локальная память потока
    int size = min(chunk, n - start); // размер подмассива (для последнего блока может быть меньше)

    // Копируем данные из глобальной памяти в локальную
    for (int i = 0; i < size; i++)
        local[i] = data[start + i];

    // Сортировка пузырьком в локальной памяти
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size - i - 1; j++)
            if (local[j] > local[j + 1]) {
                int t = local[j];
                local[j] = local[j + 1];
                local[j + 1] = t;
            }

    // Копируем отсортированные данные обратно в глобальную память
    for (int i = 0; i < size; i++)
        data[start + i] = local[i];
}

// ======================
// MAIN — основной код для всех заданий
// ======================
int main() {
    srand(0); // фиксируем генератор случайных чисел для повторяемости

    // Массивы разных размеров для экспериментов
    int sizes[3] = { 10000, 100000, 1000000 };

    for (int i = 0; i < 3; i++) {
        int n = sizes[i];
        printf("\n===============================\n");
        printf("Array size: %d\n", n);

        // ---------- Задание 1 ----------
        int* h_data = (int*)malloc(n * sizeof(int)); // выделяем память на CPU
        generateArray(h_data, n); // генерируем случайные числа

        int* d_data;
        cudaMalloc(&d_data, n * sizeof(int)); // выделяем память на GPU
        cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice); // копируем данные на GPU

        // ---------- Задание 2.1 ----------
        int* d_result;
        cudaMalloc(&d_result, sizeof(int)); // память для результата на GPU
        cudaMemset(d_result, 0, sizeof(int)); // обнуляем

        // Определяем сетку блоков
        dim3 block(BLOCK_SIZE);
        dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // События CUDA для измерения времени
        cudaEvent_t s1, e1;
        cudaEventCreate(&s1);
        cudaEventCreate(&e1);

        cudaEventRecord(s1);
        reduceGlobal << <grid, block >> > (d_data, d_result, n); // запуск ядра
        cudaEventRecord(e1);
        cudaEventSynchronize(e1); // ждем окончания

        float t21;
        cudaEventElapsedTime(&t21, s1, e1); // вычисляем время
        printf("Task 2.1 (Reduction - global): %.4f ms\n", t21);

        cudaMemset(d_result, 0, sizeof(int)); // обнуляем результат перед следующим ядром

        // ---------- Задание 2.2 ----------
        cudaEvent_t s2, e2;
        cudaEventCreate(&s2);
        cudaEventCreate(&e2);

        cudaEventRecord(s2);
        reduceShared << <grid, block >> > (d_data, d_result, n); // запуск ядра с shared памятью
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);

        float t22;
        cudaEventElapsedTime(&t22, s2, e2);
        printf("Task 2.2 (Reduction - shared): %.4f ms\n", t22);

        cudaFree(d_result); // освобождаем память для результата

        // ---------- Задание 3 ----------
        int chunk = 64; // размер подмассива для сортировки
        dim3 gridSort((n + chunk - 1) / chunk); // количество блоков

        cudaEvent_t s3, e3;
        cudaEventCreate(&s3);
        cudaEventCreate(&e3);

        cudaEventRecord(s3);
        bubbleSortLocal << <gridSort, 1 >> > (d_data, n, chunk); // сортировка пузырьком
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);

        float t3;
        cudaEventElapsedTime(&t3, s3, e3);
        printf("Task 3 (Bubble sort - local): %.4f ms\n", t3);

        // Освобождаем память
        cudaFree(d_data);
        free(h_data);
    }

    return 0;
}
