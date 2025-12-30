// for_cpu_sorts.cpp
// Практическая работа №3
// Тема: Реализация и сравнение алгоритмов сортировки (Merge Sort, Quick Sort, Heap Sort)
//
// Программа:
// 1) Генерирует массивы целых чисел заданного размера (10 000, 100 000, 1 000 000)
// 2) Сортирует массивы последовательными алгоритмами:
//    - Merge Sort (сортировка слиянием)
//    - Quick Sort (быстрая сортировка)
//    - Heap Sort (пирамидальная сортировка)
// 3) Измеряет время выполнения каждого алгоритма на CPU
// 4) Выводит результаты в консоль для последующего анализа и сравнения
//
// Данная программа используется как последовательная (CPU) версия
// для сравнения с параллельными реализациями сортировок на GPU (CUDA)
//
// Компиляция (пример, g++):
// g++ -O2 -std=c++17 cpu_sorts.cpp -o cpu_sorts
//
// Компиляция (Visual Studio):
// Создать консольный проект C++
// Вставить данный файл и собрать проект в режиме Release
//
// Запуск:
// for_cpu_sorts.exe


#include <iostream>      // Ввод / вывод
#include <vector>        // Контейнер vector
#include <algorithm>    // std::swap
#include <chrono>        // Замер времени
#include <cstdlib>       // rand()
#include <ctime>         // time()

using namespace std;
using namespace chrono;

/* -----------------------------------------
   MERGE SORT (последовательная)
   -----------------------------------------
*/

// Функция слияния двух отсортированных частей массива
void merge(vector<int>& arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;     // Размер левой части
    int n2 = right - mid;        // Размер правой части

    // Временные массивы
    vector<int> L(n1), R(n2);

    // Копируем данные во временные массивы
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // Индексы для слияния
    int i = 0, j = 0, k = left;

    // Сливаем два массива обратно в arr
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    // Копируем оставшиеся элементы
    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

// Рекурсивная функция Merge Sort
void mergeSort(vector<int>& arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        // Сортируем левую и правую части
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Сливаем отсортированные части
        merge(arr, left, mid, right);
    }
}

/* -----------------------------------------
   QUICK SORT (последовательная)
   -----------------------------------------
*/

// Разделение массива относительно опорного элемента
int partition(vector<int>& arr, int low, int high)
{
    int pivot = arr[high];   // Опорный элемент
    int i = low - 1;         // Индекс меньшего элемента

    for (int j = low; j < high; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Рекурсивная функция Quick Sort
void quickSort(vector<int>& arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);

        // Рекурсивно сортируем части
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/*
   -----------------------------------------
   HEAP SORT (последовательная)
   -----------------------------------------
*/

// Восстановление свойства кучи
void heapify(vector<int>& arr, int n, int i)
{
    int largest = i;        // Корень
    int left = 2 * i + 1;   // Левый потомок
    int right = 2 * i + 2;  // Правый потомок

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    // Если корень не самый большой — меняем и рекурсивно продолжаем
    if (largest != i)
    {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

// Основная функция Heap Sort
void heapSort(vector<int>& arr)
{
    int n = arr.size();

    // Построение кучи
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Извлекаем элементы из кучи
    for (int i = n - 1; i > 0; i--)
    {
        swap(arr[0], arr[i]);      // Перемещаем максимум в конец
        heapify(arr, i, 0);        // Восстанавливаем кучу
    }
}

/*
   Другие функции
*/

// Генерация массива случайных чисел
vector<int> generateArray(int size)
{
    vector<int> arr(size);
    for (int i = 0; i < size; i++)
        arr[i] = rand() % 100000;
    return arr;
}

// Тестирование всех сортировок для одного размера массива
void testSorts(int size)
{
    cout << "\n";
    cout << "Array size: " << size << endl;

    vector<int> arr1 = generateArray(size);
    vector<int> arr2 = arr1;
    vector<int> arr3 = arr1;

    // -------- Merge Sort --------
    auto start = high_resolution_clock::now();
    mergeSort(arr1, 0, size - 1);
    auto end = high_resolution_clock::now();
    cout << "Merge Sort CPU: "
        << duration_cast<milliseconds>(end - start).count()
        << " ms" << endl;

    // -------- Quick Sort --------
    start = high_resolution_clock::now();
    quickSort(arr2, 0, size - 1);
    end = high_resolution_clock::now();
    cout << "Quick Sort CPU: "
        << duration_cast<milliseconds>(end - start).count()
        << " ms" << endl;

    // -------- Heap Sort --------
    start = high_resolution_clock::now();
    heapSort(arr3);
    end = high_resolution_clock::now();
    cout << "Heap Sort CPU: "
        << duration_cast<milliseconds>(end - start).count()
        << " ms" << endl;
}

/*
   MAIN part
*/

int main()
{
    srand(time(nullptr));   // Инициализация генератора случайных чисел

    // Тестируем для разных размеров массива
    testSorts(10000);
    testSorts(100000);
    testSorts(1000000);

    return 0;
}
