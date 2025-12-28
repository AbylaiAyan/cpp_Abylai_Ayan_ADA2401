// practice2.cpp
// Практическая работа №2
// Тема: Параллельная реализация простых алгоритмов сортировки
// (пузырьком, выбором, вставкой) на CPU с использованием OpenMP
//
// Программа:
// 1) Создаёт массив из N элементов и заполняет его случайными числами
// 2) Выполняет сортировки:
//    - пузырьком (последовательно и параллельно)
//    - выбором (последовательно и параллельно)
//    - вставками (последовательно и параллельно)
// 3) Измеряет время выполнения каждой версии с помощью <chrono>
// 4) Выводит результаты для сравнения
//

#include <iostream>     // cin, cout
#include <vector>       // Контейнер vector
#include <random>       // Генерация случайных чисел
#include <chrono>       // Измерение времени выполнения
#include <algorithm>    // для swap()

// Подключаем OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// 
// Генерация массива
// Создаёт массив из N элементов и заполняет его
// случайными числами в диапазоне [1; 100000]
// 
vector<int> generateArray(size_t N) {
    vector<int> a(N);   // создаём массив нужного размера

    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100000);

    // Заполняем массив случайными значениями
    for (size_t i = 0; i < N; ++i)
        a[i] = dist(gen);

    return a; // возвращаем готовый массив
}

// 
// 1. Сортировка ПУЗЫРЬКОМ (Bubble Sort)
// 
// Последовательная версия сортировки пузырьком
// Идея: сравниваем соседние элементы и меняем их местами,
// если они стоят в неправильном порядке
// Временная сложность: O(n^2)
// -------------------------------------------------------
void bubbleSortSequential(vector<int>& a) {
    size_t n = a.size();

    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = 0; j < n - i - 1; ++j) {
            // Если текущий элемент больше следующего — меняем их местами
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}


// Параллельная версия пузырьковой сортировки
// Используется odd-even схема (чётно-нечётные фазы)

// На чётных шагах сравниваются пары (0,1), (2,3), и тд
// На нечётных шагах — (1,2), (3,4), и тд
// Это позволяет избежать конфликтов между потоками
// -------------------------------------------------------
void bubbleSortParallel(vector<int>& a) {
    size_t n = a.size();

    for (size_t phase = 0; phase < n; ++phase) {

        // ЧЁТНАЯ фаза
        if (phase % 2 == 0) {
#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(n - 1); i += 2) {
                if (a[i] > a[i + 1]) {
                    swap(a[i], a[i + 1]);
                }
            }
        }
        // НЕЧЁТНАЯ фаза
        else {
#pragma omp parallel for
            for (int i = 1; i < static_cast<int>(n - 1); i += 2) {
                if (a[i] > a[i + 1]) {
                    swap(a[i], a[i + 1]);
                }
            }
        }
    }
}

// 
// 2. Сортировка ВЫБОРОМ (Selection Sort)
// 
// Последовательная версия сортировки выбором
// На каждом шаге ищем минимальный элемент в неотсортированной части массива
// -------------------------------------------------------------------------
void selectionSortSequential(vector<int>& a) {
    size_t n = a.size();

    for (size_t i = 0; i < n - 1; ++i) {
        size_t minIndex = i;

        // Поиск минимального элемента
        for (size_t j = i + 1; j < n; ++j) {
            if (a[j] < a[minIndex]) {
                minIndex = j;
            }
        }

        // Перемещаем минимальный элемент в начало
        swap(a[i], a[minIndex]);
    }
}


// Параллельная версия сортировки выбором
// Поиск минимального элемента выполняется параллельно
// Каждый поток ищет свой локальный минимум, затем выбирается общий минимальный элемент
// --------------------------------------------------------------------------------------
void selectionSortParallel(vector<int>& a) {
    size_t n = a.size();

    for (size_t i = 0; i < n - 1; ++i) {
        int minValue = a[i];
        size_t minIndex = i;

#pragma omp parallel
        {
            int localMin = minValue;
            size_t localIndex = minIndex;

#pragma omp for nowait
            for (int j = i + 1; j < static_cast<int>(n); ++j) {
                if (a[j] < localMin) {
                    localMin = a[j];
                    localIndex = j;
                }
            }

            // Критическая секция — обновление общего минимума
#pragma omp critical
            {
                if (localMin < minValue) {
                    minValue = localMin;
                    minIndex = localIndex;
                }
            }
        }

        swap(a[i], a[minIndex]);
    }
}

// 
// 3. Сортировка ВСТАВКАМИ (Insertion Sort)
// 
// Последовательная сортировка вставками
// Элементы по одному вставляются в уже отсортированную часть
// -----------------------------------------------------------
void insertionSortSequential(vector<int>& a) {
    size_t n = a.size();

    for (size_t i = 1; i < n; ++i) {
        int key = a[i];
        int j = static_cast<int>(i) - 1;

        // Сдвигаем элементы вправо, пока не найдём нужное место
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}


// Параллельная версия сортировки вставками (учебная)
// Из-за зависимостей между итерациями данный алгоритм плохо подходит для параллелизации
// -------------------------------------------------------
void insertionSortParallel(vector<int>& a) {
    size_t n = a.size();

#pragma omp parallel for
    for (int i = 1; i < static_cast<int>(n); ++i) {
        int key = a[i];
        int j = i - 1;

        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// 
// MAIN
// 
int main() {

    // Ввод размера массива
    size_t N;
    cout << "Enter array size: ";
    cin >> N;

    // Создаём исходный массив
    vector<int> base = generateArray(N);


    // ПУЗЫРЬКОВАЯ СОРТИРОВКА
    // ===================================================
    auto a1 = base;
    auto a2 = base;

    auto t1 = chrono::high_resolution_clock::now();
    bubbleSortSequential(a1);
    auto t2 = chrono::high_resolution_clock::now();

    auto t3 = chrono::high_resolution_clock::now();
    bubbleSortParallel(a2);
    auto t4 = chrono::high_resolution_clock::now();

    cout << "\nBubble Sort:\n";
    cout << "  Sequential: "
        << chrono::duration<double, milli>(t2 - t1).count() << " ms\n";
    cout << "  Parallel:   "
        << chrono::duration<double, milli>(t4 - t3).count() << " ms\n";


    // СОРТИРОВКА ВЫБОРОМ
    // ===================================================
    auto b1 = base;
    auto b2 = base;

    t1 = chrono::high_resolution_clock::now();
    selectionSortSequential(b1);
    t2 = chrono::high_resolution_clock::now();

    t3 = chrono::high_resolution_clock::now();
    selectionSortParallel(b2);
    t4 = chrono::high_resolution_clock::now();

    cout << "\nSelection Sort:\n";
    cout << "  Sequential: "
        << chrono::duration<double, milli>(t2 - t1).count() << " ms\n";
    cout << "  Parallel:   "
        << chrono::duration<double, milli>(t4 - t3).count() << " ms\n";


    // СОРТИРОВКА ВСТАВКАМИ
    // ===================================================
    auto c1 = base;
    auto c2 = base;

    t1 = chrono::high_resolution_clock::now();
    insertionSortSequential(c1);
    t2 = chrono::high_resolution_clock::now();

    t3 = chrono::high_resolution_clock::now();
    insertionSortParallel(c2);
    t4 = chrono::high_resolution_clock::now();

    cout << "\nInsertion Sort:\n";
    cout << "  Sequential: "
        << chrono::duration<double, milli>(t2 - t1).count() << " ms\n";
    cout << "  Parallel:   "
        << chrono::duration<double, milli>(t4 - t3).count() << " ms\n";

#ifdef _OPENMP
    // Вывод количества потоков OpenMP
    cout << "\nOpenMP max threads: " << omp_get_max_threads() << endl;
#else
    cout << "\nOpenMP not enabled\n";
#endif

    return 0;
}