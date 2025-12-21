// task4.cpp
// Задание 4
// Вычисление среднего значения элементов массива
// Последовательно и параллельно с использованием OpenMP (reduction)

#include <iostream>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>   // библиотека для работы с OpenMP
#endif

using namespace std;
using namespace chrono;

int main() {

    // Размер массива по условию задания
    int size = 5'000'000;

    // Динамически выделяем память под массив
    int* arr = new int[size];

    // Создаём генератор случайных чисел
    random_device rd;          // источник случайности
    mt19937 gen(rd());         // генератор Mersenne Twister
    uniform_int_distribution<int> dist(1, 100); // числа от 1 до 100

    // Заполняем массив случайными значениями
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    /* =====================================================
       ПОСЛЕДОВАТЕЛЬНОЕ ВЫЧИСЛЕНИЕ СРЕДНЕГО ЗНАЧЕНИЯ
       ===================================================== */

       // Начинаем замер времени
    auto startSeq = high_resolution_clock::now();

    // Переменная для суммы элементов массива
    long long sumSeq = 0;

    // Обычный цикл без распараллеливания
    for (int i = 0; i < size; i++) {
        sumSeq += arr[i];
    }

    // Вычисляем среднее значение
    double avgSeq = (double)sumSeq / size;

    // останавливаем замер времени
    auto endSeq = high_resolution_clock::now();
    auto timeSeq = duration_cast<milliseconds>(endSeq - startSeq);

#ifdef _OPENMP
    /* =====================================================
       ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ СРЕДНЕГО ЗНАЧЕНИЯ (OpenMP)
       ===================================================== */

       // начало замера времени
    auto startPar = high_resolution_clock::now();

    // Общая переменная суммы (будет использоваться в reduction)
    long long sumPar = 0;

    // Параллельный цикл с редукцией
    // Каждый поток считает свою часть суммы,
    // после чего OpenMP автоматически объединяет результаты
#pragma omp parallel for reduction(+:sumPar)
    for (int i = 0; i < size; i++) {
        sumPar += arr[i];
    }

    // Вычисляем среднее значение
    double avgPar = (double)sumPar / size;

    // Останавливаем замер времени
    auto endPar = high_resolution_clock::now();
    auto timePar = duration_cast<milliseconds>(endPar - startPar);

    // Вывод результатов
    cout << "Sequential average: " << avgSeq
        << ", time: " << timeSeq.count() << " ms" << endl;

    cout << "Parallel average:   " << avgPar
        << ", time: " << timePar.count() << " ms" << endl;

#else
    // Если OpenMP не поддерживается
    cout << "Sequential average: " << avgSeq << endl;
    cout << "OpenMP is not enabled" << endl;
#endif

    // Освобождаем выделенную память
    delete[] arr;

    return 0;
}

