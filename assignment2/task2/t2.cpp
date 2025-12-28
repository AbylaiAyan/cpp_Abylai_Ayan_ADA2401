// lab2.cpp
// Работа с массивами и параллелизация OpenMP
//
// Программа выполняет следующие действия:
// 1) Создаёт массив из 10000 элементов
// 2) Заполняет массив случайными числами
// 3) Находит минимальное и максимальное значения массива
//    - последовательным способом
//    - параллельным способом с использованием OpenMP
// 4) Измеряет время выполнения каждого варианта
// 5) Сравнивает полученные результаты
//
// Сборка (пример для GCC):
// g++ -fopenmp -O2 -std=c++17 lab2.cpp -o lab2.exe
//
// Запуск:
// ./lab2.exe
//

#include <iostream>   // ввод и вывод данных
#include <vector>     // работа с динамическими массивами (vector)
#include <cstdlib>    // функция rand()
#include <omp.h>      // библиотека OpenMP для параллельных вычислений

int main() {

    // Размер массива
    const int N = 10000;

    // Создаём массив из N элементов
    std::vector<int> arr(N);

    // Заполняем массив случайными числами
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100000;  // числа от 0 до 99999
    }

    // ПОСЛЕДОВАТЕЛЬНЫЙ АЛГОРИТМ

    // Засекаем время начала выполнения
    double start = omp_get_wtime();

    // Начальные значения минимума и максимума
    int min_seq = arr[0];
    int max_seq = arr[0];

    // Последовательно просматриваем массив
    for (int i = 1; i < N; i++) {
        if (arr[i] < min_seq)
            min_seq = arr[i];

        if (arr[i] > max_seq)
            max_seq = arr[i];
    }

    // Время окончания выполнения
    double end = omp_get_wtime();

    // Вывод времени работы последовательного алгоритма
    std::cout << "Sequential algorithm time: "
        << (end - start) * 1000 << " ms" << std::endl;

    // ПАРАЛЛЕЛЬНЫЙ АЛГОРИТМ (OpenMP)

    // Снова замеряем время
    start = omp_get_wtime();

    // Начальные значения для параллельного варианта
    int min_par = arr[0];
    int max_par = arr[0];

    // Параллельный цикл
    // reduction используется для корректного вычисления min и max
#pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; i++) {
        if (arr[i] < min_par)
            min_par = arr[i];

        if (arr[i] > max_par)
            max_par = arr[i];
    }

    // Время окончания
    end = omp_get_wtime();

    // Вывод времени работы параллельного алгоритма
    std::cout << "Parallel algorithm time:   "
        << ( end - start )* 1000 << " ms" << std::endl;

    return 0;
}
