// task2.cpp
// Задание 2
// Последовательный поиск минимального и максимального элементов
// Замер времени выполнения

#include <iostream>
#include <random>
#include <chrono>

using namespace std;
using namespace chrono;

int main() {

    int size = 1'000'000;

    // Динамический массив
    int* arr = new int[size];

    // Генератор случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    // Заполняем массив
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    // Начинаем замер времени
    auto start = high_resolution_clock::now();

    // Ищем минимум и максимум
    int minVal = arr[0];
    int maxVal = arr[0];

    for (int i = 1; i < size; i++) {
        if (arr[i] < minVal)
            minVal = arr[i];
        if (arr[i] > maxVal)
            maxVal = arr[i];
    }

    // Заканчиваем замер времени
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // Выводим результаты
    cout << "Min value: " << minVal << endl;
    cout << "Max value: " << maxVal << endl;
    cout << "Execution time: " << duration.count() << " ms" << endl;

    // Освобождаем память
    delete[] arr;

    return 0;
}
