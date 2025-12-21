// task1.cpp
// Задание 1
// Динамический массив из 50 000 элементов
// Заполнение случайными числами, вычисление среднего и замер времени

#include <iostream>
#include <random>
#include <chrono>   // для замера времени

using namespace std;
using namespace chrono;

int main() {

    int size = 50000;

    // Выделяем память под динамический массив
    int* arr = new int[size];

    // Генератор случайных чисел от 1 до 100
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    // Заполняем массив случайными числами
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    // НАЧАЛО замера времени
    auto start = high_resolution_clock::now();

    // Считаем сумму элементов массива
    long long sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    // Вычисляем среднее значение
    double average = (double)sum / size;

    // КОНЕЦ замера времени
    auto end = high_resolution_clock::now();

    // Вычисляем время в секундах
    duration<double> time_sec = end - start;

    // Выводим результат
    cout << "Average value: " << average << endl;
    cout << "Execution time: " << time_sec.count() << " seconds" << endl;

    // Освобождаем выделенную память
    delete[] arr;

    // Чтобы окно консоли не закрывалось
    cin.get();

    return 0;
}
