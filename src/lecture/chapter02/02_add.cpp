#include "../../include/logger.hpp"

#define size 5

void arrayAdd(const int *a, const int *b, int *c, int _size)
{
    for (int i = 0; i < _size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int a[size] = {1, 2, 3, 4, 5};
    int b[size] = {10, 20, 30, 40, 50};
    int c[size];

    arrayAdd(a, b, c, size);

    LOG_INFO(a, " + ", b, " = ", c);

    return 0;
}

/*
 * [2025-09-22 22:34:46] [src/lecture/chapter02/02_add.cpp:21] ℹ️ [1, 2, 3, 4, 5] + [10, 20, 30, 40, 50] = [11, 22, 33,
 * 44, 55]
 */
