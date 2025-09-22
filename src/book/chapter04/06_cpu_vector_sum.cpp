#include "../../include/logger.hpp"

#define N 10


void add(int *a, int *b, int *c)
{
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid++;
    }
}

int main(void)
{
    int a[N], b[N], c[N];

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    add(a, b, c);
    for (int i = 0; i < N; i++) {
        LOG_INFO(a[i], " + ", b[i], " = ", c[i]);
    }
    return 0;
}

/*
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ 0 + 0 = 0
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -1 + 1 = 0
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -2 + 4 = 2
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -3 + 9 = 6
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -4 + 16 = 12
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -5 + 25 = 20
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -6 + 36 = 30
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -7 + 49 = 42
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -8 + 64 = 56
 * [2025-09-03 21:05:38] [src/chapter04/06_cpu_vector_sum.cpp:25] ℹ️ -9 + 81 = 72
 */
