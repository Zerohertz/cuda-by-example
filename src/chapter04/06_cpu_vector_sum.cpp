#include <iostream>

using namespace std;

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
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }
    return 0;
}

/*
0 + 0 = 0
-1 + 1 = 0
-2 + 4 = 2
-3 + 9 = 6
-4 + 16 = 12
-5 + 25 = 20
-6 + 36 = 30
-7 + 49 = 42
-8 + 64 = 56
-9 + 81 = 72
*/
