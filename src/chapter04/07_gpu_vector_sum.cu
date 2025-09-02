#include <iostream>

using namespace std;

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void)
{
    int  a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}

/*
 * 0 + 0 = 0
 * -1 + 1 = 0
 * -2 + 4 = 2
 * -3 + 9 = 6
 * -4 + 16 = 12
 * -5 + 25 = 20
 * -6 + 36 = 30
 * -7 + 49 = 42
 * -8 + 64 = 56
 * -9 + 81 = 72
 */
