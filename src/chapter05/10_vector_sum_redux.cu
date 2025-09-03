#include <iostream>
#include <memory>

using namespace std;

#define BLOCK_DIM 128
#define N         (64 * 1024)

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    unique_ptr<int[]> a, b, c;
    int              *dev_a, *dev_b, *dev_c;

    a = make_unique<int[]>(N);
    b = make_unique<int[]>(N);
    c = make_unique<int[]>(N);

    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a.get(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.get(), N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<(N + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(&c[0], dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
        if ((a[i] + b[i]) != c[i]) {
            cout << "Error: " << a[i] << " + " << b[i] << " != " << c[i] << endl;
            return 1;
        }
    }
    cout << "Success!" << endl;
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
 * ...
 * -65532 + -524272 = -589804
 * -65533 + -393207 = -458740
 * -65534 + -262140 = -327674
 * -65535 + -131071 = -196606
 * Success!
 */
