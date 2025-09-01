#include <iostream>

#include "../common/book.h"
#include "../common/util.h"

constexpr int a = 2;
constexpr int b = 7;

using namespace std;

__global__ void add(int a, int b, int *c) { *c = a + b; }

int main(void)
{
    int  c;
    int *dev_c;
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(int)));
    add<<<1, 1>>>(a, b, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    // WARN:
    // [1]    7461 segmentation fault (core dumped)
    // cout << a << " + " << b << " = " << *dev_c << endl;
    cout << a << " + " << b << " = " << c << endl;
    cudaFree(dev_c);
    CHECK_CUDA_ERROR();
    // WARN:
    // 2 + 7 = 0
    // CUDA error at src/chapter03/03_passing_parameters.cu:23 - the provided PTX
    // was compiled with an unsupported toolchain.
    // NOTE:
    // -gencode arch=compute_90,code=sm_90
    return 0;
}
