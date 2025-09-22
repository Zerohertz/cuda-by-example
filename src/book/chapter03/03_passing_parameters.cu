#include "../../include/handler.cuh"

constexpr int a = 2;
constexpr int b = 7;

using namespace std;

__global__ void add(int a, int b, int *c) { *c = a + b; }

int main(void)
{
    int  c;
    int *dev_c;

    CUDA_CHECK(cudaMalloc((void **)&dev_c, sizeof(int)));
    add<<<1, 1>>>(a, b, dev_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    // WARN:
    // [1]    7461 segmentation fault (core dumped)
    // cout << a << " + " << b << " = " << *dev_c << endl;
    LOG_INFO(a, " + ", b, " = ", c);

    // WARN:
    // 2 + 7 = 0
    // CUDA error at src/chapter03/03_passing_parameters.cu:23 - the provided PTX
    // was compiled with an unsupported toolchain.
    // NOTE:
    // -gencode arch=compute_90,code=sm_90

    CUDA_CHECK(cudaFree(dev_c));

    return 0;
}

/*
 * [2025-09-03 20:45:42] [src/chapter03/03_passing_parameters.cu:15] ✅ cudaMalloc((void **)&dev_c, sizeof(int))
 * [2025-09-03 20:45:42] [src/chapter03/03_passing_parameters.cu:17] ✅ cudaGetLastError()
 * [2025-09-03 20:45:42] [src/chapter03/03_passing_parameters.cu:18] ✅ cudaMemcpy(&c, dev_c, sizeof(int),
 * cudaMemcpyDeviceToHost)
 * [2025-09-03 20:45:42] [src/chapter03/03_passing_parameters.cu:23] ℹ️ 2 + 7 = 9
 * [2025-09-03 20:45:42] [src/chapter03/03_passing_parameters.cu:32] ✅ cudaFree(dev_c)
 */
