#include "../../include/handler.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i]  = a[i] + b[i];
}

int main()
{
    const int arraySize    = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int       c[arraySize] = {0};

    addWithCuda(c, a, b, arraySize);

    LOG_INFO(a, " + ", b, " = ", c);

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_a, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));
}

/*
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:35] ✅ cudaSetDevice(0)
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:36] ✅ cudaMalloc((void **)&dev_c, size * sizeof(int))
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:37] ✅ cudaMalloc((void **)&dev_a, size * sizeof(int))
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:38] ✅ cudaMalloc((void **)&dev_b, size * sizeof(int))
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:39] ✅ cudaMemcpy(dev_a, a, size * sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:40] ✅ cudaMemcpy(dev_b, b, size * sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:43] ✅ cudaGetLastError()
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:44] ✅ cudaDeviceSynchronize()
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:45] ✅ cudaMemcpy(c, dev_c, size * sizeof(int),
 * cudaMemcpyDeviceToHost)
 * [2025-09-22 22:28:47] [src/lecture/chapter02/01_add.cu:22] ℹ️ [1, 2, 3, 4, 5] + [10, 20, 30, 40, 50] = [11, 22, 33,
 * 44, 55]
 * [2025-09-22 22:28:48] [src/lecture/chapter02/01_add.cu:24] ✅ cudaDeviceReset()
 */
