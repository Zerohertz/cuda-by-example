#include <stdio.h>

#include "../../include/handler.cuh"

__constant__ int constant_data;

__global__ void kernel()
{
    int data = constant_data;
    printf("Data from constant memory: %d\n", data);
    // WARN: __global__ 내에선 사용 불가
    // LOG_INFO("Data from constant memory: ", data);
    // std::cout << "Data from constant memory: " << data << std::endl;
}

int main()
{
    int host_data = 100;

    CUDA_CHECK(cudaMemcpyToSymbol(constant_data, &host_data, sizeof(int)));

    kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/*
 * [2025-09-23 00:17:21] [src/lecture/chapter04/07_constant.cu:20] ✅ cudaMemcpyToSymbol(constant_data, &host_data,
 * sizeof(int))
 * [2025-09-23 00:17:21] [src/lecture/chapter04/07_constant.cu:23] ✅ cudaGetLastError()
 * Data from constant memory: 100
 * [2025-09-23 00:17:21] [src/lecture/chapter04/07_constant.cu:25] ✅ cudaDeviceSynchronize()
 */
