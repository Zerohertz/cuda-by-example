#include "../common/handler.cuh"

__global__ void kernel(void) {}

int main(void)
{
    kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    LOG_INFO("Hello, World!");

    return 0;
}

/*
 * [2025-09-03 20:44:24] [src/chapter03/02_kernel_call.cu:8] ✅ cudaGetLastError()
 * [2025-09-03 20:44:24] [src/chapter03/02_kernel_call.cu:9] ℹ️ Hello, World!
 */
