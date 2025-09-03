#include "../common/handler.cuh"

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

    CUDA_CHECK(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    CUDA_CHECK(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<N, 1>>>(dev_a, dev_b, dev_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        LOG_INFO(a[i], " + ", b[i], " = ", c[i]);
    }
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));

    return 0;
}

/*
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:17] ✅ cudaMalloc((void **)&dev_a, N * sizeof(int))
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:18] ✅ cudaMalloc((void **)&dev_b, N * sizeof(int))
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:19] ✅ cudaMalloc((void **)&dev_c, N * sizeof(int))
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:26] ✅ cudaMemcpy(dev_a, a, N * sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:27] ✅ cudaMemcpy(dev_b, b, N * sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:30] ✅ cudaGetLastError()
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:32] ✅ cudaMemcpy(c, dev_c, N * sizeof(int),
 * cudaMemcpyDeviceToHost)
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ 0 + 0 = 0
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -1 + 1 = 0
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -2 + 4 = 2
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -3 + 9 = 6
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -4 + 16 = 12
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -5 + 25 = 20
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -6 + 36 = 30
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -7 + 49 = 42
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -8 + 64 = 56
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:34] ℹ️ -9 + 81 = 72
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:36] ✅ cudaFree(dev_a)
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:37] ✅ cudaFree(dev_b)
 * [2025-09-03 21:09:30] [src/chapter04/07_gpu_vector_sum.cu:38] ✅ cudaFree(dev_c)
 */
