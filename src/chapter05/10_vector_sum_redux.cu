#include <memory>

#include "../common/handler.cuh"

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
    std::unique_ptr<int[]> a, b, c;
    int                   *dev_a, *dev_b, *dev_c;

    a = std::make_unique<int[]>(N);
    b = std::make_unique<int[]>(N);
    c = std::make_unique<int[]>(N);

    CUDA_CHECK(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    CUDA_CHECK(cudaMemcpy(dev_a, a.get(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b.get(), N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<(N + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(dev_a, dev_b, dev_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&c[0], dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        LOG_INFO(a[i], " + ", b[i], " = ", c[i]);
        if ((a[i] + b[i]) != c[i]) {
            LOG_ERROR("Error: ", a[i], " + ", b[i], " != ", c[i]);
            return 1;
        }
    }
    LOG_SUCCESS("Vector addition completed successfully!");

    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));

    return 0;
}

/*
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:26] ✅ cudaMalloc((void **)&dev_a, N * sizeof(int))
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:27] ✅ cudaMalloc((void **)&dev_b, N * sizeof(int))
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:28] ✅ cudaMalloc((void **)&dev_c, N * sizeof(int))
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:35] ✅ cudaMemcpy(dev_a, a.get(), N * sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:36] ✅ cudaMemcpy(dev_b, b.get(), N * sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:39] ✅ cudaGetLastError()
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:41] ✅ cudaMemcpy(&c[0], dev_c, N * sizeof(int),
 * cudaMemcpyDeviceToHost)
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ 0 + 0 = 0
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -1 + 1 = 0
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -2 + 4 = 2
 * ...
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -65530 + -786396 = -851926
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -65531 + -655335 = -720866
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -65532 + -524272 = -589804
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -65533 + -393207 = -458740
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -65534 + -262140 = -327674
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:44] ℹ️ -65535 + -131071 = -196606
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:50] ✅ Vector addition completed successfully!
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:52] ✅ cudaFree(dev_a)
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:53] ✅ cudaFree(dev_b)
 * [2025-09-03 21:20:34] [src/chapter05/10_vector_sum_redux.cu:54] ✅ cudaFree(dev_c)
 */
