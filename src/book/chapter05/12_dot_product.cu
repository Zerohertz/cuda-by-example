#include <memory>

#include "../../include/handler.cuh"

#define imin(a, b)     (a < b ? a : b)
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

const int N               = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid   = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];

    int   tid        = threadIdx.x + blockIdx.x * blockDim.x;
    int   cacheIndex = threadIdx.x;
    float tmp        = 0;

    while (tid < N) {
        tmp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = tmp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(void)
{
    std::unique_ptr<float[]> a, b, partial_c;
    float                    c;
    float                   *dev_a, *dev_b, *dev_partial_c;

    a         = std::make_unique<float[]>(N);
    b         = std::make_unique<float[]>(N);
    partial_c = std::make_unique<float[]>(blocksPerGrid);

    CUDA_CHECK(cudaMalloc((void **)&dev_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float)));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    CUDA_CHECK(cudaMemcpy(dev_a, a.get(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b.get(), N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(partial_c.get(), dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    LOG_INFO("Does GPU value ", c, " = ", 2 * sum_squares(static_cast<float>(N - 1)), "?");

    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_partial_c));

    return 0;
}

/*
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:49] ✅ cudaMalloc((void **)&dev_a, N * sizeof(float))
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:50] ✅ cudaMalloc((void **)&dev_b, N * sizeof(float))
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:51] ✅ cudaMalloc((void **)&dev_partial_c, blocksPerGrid *
 * sizeof(float))
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:58] ✅ cudaMemcpy(dev_a, a.get(), N * sizeof(float),
 * cudaMemcpyHostToDevice)
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:59] ✅ cudaMemcpy(dev_b, b.get(), N * sizeof(float),
 * cudaMemcpyHostToDevice)
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:62] ✅ cudaGetLastError()
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:64] ✅ cudaMemcpy(partial_c.get(), dev_partial_c,
 * blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost)
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:70] ℹ️ Does GPU value 2.57236e+13 = 2.57236e+13?
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:72] ✅ cudaFree(dev_a)
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:73] ✅ cudaFree(dev_b)
 * [2025-09-03 21:23:37] [src/chapter05/11_dot_product.cu:74] ✅ cudaFree(dev_partial_c)
 */
