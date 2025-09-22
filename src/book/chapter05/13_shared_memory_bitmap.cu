#include "../../include/cpu_bitmap.hpp"
#include "../../include/handler.cuh"

#define DIM 1024
#define PI  3.1415926535897932f


__global__ void kernel(unsigned char *ptr)
{
    int x      = threadIdx.x + blockIdx.x * blockDim.x;
    int y      = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];

    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
        255 * (sinf(x * 2.0f * PI / period) + 1.0f) * (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
    __syncthreads();

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

struct DataBlock
{
    unsigned char *dev_bitmap;
};

int main(void)
{
    DataBlock      data;
    CPUBitmap      bitmap(DIM, DIM, &data, __FILE__);
    unsigned char *dev_bitmap;

    CUDA_CHECK(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    data.dev_bitmap = dev_bitmap;

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    CUDA_CHECK(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_bitmap));

    bitmap.save_to_png();

    return 0;
}

/*
 * [2025-09-09 21:17:30] [src/chapter05/13_shared_memory_bitmap.cu:39] ✅ cudaMalloc((void **)&dev_bitmap,
 * bitmap.image_size())
 * [2025-09-09 21:17:30] [src/chapter05/13_shared_memory_bitmap.cu:47] ✅ cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
 * bitmap.image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:17:30] [src/chapter05/13_shared_memory_bitmap.cu:48] ✅ cudaFree(dev_bitmap)
 */
