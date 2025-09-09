#include "../include/cpu_bitmap.hpp"
#include "../include/handler.cuh"

#define DIM 1000

struct cuComplex
{
    float      r;
    float      i;
    __device__ cuComplex(float a, float b)
        : r(a)
        , i(b)
    {
    }
    __device__ float     magnitude2(void) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex &a) { return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i); }
    __device__ cuComplex operator+(const cuComplex &a) { return cuComplex(r + a.r, i + a.i); }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float       jx    = scale * static_cast<float>(DIM / 2.0f - x) / (DIM / 2.0f);
    float       jy    = scale * static_cast<float>(DIM / 2.0f - y) / (DIM / 2.0f);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr)
{
    int x      = blockIdx.x;
    int y      = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue      = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
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

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_bitmap);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_bitmap));

    bitmap.save_to_png();

    return 0;
}

/*
 * [2025-09-03 21:12:03] [src/chapter04/09_gpu_julia.cu:63] ✅ cudaMalloc((void **)&dev_bitmap, bitmap.image_size())
 * [2025-09-03 21:12:03] [src/chapter04/09_gpu_julia.cu:68] ✅ cudaGetLastError()
 * [2025-09-03 21:12:03] [src/chapter04/09_gpu_julia.cu:70] ✅ cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
 * bitmap.image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-03 21:12:03] [src/chapter04/09_gpu_julia.cu:71] ✅ cudaFree(dev_bitmap)
 */
