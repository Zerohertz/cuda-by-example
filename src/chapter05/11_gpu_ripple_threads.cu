#include "../include/cpu_anim.hpp"
#include "../include/handler.cuh"
#include "../include/utils.hpp"

#define DIM 1024
#define PI  3.1415926535897932f


__global__ void kernel(unsigned char *ptr, int ticks)
{
    int x      = threadIdx.x + blockIdx.x * blockDim.x;
    int y      = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if (x >= DIM || y >= DIM)
        return;
    float         fx    = x - DIM / 2.0f;
    float         fy    = y - DIM / 2.0f;
    float         d     = sqrtf(fx * fx + fy * fy);
    unsigned char grey  = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

struct DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void generate_frame(DataBlock *d, int ticks)
{
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    CUDA_CHECK(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}
void cleanup(DataBlock *d) { CUDA_CHECK(cudaFree(d->dev_bitmap)); }

int main(void)
{
    DataBlock     data;
    CPUAnimBitmap bitmap(DIM, DIM, &data, __FILE__);
    data.bitmap = &bitmap;

    CUDA_CHECK(cudaMalloc((void **)&data.dev_bitmap, bitmap.image_size()));

    bitmap.start_recording(5); // 50ms per frame

    LOG_INFO("Generating ripple animation frames...");
    for (int ticks = 0; ticks < 50; ticks++) {
        generate_frame(&data, ticks);
        bitmap.capture_frame();
        if (ticks % 10 == 0) {
            LOG_INFO("Progress: ", ticks, "/50 frames");
        }
    }
    bitmap.stop_recording();

    for (int i = 0; i < 3; i++) {
        generate_frame(&data, i * 10);
    }

    bitmap.save_to_gif();
    LOG_INFO("GIF saved successfully!");

    cleanup(&data);
    return 0;
}
