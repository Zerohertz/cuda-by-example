#include "../include/cpu_anim.hpp"
#include "../include/handler.cuh"

#define DIM           1024
#define PI            3.1415926535897932f
#define FRAME_PER_SEC 20
#define MS_PER_FRAME  1000 / FRAME_PER_SEC
#define NUM_FRAME     50


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

    bitmap.start_recording(MS_PER_FRAME);

    LOG_INFO("Generating ripple animation frames...");
    for (int ticks = 0; ticks < NUM_FRAME; ticks++) {
        generate_frame(&data, ticks);
        bitmap.capture_frame();
        if (ticks % 10 == 0) {
            LOG_INFO("Progress: ", ticks, "/", NUM_FRAME, " frames");
        }
    }

    bitmap.stop_recording();

    bitmap.save_to_gif();
    LOG_INFO("GIF saved successfully!");

    cleanup(&data);
    return 0;
}

/*
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:50] ✅ cudaMalloc((void **)&data.dev_bitmap,
 * bitmap.image_size())
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:54] ℹ️ Generating ripple animation frames...
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:59] ℹ️ Progress: 0/50 frames
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:59] ℹ️ Progress: 10/50 frames
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:59] ℹ️ Progress: 20/50 frames
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:59] ℹ️ Progress: 30/50 frames
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:59] ℹ️ Progress: 40/50 frames
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:10] [src/chapter05/11_gpu_ripple_threads.cu:40] ✅ cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
 * d->bitmap->image_size(), cudaMemcpyDeviceToHost)
 * [2025-09-09 21:04:15] [src/chapter05/11_gpu_ripple_threads.cu:66] ℹ️ GIF saved successfully!
 * [2025-09-09 21:04:15] [src/chapter05/11_gpu_ripple_threads.cu:42] ✅ cudaFree(d->dev_bitmap)
 */
