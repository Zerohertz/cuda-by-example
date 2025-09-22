#include "../../include/handler.cuh"

int main(void)
{
    cudaDeviceProp prop;
    int            dev;

    CUDA_CHECK(cudaGetDevice(&dev));
    LOG_INFO("ID of current CUDA device: ", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    CUDA_CHECK(cudaChooseDevice(&dev, &prop));
    LOG_INFO("ID of CUDA device closet to version 1.3: ", dev);

    CUDA_CHECK(cudaSetDevice(dev));
    return 0;
}

/*
 * [2025-09-03 20:54:42] [src/chapter03/05_device_properties.cu:8] ✅ cudaGetDevice(&dev)
 * [2025-09-03 20:54:42] [src/chapter03/05_device_properties.cu:9] ℹ️ ID of current CUDA device: 0
 * [2025-09-03 20:54:42] [src/chapter03/05_device_properties.cu:14] ✅ cudaChooseDevice(&dev, &prop)
 * [2025-09-03 20:54:42] [src/chapter03/05_device_properties.cu:15] ℹ️ ID of CUDA device closet to version 1.3: 0
 * [2025-09-03 20:54:43] [src/chapter03/05_device_properties.cu:17] ✅ cudaSetDevice(dev)
 */
