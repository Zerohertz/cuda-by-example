#include "../common/handler.cuh"

int main(void)
{
    cudaDeviceProp prop;

    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    for (int i = 0; i < count; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        LOG_INFO("===== General Information for Device ", i, " =====");
        LOG_INFO("Name:\t", prop.name);
        LOG_INFO("Compute Capability:\t", prop.major, ".", prop.minor);
        LOG_INFO("Clock Rate:\t", prop.clockRate);
        LOG_INFO("Device Copy Overlap:\t", (prop.deviceOverlap ? "Enabled" : "Disabled"));
        LOG_INFO("Kernel Execition Timeout:\t", (prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled"));
        LOG_INFO("===== Memory Information for Device ", i, " =====");
        LOG_INFO("Total Global Mem:\t", prop.totalGlobalMem);
        LOG_INFO("Total Constant Mem:\t", prop.totalConstMem);
        LOG_INFO("Max Mem pitch:\t", prop.memPitch);
        LOG_INFO("Texture Alignment:\t", prop.textureAlignment);
        LOG_INFO("===== MP Information for Device ", i, " =====");
        LOG_INFO("Multiprocessor Count:\t", prop.multiProcessorCount);
        LOG_INFO("Shared Mem per MP:\t", prop.sharedMemPerBlock);
        LOG_INFO("Registers per MP:\t", prop.regsPerBlock);
        LOG_INFO("Threads in Warp:\t", prop.warpSize);
        LOG_INFO("Max Threads per Block:\t", prop.maxThreadsPerBlock);
        LOG_INFO("Max Trhead Dimensions:\t(",
                 prop.maxThreadsDim[0],
                 ", ",
                 prop.maxThreadsDim[1],
                 ", ",
                 prop.maxThreadsDim[2],
                 ")");
        LOG_INFO(
            "Max Grid Dimensions:\t(", prop.maxGridSize[0], ", ", prop.maxGridSize[1], ", ", prop.maxGridSize[2], ")");
    }

    return 0;
}

/*
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:8] ✅ cudaGetDeviceCount(&count)
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:11] ✅ cudaGetDeviceProperties(&prop, i)
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:13] ℹ️ ===== General Information for Device 0 =====
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:14] ℹ️ Name:        NVIDIA H100 80GB HBM3
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:15] ℹ️ Compute Capability:  9.0
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:16] ℹ️ Clock Rate:  1980000
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:17] ℹ️ Device Copy Overlap: Enabled
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:18] ℹ️ Kernel Execition Timeout:    Disabled
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:19] ℹ️ ===== Memory Information for Device 0 =====
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:20] ℹ️ Total Global Mem:    84929216512
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:21] ℹ️ Total Constant Mem:  65536
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:22] ℹ️ Max Mem pitch:       2147483647
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:23] ℹ️ Texture Alignment:   512
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:24] ℹ️ ===== MP Information for Device 0 =====
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:25] ℹ️ Multiprocessor Count:        132
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:26] ℹ️ Shared Mem per MP:   49152
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:27] ℹ️ Registers per MP:    65536
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:28] ℹ️ Threads in Warp:     32
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:29] ℹ️ Max Threads per Block:       1024
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:36] ℹ️ Max Trhead Dimensions:       (1024, 1024, 64)
 * [2025-09-03 20:53:34] [src/chapter03/04_querying_devices.cu:38] ℹ️ Max Grid Dimensions: (2147483647, 65535, 65535)
 */
