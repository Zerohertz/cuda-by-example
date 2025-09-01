#include <iostream>

using namespace std;

int main(void)
{
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);

        cout << "===== General Information for Device " << i << " =====" << endl;
        cout << "Name:\t" << prop.name << endl;
        cout << "Compute Capability:\t" << prop.major << "." << prop.minor << endl;
        cout << "Clock Rate:\t" << prop.clockRate << endl;
        cout << "Device Copy Overlap:\t";
        if (prop.deviceOverlap)
            cout << "Enabled" << endl;
        else
            cout << "Disabled" << endl;
        cout << "Kernel Execition Timeout:\t";
        if (prop.kernelExecTimeoutEnabled)
            cout << "Enabled" << endl;
        else
            cout << "Disabled" << endl;
        cout << "===== Memory Information for Device " << i << " =====" << endl;
        cout << "Total Global Mem:\t" << prop.totalGlobalMem << endl;
        cout << "Total Constant Mem:\t" << prop.totalConstMem << endl;
        cout << "Max Mem pitch:\t" << prop.memPitch << endl;
        cout << "Texture Alignment:\t" << prop.textureAlignment << endl;
        cout << "===== MP Information for Device " << i << " =====" << endl;
        cout << "Multiprocessor Count:\t" << prop.multiProcessorCount << endl;
        cout << "Shared Mem per MP:\t" << prop.sharedMemPerBlock << endl;
        cout << "Registers per MP:\t" << prop.regsPerBlock << endl;
        cout << "Threads in Warp:\t" << prop.warpSize << endl;
        cout << "Max Threads per Block:\t" << prop.maxThreadsPerBlock << endl;
        cout << "Max Trhead Dimensions:\t(" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
             << prop.maxThreadsDim[2] << ")" << endl;
        cout << "Max Grid Dimensions:\t(" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
             << prop.maxGridSize[2] << ")" << endl;
    }
}

/*
===== General Information for Device 0 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 0 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 0 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 1 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 1 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 1 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 2 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 2 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 2 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 3 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 3 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 3 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 4 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 4 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 4 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 5 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 5 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 5 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 6 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 6 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 6 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
===== General Information for Device 7 =====
Name:   NVIDIA H100 80GB HBM3
Compute Capability:     9.0
Clock Rate:     1980000
Device Copy Overlap:    Enabled
Kernel Execition Timeout:       Disabled
===== Memory Information for Device 7 =====
Total Global Mem:       84929216512
Total Constant Mem:     65536
Max Mem pitch:  2147483647
Texture Alignment:      512
===== MP Information for Device 7 =====
Multiprocessor Count:   132
Shared Mem per MP:      49152
Registers per MP:       65536
Threads in Warp:        32
Max Threads per Block:  1024
Max Trhead Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:    (2147483647, 65535, 65535)
*/

// struct __device_builtin__ cudaDeviceProp
// {
//     char         name[256];                  /**< ASCII string identifying device */
//     cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
//     char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and
//     non-Windows platforms */ unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined
//     on TCC and non-Windows platforms */ size_t       totalGlobalMem;             /**< Global memory available on
//     device in bytes */ size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
//     int          regsPerBlock;               /**< 32-bit registers available per block */
//     int          warpSize;                   /**< Warp size in threads */
//     size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
//     int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
//     int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
//     int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
//     int          clockRate;                  /**< Deprecated, Clock frequency in kilohertz */
//     size_t       totalConstMem;              /**< Constant memory available on device in bytes */
//     int          major;                      /**< Major compute capability */
//     int          minor;                      /**< Minor compute capability */
//     size_t       textureAlignment;           /**< Alignment requirement for textures */
//     size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched
//     memory */ int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel.
//     Deprecated. Use instead asyncEngineCount. */ int          multiProcessorCount;        /**< Number of
//     multiprocessors on device */ int          kernelExecTimeoutEnabled;   /**< Deprecated, Specified whether there is
//     a run time limit on kernels */ int          integrated;                 /**< Device is integrated as opposed to
//     discrete */ int          canMapHostMemory;           /**< Device can map host memory with
//     cudaHostAlloc/cudaHostGetDevicePointer */ int          computeMode;                /**< Deprecated, Compute mode
//     (See ::cudaComputeMode) */ int          maxTexture1D;               /**< Maximum 1D texture size */ int
//     maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */ int          maxTexture1DLinear; /**<
//     Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth()
//     instead. */ int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */ int
//     maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */ int          maxTexture2DLinear[3];
//     /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */ int
//     maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed
//     */ int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */ int maxTexture3DAlt[3]; /**<
//     Maximum alternate 3D texture dimensions */ int          maxTextureCubemap;          /**< Maximum Cubemap texture
//     dimensions */ int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */ int
//     maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */ int maxTextureCubemapLayered[2];/**<
//     Maximum Cubemap layered texture dimensions */ int          maxSurface1D;               /**< Maximum 1D surface
//     size */ int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */ int maxSurface3D[3]; /**<
//     Maximum 3D surface dimensions */ int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface
//     dimensions */ int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */ int
//     maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */ int maxSurfaceCubemapLayered[2];/**<
//     Maximum Cubemap layered surface dimensions */ size_t       surfaceAlignment;           /**< Alignment
//     requirements for surfaces */ int          concurrentKernels;          /**< Device can possibly execute multiple
//     kernels concurrently */ int          ECCEnabled;                 /**< Device has ECC support enabled */ int
//     pciBusID;                   /**< PCI bus ID of the device */ int          pciDeviceID;                /**< PCI
//     device ID of the device */ int          pciDomainID;                /**< PCI domain ID of the device */ int
//     tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */ int
//     asyncEngineCount;           /**< Number of asynchronous engines */ int          unifiedAddressing;          /**<
//     Device shares a unified address space with the host */ int          memoryClockRate;            /**< Deprecated,
//     Peak memory clock frequency in kilohertz */ int          memoryBusWidth;             /**< Global memory bus width
//     in bits */ int          l2CacheSize;                /**< Size of L2 cache in bytes */ int
//     persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */ int
//     maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */ int streamPrioritiesSupported;
//     /**< Device supports stream priorities */ int          globalL1CacheSupported;     /**< Device supports caching
//     globals in L1 */ int          localL1CacheSupported;      /**< Device supports caching locals in L1 */ size_t
//     sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */ int
//     regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */ int          managedMemory;
//     /**< Device supports allocating managed memory on this system */ int          isMultiGpuBoard;            /**<
//     Device is on a multi-GPU board */ int          multiGpuBoardGroupID;       /**< Unique identifier for a group of
//     devices on the same multi-GPU board */ int          hostNativeAtomicSupported;  /**< Link between the device and
//     the host supports native atomic operations */ int          singleToDoublePrecisionPerfRatio; /**< Deprecated,
//     Ratio of single precision performance (in floating-point operations per second) to double precision performance
//     */ int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without
//     calling cudaHostRegister on it */ int          concurrentManagedAccess;    /**< Device can coherently access
//     managed memory concurrently with the CPU */ int          computePreemptionSupported; /**< Device supports Compute
//     Preemption */ int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at
//     the same virtual address as the CPU */ int          cooperativeLaunch;          /**< Device supports launching
//     cooperative kernels via ::cudaLaunchCooperativeKernel */ int          cooperativeMultiDeviceLaunch; /**<
//     Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */ size_t       sharedMemPerBlockOptin; /**<
//     Per device maximum shared memory per block usable by special opt in */ int
//     pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */ int
//     directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
//     int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
//     int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
//     size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
//     int          hostRegisterSupported;      /**< Device supports host memory registration via ::cudaHostRegister. */
//     int          sparseCudaArraySupported;   /**< 1 if the device supports sparse CUDA arrays and sparse CUDA
//     mipmapped arrays, 0 otherwise */ int          hostRegisterReadOnlySupported; /**< Device supports using the
//     ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU
//     */ int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the
//     device */ int          memoryPoolsSupported;       /**< 1 if the device supports using the cudaMallocAsync and
//     cudaMemPool family of APIs, 0 otherwise */ int          gpuDirectRDMASupported;     /**< 1 if the device supports
//     GPUDirect RDMA APIs, 0 otherwise */ unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted
//     according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */ int          gpuDirectRDMAWritesOrdering;/**< See
//     the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */ unsigned int memoryPoolSupportedHandleTypes;
//     /**< Bitmask of handle types supported with mempool-based IPC */ int          deferredMappingCudaArraySupported;
//     /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */ int ipcEventSupported;
//     /**< Device supports IPC Events. */ int          clusterLaunch;              /**< Indicates device supports
//     cluster launch */ int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */ int
//     reserved2[2]; int          reserved1[1];               /**< Reserved for future use */ int          reserved[60];
//     /**< Reserved for future use */
// };
