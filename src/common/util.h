#define CHECK()                                                                                                \
    do {                                                                                                       \
        cudaError_t error = cudaGetLastError();                                                                \
        if (error != cudaSuccess) {                                                                            \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) \
                      << std::endl;                                                                            \
        }                                                                                                      \
    } while (0)

#define CHECK_CUDA_ERROR()                                                                                     \
    do {                                                                                                       \
        cudaError_t error = cudaGetLastError();                                                                \
        if (error != cudaSuccess) {                                                                            \
            std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) \
                      << std::endl;                                                                            \
        }                                                                                                      \
    } while (0)
