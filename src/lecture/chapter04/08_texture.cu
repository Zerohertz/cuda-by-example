#include "../../include/handler.cuh"

#define N 1024

__global__ void kernel(cudaTextureObject_t tex)
{
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x   = tex1Dfetch<float>(tex, tid);
}

void call_kernel(cudaTextureObject_t tex) { kernel<<<1, 256>>>(tex); }

int main()
{
    float *buffer;
    float  value = 023323.0f;
    buffer       = &value;
    CUDA_CHECK(cudaMalloc(&buffer, N * sizeof(float)));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType                = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr      = buffer;
    resDesc.res.linear.desc.f      = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x      = 32;
    resDesc.res.linear.sizeInBytes = N * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    call_kernel(tex);

    CUDA_CHECK(cudaDestroyTextureObject(tex));
    CUDA_CHECK(cudaFree(buffer));

    return 0;
}

/*
 * [2025-09-23 00:27:40] [src/lecture/chapter04/08_texture.cu:18] ✅ cudaMalloc(&buffer, N * sizeof(float))
 * [2025-09-23 00:27:40] [src/lecture/chapter04/08_texture.cu:33] ✅ cudaCreateTextureObject(&tex, &resDesc, &texDesc,
 * NULL)
 * [2025-09-23 00:27:40] [src/lecture/chapter04/08_texture.cu:37] ✅ cudaDestroyTextureObject(tex)
 * [2025-09-23 00:27:40] [src/lecture/chapter04/08_texture.cu:38] ✅ cudaFree(buffer)
 */
