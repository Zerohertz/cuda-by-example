#include <iostream>

#include "../common/util.h"

using namespace std;

int main(void)
{
    cudaDeviceProp prop;
    int            dev;

    cudaGetDevice(&dev);
    cout << "ID of current CUDA device: " << dev << endl;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    cudaChooseDevice(&dev, &prop);
    cout << "ID of CUDA device closet to version 1.3: " << dev << endl;

    cudaSetDevice(dev);

    CHECK_CUDA_ERROR();
    return 0;
}

/*
 * ID of current CUDA device: 0
 * ID of CUDA device closet to version 1.3: 0
 */
