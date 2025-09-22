#include "../../include/handler.cuh"
#include "employees_salary.hpp"


__global__ void task(const double *salaries, double *new_salaries, int *size)
{
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    if (idx < *size) {
        new_salaries[idx] = salaries[idx] * (salaries[idx] * 15 / 100) + 5000;
    }
}

int main()
{
    const int size = sizeof(array_of_salaries) / sizeof(double);
    double   *dev_salaries;
    double   *dev_new_salaries;
    double    new_salaries[size];
    int      *dev_size;

    CUDA_CHECK(cudaMalloc((void **)&dev_salaries, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&dev_new_salaries, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&dev_size, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev_salaries, array_of_salaries, size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_salaries, array_of_salaries, size * sizeof(double), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blockPerGrid    = (size + threadsPerBlock - 1) / threadsPerBlock;

    task<<<blockPerGrid, threadsPerBlock>>>(dev_salaries, dev_new_salaries, dev_size);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&new_salaries, dev_new_salaries, size * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++) {
        LOG_INFO(new_salaries[i]);
    }

    CUDA_CHECK(cudaFree(dev_salaries));
    CUDA_CHECK(cudaFree(dev_new_salaries));
    CUDA_CHECK(cudaFree(dev_size));

    return 0;
}

/*
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:21] ✅ cudaMalloc((void **)&dev_salaries, size *
 * sizeof(double))
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:22] ✅ cudaMalloc((void **)&dev_new_salaries,
 * size * sizeof(double))
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:23] ✅ cudaMalloc((void **)&dev_size,
 * sizeof(int))
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:25] ✅ cudaMemcpy(dev_salaries,
 * array_of_salaries, size * sizeof(double), cudaMemcpyHostToDevice)
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:26] ✅ cudaMemcpy(dev_size, &size, sizeof(int),
 * cudaMemcpyHostToDevice)
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:27] ✅ cudaMemcpy(dev_salaries,
 * array_of_salaries, size * sizeof(double), cudaMemcpyHostToDevice)
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:34] ✅ cudaDeviceSynchronize()
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:35] ✅ cudaMemcpy(&new_salaries,
 * dev_new_salaries, size * sizeof(double), cudaMemcpyDeviceToHost)
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25966018358.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 37174016552.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26989942133.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31532517205.400002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33274883312.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33933589535.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33438140153.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 30642215093.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25444798174.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25996108280.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 36933064165.399994
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 30788150375.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32485669757.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33088209533.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25888065514.400002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33550983422.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34728834686.149994
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 29373199293.350002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34947726383.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26578577540.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 30776598683.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34028115389.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32700887288.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 29438152405.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 36389796027.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33652790144.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 24397759490.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32818233725.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32882955152.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 36048822935.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 37343119431.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33564463594.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26735017899.350002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 35633380341.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26909068310.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33243381473.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 36044263940.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33687755182.400002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26142059894.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 35000745290.149994
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 35802316940.149994
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26558628728.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31567464125.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25612744226.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32590742506.400002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 27307226336.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 29254503632.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32885483529.350002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31854371837.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 28614423754.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32760585735.350002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25822302269.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32179806046.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 36681619429.400002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 28185741560.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32202598340.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 35135936643.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 27785983403.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 28204601093.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 36812881760.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34637087933.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31672283162.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 30751871763.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 28574342864.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 28290003044.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34142092088.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 27891573341.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 24592578269.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31779485269.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31523027093.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33525307640.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34512501335.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33384478813.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 27722493411.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 28282056260.000000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34211830366.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26236702126.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 30471336255.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 24041663061.350002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 30077118941.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 27537729053.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26028839914.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25847949949.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 24811226542.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31032874011.350002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34505018888.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31608902660.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 32249319946.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 26942623845.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 27920946608.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34652659017.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 25583992118.149998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 33954425326.399998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 35593912119.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31774376300.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31652986724.150002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34881436929.349998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 24426079433.750000
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 34521423773.599998
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:38] ℹ️ 31601053409.600002
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:41] ✅ cudaFree(dev_salaries)
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:42] ✅ cudaFree(dev_new_salaries)
 * [2025-09-22 23:23:24] [src/lecture/chapter03/04_employees_salary.cu:43] ✅ cudaFree(dev_size)
 */
