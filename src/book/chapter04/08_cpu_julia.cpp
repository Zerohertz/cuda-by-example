#include "../../include/cpu_bitmap.hpp"

#define DIM 1000

struct cuComplex
{
    float r;
    float i;
    cuComplex(float a, float b)
        : r(a)
        , i(b)
    {
    }
    float     magnitude2(void) { return r * r + i * i; }
    cuComplex operator*(const cuComplex &a) { return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i); }
    cuComplex operator+(const cuComplex &a) { return cuComplex(r + a.r, i + a.i); }
};

int julia(int x, int y)
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

void kernel(unsigned char *ptr)
{
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue      = julia(x, y);
            ptr[offset * 4 + 0] = 255 * juliaValue;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

int main(void)
{
    CPUBitmap      bitmap(DIM, DIM, nullptr, __FILE__);
    unsigned char *ptr = bitmap.get_ptr();

    kernel(ptr);

    bitmap.save_to_png();

    return 0;
}
