#include <iostream>

using namespace std;

__global__ void kernel(void) {}

int main(void)
{
    kernel<<<1, 1>>>();
    cout << "Hello, World!" << endl;
    return 0;
}
