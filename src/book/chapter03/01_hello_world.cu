#include "../../include/logger.hpp"

int main(void)
{
    LOG_INFO("Hello,", " World!");
    LOG_WARNING("Hello,", " World!");
    LOG_ERROR("Hello,", " World!");

    return 0;
}

/*
 * [2025-09-03 20:43:37] [src/chapter03/01_hello_world.cu:5] ℹ️ Hello, World!
 * [2025-09-03 20:43:37] [src/chapter03/01_hello_world.cu:6] ⚠️ Hello, World!
 * [2025-09-03 20:43:37] [src/chapter03/01_hello_world.cu:7] ❌ Hello, World!
 */
