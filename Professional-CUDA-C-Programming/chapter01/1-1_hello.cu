/**
 * @file 1-1 hello.cu
 * @author Chang Chiang (Chang_Chiang@outlook.com)
 * @brief GPU 多线程输出 "Hello World from GPU!"
 * nvcc 1-1_hello.cu -o hello
 * @version 0.1
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
