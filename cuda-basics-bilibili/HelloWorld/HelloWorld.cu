/**
 * @file HelloWorld.cu
 * @author Chang Chiang (Chang_Chiang@outlook.com)
 * @brief GPU 上 16 个线程执行 HelloWorld kernel 函数
 * 编译: nvcc -o HelloWorld HelloWorld.cu
 * 执行: ./HelloWorld
 * 说明: .cu 文件结尾, 默认包含 cuda 相关头文件
 * @version 0.1
 * @date 2025-02-06
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the the GPU!\n");
}


int main(void)
{
    hello_from_gpu<<<4, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}
