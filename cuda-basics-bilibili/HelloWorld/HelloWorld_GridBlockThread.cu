/**
 * @file HelloWorld_GridBlockThread.cu
 * @author Chang Chiang (Chang_Chiang@outlook.com)
 * @brief grid-block-thread 三维模型
 * 编译: nvcc -o HelloWorld_GridBlockThread HelloWorld_GridBlockThread.cu
 * 执行: ./HelloWorld_GridBlockThread
 * 说明: 一个 kernel 函数对应一个 grid,
 *       grid 由大量 block 组成,
 *       block 由大量 thread 构建
 *
 * 一维线程 id 索引计算:
 *     kernel_fun<<<2, 4>>>,
 *     gridDim.x = 2, blockDim.x = 4
 *     blockIdx.x: [0, 1], threadIdx.x: [0, 3]
 *     线程唯一标识 id [0, 7]: id = threadIdx.x + blockIdx.x * blockDim.x
 * 二维线程 id 索引计算:
 *     dim3 grid_size(2, 2)
 *     dim3 block_size(4, 4)
 *     kernel_fun<<<grid_size,block_size>>>,
 *     gridDim.x = 2, gridDim.y = 2,
 *     blockDim.x = 4, blockDim.y = 4,
 *     blockIdx.x: [0, 1], blockIdx.y: [0, 1],
 *     threadIdx.x: [0, 3], threadIdx.y: [0, 3]
 *     线程索引 tid [0, 15]: tid = threadIdx.x + threadIdx.y * blockDim.x
 *     线程块索引 bid [0, 3]: bid = blockIdx.x + blockIdx.y * gridDim.x
 *     线程唯一标识 id [0, 63]: id = tid + bid * ( blockDim.y * blockDim.x)
 * 三维线程 id 索引计算:
 *     线程索引 tid: tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x
 *     线程块索引 bid: bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.z
 *     线程唯一标识 id: id = tid + bid * (blockDim.z * blockDim.y * blockDim.x)
 * @version 0.1
 * @date 2025-02-06
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <stdio.h>

__global__ void hello_from_gpu()
{
    // printf("GridDim,  x: %d, y: %d, z: %d\n", gridDim.x, gridDim.y, gridDim.z);
    // printf("BlockDim, x: %d, y: %d, z: %d\n", blockDim.x, blockDim.y, blockDim.z);

    printf("Hello World from block-(%d, %d, %d) and thread-(%d, %d, %d)!\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void)
{
    const dim3 grid_size(2, 2, 2);   // grid 内的 block 维度, 即 grid 内的 block 数量
    const dim3 block_size(2, 2, 2);  // block 内的 thread 维度, 即 block 内的 thread 数量

    // hello_from_gpu<<<2, 4>>>();
    hello_from_gpu<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}

