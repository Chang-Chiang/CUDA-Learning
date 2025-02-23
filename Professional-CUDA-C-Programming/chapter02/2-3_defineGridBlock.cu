/**
 * @file 2-3_defineGridBlock.cu
 * @author Chang Chiang (Chang_Chiang@outlook.com)
 * @brief Demonstrate defining the dimensions of a block of threads and a grid of blocks from the host.
 *
 * nvcc 2-3_defineGridBlock.cu -o defineGridBlock
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 1024;

    // define grid and block structure
    dim3 block (1024);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 512;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset device before you leave
    cudaDeviceReset();

    return(0);
}
