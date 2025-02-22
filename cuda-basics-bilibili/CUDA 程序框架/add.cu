// nvcc -o add add.cu -arch=compute_60 -code=sm_60
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *x, const double *y, double *z);

void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;

    // 主机分配内存
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    // 初始化数组
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    // 设备分配内存, GPU 显存
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

/**
 * @brief 核函数, GPU 上运行
 *
 * @param x 数组 x 的地址
 * @param y 数组 y 的地址
 * @param z 数组 z 的地址
 */
void __global__ add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

/**
 * @brief 校验数组 z 每一个元素值是否等于 c
 *
 * @param z 数组 z 的地址
 * @param N 数组 z 的长度
 */
void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

