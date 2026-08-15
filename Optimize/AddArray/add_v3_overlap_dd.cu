#include "../common/error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;

const int N = 100000000;        // 总数据量
const int NUM_STREAMS = 10;     // 使用的流数量, GTX1060 10 个 SM
const int N1 = N / NUM_STREAMS; // 每个流处理的数据量
const int M = sizeof(real) * N;

const int THREAD_PER_BLOCK = 128;
const int BLOCK_NUM = (N1 - 1) / THREAD_PER_BLOCK + 1;
cudaStream_t streams[NUM_STREAMS];

void timing(const real *d_x, const real *d_y, real *d_z, const int num);

int main(void)
{
    real *h_x = (real*) malloc(M);
    real *h_y = (real*) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    for (int n = 0 ; n < NUM_STREAMS; ++n)
    {
        CHECK(cudaStreamCreate(&(streams[n])));
    }

    timing(d_x, d_y, d_z, NUM_STREAMS);

    for (int n = 0 ; n < NUM_STREAMS; ++n)
    {
        CHECK(cudaStreamDestroy(streams[n]));
    }

    free(h_x);
    free(h_y);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const real *d_x, const real *d_y, real *d_z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N1)
    {
        d_z[n] = d_x[n] + d_y[n];
    }
}

void timing(const real *d_x, const real *d_y, real *d_z, const int num)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        for (int n = 0; n < num; ++n)
        {
            int offset = n * N1;
            add<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, streams[n]>>>
            (d_x + offset, d_y + offset, d_z + offset);
        }
 
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}


