#include <cuda.h>
#include <random>

#define THREAD_PER_BLOCK 256

__global__ void reduce(float *d_in, float *d_out)
{
    float *block_base = d_in + blockIdx.x * blockDim.x;

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x % (2 * i) == 0)
        {
            block_base[threadIdx.x] += block_base[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = block_base[0];
    }
}

bool check(float *h_out, float *h_res, int n)
{
  for (int i = 0; i < n; i++)
  {
    if (fabs(h_out[i] - h_res[i]) > 1e-3)
    {
        return false;
    }
  }
  return true;
}

int main()
{
    // 参数定义
    const int ARR_LEN = 32 * 1024 * 1024;
    const int ARR_SIZE = ARR_LEN * sizeof(float);
    const int BLOCK_NUM = ARR_LEN / THREAD_PER_BLOCK;
    const int OUT_SIZE = BLOCK_NUM * sizeof(float);

    // host 内存分配
    float *h_a   = (float *)malloc(ARR_SIZE);
    float *h_out = (float *)malloc(OUT_SIZE);
    float *h_res = (float *)malloc(OUT_SIZE);

    // device 内存分配
    float *d_a, *d_out;
    cudaMalloc((void **)&d_a, ARR_SIZE);
    cudaMalloc((void **)&d_out, OUT_SIZE);
    
    // 初始化数据
    std::mt19937 rng(123456);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < ARR_LEN; i++)
    {
        h_a[i] = dist(rng);
    }

    for (int i = 0; i < BLOCK_NUM; i++)
    {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += h_a[i * THREAD_PER_BLOCK + j];
        }
        h_res[i] = cur;
    }

    cudaMemcpy(d_a, h_a, ARR_SIZE, cudaMemcpyHostToDevice);

    dim3 Grid(BLOCK_NUM, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce<<<Grid, Block>>>(d_a, d_out);

    cudaMemcpy(h_out, d_out, OUT_SIZE, cudaMemcpyDeviceToHost);

    if (check(h_out, h_res, BLOCK_NUM))
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < BLOCK_NUM; i++)
        {
            printf("h_out: %lf,\nh_res: %lf\n\n", h_out[i], h_res[i]);
        }
    }

    cudaFree(d_a);
    cudaFree(d_out);

    free(h_a);
    free(h_out);
    free(h_res);
}
