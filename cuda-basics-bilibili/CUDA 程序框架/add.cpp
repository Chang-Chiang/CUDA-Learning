#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;  // ≈ 2^30 / 10 = 1G Byte / 10

    // 64 位机器, double 大小 8 字节
    const int M = sizeof(double) * N;

    // 手动为变量分配内存, 在堆上
    // 分配内存大小约为: 3 * 1GB / 10 * 8 = 2.4GB
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    add(x, y, z, N);  // for 循环遍历 10^8 次
    check(z, N);      // for 循环遍历 10^8 次

    // 释放内存
    free(x);
    free(y);
    free(z);

    return 0;
}

void add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        // 判断浮点数相等不能用 ==
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

