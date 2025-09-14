
## 测试环境

```bash
$ nvidia-smi
Sun Sep 14 14:00:22 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 581.15       CUDA Version: 13.0     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1060        On  | 00000000:01:00.0 Off |                  N/A |
| N/A   42C    P8               2W /  78W |    837MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A        40      G   /Xwayland                                 N/A      |
+---------------------------------------------------------------------------------------+

```

- GTX1060 相关信息

| 指标                      | GTX 1060 (6 GB GDDR5)     |
| ----------------------- | ------------------------- |
| 架构                      | Pascal GP106              |
| CUDA Compute Capability | 6.1 (`-arch=sm_61`)       |
| 显存容量                    | 6 GB GDDR5                |
| 显存位宽                    | 192-bit                   |
| 显存等效速率                  | 8 Gbps                    |
| 显存带宽                    | 192 GB/s                  |
| 单精度峰值 (FP32)            | ~4.4 TFLOPS               |
| 半精度峰值 (FP16)            | ~4.4 TFLOPS (1:1 于 FP32)  |
| 双精度峰值 (FP64)            | ~137 GFLOPS (1/32 于 FP32) |


-  GTX1060 内存组织

| 属性                           | 数值                         |
| ---------------------------- | -------------------------- |
| Device ID                    | 0                          |
| Device name                  | NVIDIA GeForce GTX 1060    |
| Compute capability           | 6.1                        |
| Global memory                | 5.99988 GB                 |
| Constant memory              | 64 KB                      |
| Maximum grid size (x, y, z)  | 2147483647 × 65535 × 65535 |
| Maximum block size (x, y, z) | 1024 × 1024 × 64           |
| Number of SMs                | 10                         |
| Shared memory per block      | 48 KB                      |
| Shared memory per SM         | 96 KB                      |
| Registers per block          | 64 K                       |
| Registers per SM             | 64 K                       |
| Threads per block            | 1024                       |
| Threads per SM               | 2048                       |

- 编译命令: `nvcc -O3 -arch=sm_61 xxx.cu`

## 数组加

1. 首先分别统计 cpu 和 gpu 下耗时
