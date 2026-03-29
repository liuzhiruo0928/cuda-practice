#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000

__global__ void vectorAddShared(float *d_A, float *d_B, float *d_C, int n)
{
    // shared memory（block 内）
    __shared__ float s_A[256];
    __shared__ float s_B[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1：global → shared
    if (idx < n)
    {
        s_A[tid] = d_A[idx];
        s_B[tid] = d_B[idx];
    }

    // Step 2：同步（关键🔥）
    __syncthreads();

    // Step 3：计算（用 shared memory）
    if (idx < n)
    {
        d_C[idx] = s_A[tid] + s_B[tid];
    }
}

int main()
{
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 初始化
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证
    for (int i = 0; i < 10; i++)
    {
        printf("%f\n", h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}