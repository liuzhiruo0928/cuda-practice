#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000

__global__ void vectorAddGridStride(float *d_A, float *d_B, float *d_C, int n)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 🔥 核心：grid-stride loop
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        d_C[i] = d_A[i] + d_B[i];
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
    int blocksPerGrid = 4;  // 🔥 可以随便设，不再依赖 N

    vectorAddGridStride<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

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