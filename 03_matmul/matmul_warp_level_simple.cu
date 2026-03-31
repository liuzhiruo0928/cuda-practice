#include <stdio.h>
#include <cuda_runtime.h>

#define N 256
#define TILE_SIZE 16

__global__ void matmul_warp(float* A, float* B, float* C, int n) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;

    int warpId = tid / 32;
    int lane   = tid % 32;

    // 每个block 16x16 → 分成 4 个 warp（2x2）
    int warp_row = warpId / 2;
    int warp_col = warpId % 2;

    // warp 负责 8x8 tile
    int row = blockIdx.y * TILE_SIZE + warp_row * 8 + lane / 8;
    int col = blockIdx.x * TILE_SIZE + warp_col * 8 + lane % 8;

    float sum = 0.0f;

    for (int t = 0; t < n; t += TILE_SIZE) {

        // load shared memory
        As[ty][tx] = A[(blockIdx.y * TILE_SIZE + ty) * n + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * n + (blockIdx.x * TILE_SIZE + tx)];

        __syncthreads();

        // warp-level compute
        for (int k = 0; k < TILE_SIZE; k++) {
            float a = As[warp_row * 8 + (lane / 8)][k];
            float b = Bs[k][warp_col * 8 + (lane % 8)];
            sum += a * b;
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

void matmul_cpu(float* A, float* B, float* C, int n) {
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float sum = 0.0f;
            for(int k = 0; k < n; k++){
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}


int main(){

    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);

    float *d_A, *d_B, *d_C;

    for(int i = 0; i < N*N; i++){
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);   // 256 threads = 8 warps
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    matmul_warp<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU 验证
    matmul_cpu(h_A, h_B, h_ref, N);

    // 打印前几个
    for(int i = 0; i < 5; i++){
        printf("GPU: %f | CPU: %f\n", h_C[i], h_ref[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}