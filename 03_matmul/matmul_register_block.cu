#include<stdio.h>
#include<cuda_runtime.h>

#define N 512
#define TILE_SIZE 16
#define BLOCK_SIZE 16

__global__ void matmul_register_block(float* A, float* B, float* C, int n){

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //global row和col - 每个thread 负责2x2
    int row = blockIdx.y * TILE_SIZE + ty * 2;
    int col = blockIdx.x * TILE_SIZE + tx * 2;

    //register blocking，一个thread算2x2
    float c00 = 0, c01 = 0;
    float c10 = 0, c11 = 0;

    //tiling 分区
    for ( int t = 0; t < N; t += TILE_SIZE){

        //load A
        As[ty*2][tx] = A[row * n + (t+tx)];
        As[ty*2 + 1][tx] = A[(row+1)*n + (t+tx)];

        As[ty*2][tx + 8]     = A[row * n + (t + tx + 8)];
        As[ty*2+1][tx + 8]   = A[(row+1) * n + (t + tx + 8)];

        //load B
        Bs[ty][tx*2] = B[(t+ty)*n + col];
        Bs[ty][tx*2 + 1] = B[(t+ty)*N + col + 1];

        Bs[ty + 8][tx*2]     = B[(t+ty+8)*n + col];
        Bs[ty + 8][tx*2 + 1] = B[(t+ty+8)*n + col + 1];

        __syncthreads();

        //计算
        for(int k = 0; k < TILE_SIZE; k++){
            float a0 = As[ty*2][k];
            float a1 = As[ty*2 +1][k];

            float b0 = Bs[k][tx*2];
            float b1 = Bs[k][tx*2 +1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();

    }
    C[row * n + col] = c00;
    C[row * n + col + 1] = c01;
    C[(row + 1)* N + col] = c10;
    C[(row + 1)* N + col + 1] = c11;

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


int main() {

    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);

    float *d_A, *d_B, *d_C;

    // 初始化
    for(int i = 0; i < N * N; i++){
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(8, 8);   // 每个thread算2x2 → 16x16 tile
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    matmul_register_block<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU验证
    matmul_cpu(h_A, h_B, h_ref, N);

    // 检查前几个值
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