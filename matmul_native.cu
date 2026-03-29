#include <stdio.h>
#include <cuda_runtime.h>
#define N 4

__global__ void matmul(float *A, float *B, float *C, int n){
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;
    for(int k = 0; k < n; k++){
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

int main(){
    int size = N * N * sizeof(float);

    float A[N*N], B[N*N], C[N*N];
    for(int i = 0; i < N*N; i++){
        A[i] = 1.0f; // Initialize A with 1s
        B[i] = 1.0f; // Initialize B with 1s
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    //dim3是用来描述线程布局的三维结构
    dim3 threads(N,N);
    matmul<<<1, threads>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    for(int i = 0; i < N*N; i++){
        printf("%f ",C[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}