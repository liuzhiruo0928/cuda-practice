#include <stdio.h>
#include <cuda_runtime.h>
#define TILE 16
#define N 100

__global__ void matmul_tiled(float *A, float *B, float *C, int n){
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0;

    //支持非整除
    int numTiles = (n + TILE - 1) / TILE;

    for(int t = 0; t < numTiles; t++){

        //A矩阵是行访问[row][k]- row不变 k变化
        int A_col = t * TILE + threadIdx.x;
        if( row < n && A_col < n){
            s_A[threadIdx.y][threadIdx.x] = A[row * n + A_col];
        }else{
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //B矩阵是列访问[k][col]- col不变 k变化
        int B_row = t * TILE + threadIdx.y;
        if( B_row < n && col < n){
            s_B[threadIdx.y][threadIdx.x] = B[B_row * n + col];

        }else{
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        //求和
        for( int k = 0; k < TILE; k++){
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    //写回保护
    if(row < n && col < n){
        C[row * n + col] = sum;
    }


}

int main(){
    int size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = ( float * )malloc(size);
    h_B = ( float * )malloc(size);
    h_C = ( float * )malloc(size);

    for(int i = 0; i < N*N; i++){
        h_A[i]=1.0f;
        h_B[i]=1.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((N+ TILE -1)/TILE, (N+ TILE -1)/TILE);

    matmul_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for(int j = 0; j < N*N; j++){
        printf("%f ",h_C[j]);
        if((j+1)%N == 0){
            printf("\n");
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}