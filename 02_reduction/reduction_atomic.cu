#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024

__inline__ __device__
float warpReduceSum(float val){
    for(int offset = warpSize / 2; offset > 0; offset /=2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduction_atomic(float *d_A, float *d_out, int n){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;

    //grid-stride loop
    for(int i = idx; i < n; i += blockDim.x * gridDim.x){
        sum += d_A[i];
    }

    //在warp内求和
    sum = warpReduceSum(sum);

    //取warp内的位置-tid的后五位-31- 0001 1111
    int lane = tid & 31;

    if(lane == 0){
        atomicAdd(d_out, sum);
    }
}

int main(){
    int size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    for (int i = 0; i < N; i++){
        h_A[i] = 1.0f;
    }

    float * d_A, *d_out;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemset(d_out, 0, sizeof(float));
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = 4;

    reduction_atomic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_out, N);

    float result;

    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum = %f\n", result);

    cudaFree(d_A);
    cudaFree(d_out);
    free(h_A);

    return 0;



}