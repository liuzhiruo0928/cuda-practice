#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024

__global__ void reduction(float *d_A, float *d_out){
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    for(int i = idx; i < N; i += blockDim.x * gridDim.x ){
        sum += d_A[i];
    }

    s_data[tid] = sum;

    __syncthreads();

    //tree reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            s_data[tid] += s_data[tid + stride];
        }

        __syncthreads();
    }

    if(tid == 0){
        d_out[blockIdx.x] = s_data[0];
    }
}

int main(){
    float h_A[N];
    float *d_A, *d_out;
    float result;
    int size = N * sizeof(float);


    for(int i = 0; i < N; i++){
        h_A[i] = 1.0f;
    }
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    reduction<<<1, 256>>>(d_A,d_out);
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %f\n", result);

    cudaFree(d_A);
    cudaFree(d_out);

    return 0;

}