#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024

__inline__ __device__
float warpReduceSum(float val){
    for(int offset = 16; offset > 0 ; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void  reduction_opt(float *d_A, float *d_out){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    float sum = 0.0f;

    for( int i = idx; i < N; i += blockDim.x * gridDim.x){
        sum += d_A[i];
    }

    sum = warpReduceSum(sum);

    __shared__ float s_data[32];

    int lane = tid % 32;
    int warpId = tid / 32;

    if(lane == 0){
        s_data[warpId] = sum;
    }
    __syncthreads();

    if(warpId == 0){
        sum = (tid < blockDim.x / 32) ? s_data[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    if(tid == 0){
        d_out[blockIdx.x]= sum;
    }
}

int main(){
    int size = N * sizeof(float);
    float *h_A = (float*) malloc (size);
    for (int i = 0; i < N; i++){
        h_A[i] = 1.0f;
    }


    float *d_A, *d_out;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)/ threadsPerBlock;


    cudaMalloc(&d_A, size);
    cudaMalloc(&d_out, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_A,h_A,size, cudaMemcpyHostToDevice);

    reduction_opt<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_out);

    float *h_out = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_out,d_out,blocksPerGrid *  sizeof(float),cudaMemcpyDeviceToHost);

    float final_sum = 0.0f;

    for(int i = 0; i < blocksPerGrid; i++){
        final_sum += h_out[i];
    }

    printf("sum = %f\n", final_sum);

    cudaFree(d_A);
    cudaFree(d_out);

    free(h_A);
    free(h_out);

    return 0;
}