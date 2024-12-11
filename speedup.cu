#include "freshman.h"

__global__ void gpu(unsigned* ptr){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j = 0; j < 64; j++)ptr[i] = (ptr[i] + 3777)*1919%893;
}

int main(){
    
    unsigned* ptr = (unsigned*)malloc(1024*1024*1024*sizeof(unsigned));
    double start;
    // start = cpuSecond();
    // for(int i = 0; i < 1024*1024*1024; i++){
    //     for(int j = 0; j < 2; j++)ptr[i] = (ptr[i] + 3777)*1919%893;
    // }
    // printf("Time: %lf\n",cpuSecond() - start);
    cudaMalloc(&ptr,1024*1024*1024*sizeof(unsigned));
    start = cpuSecond();
    gpu<<<1024*1024*1024,1024>>>(ptr);
    cudaDeviceSynchronize();
    printf("Time: %lf\n",cpuSecond() - start);
    
}