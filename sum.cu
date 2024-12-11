#include "freshman.h"
const int N = 4096 * 4096;
__global__ void sum(float* a, float* b){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    b[tid] = a[tid] * 3.14;
}
void sum_d(float* a, float* b,int n){
    for(int i = 0;i < n;i++){
        b[i] = a[i] * 3.14;
    }
}
int main(){
    float *a,*b;
    float *a_d,*b_d;
    double start;
    cudaMalloc(&a,N * sizeof(float));
    cudaMalloc(&b,N * sizeof(float));
    start = cpuSecond();
    sum<<<N/1024,1024>>>(a,b);
    cudaDeviceSynchronize();
    printf("%lf\n",100*(cpuSecond() - start));
    a_d = (float*)malloc(N * sizeof(float));
    b_d = (float*)malloc(N * sizeof(float));
    start = cpuSecond();

    sum_d(a_d,b_d,N);
    printf("%lf\n",cpuSecond() - start);   
}