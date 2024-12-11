#include <stdio.h>
__global__ void shuffle(){
    int tid = threadIdx.x;
    
    int get = __shfl(tid,2,8);
    printf("%d,%d\n",tid,get);
}

int main(){
    shuffle<<<1,32>>>();
    cudaDeviceSynchronize();
}