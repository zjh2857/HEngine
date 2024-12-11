#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "freshman.h"
__global__ void genRandom(unsigned long long* a){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    a[tid] = tid;
}
int main(){
    unsigned long long N = 32768;
    unsigned long long q,psi,psiinv,ninv;
    unsigned int q_bit;
    getParams(q, psi, psiinv, ninv,q_bit, N);
    // void getParams(unsigned long long& q, unsigned long long& psi, unsigned long long& psiinv, unsigned long long& ninv, unsigned int& q_bit, unsigned long long n)

    unsigned long long* psiTable;
    unsigned long long* psiinvTable;
    unsigned long long mu;

    cudaMalloc(&psiTable, N * sizeof(unsigned long long));
    cudaMalloc(&psiinvTable, N * sizeof(unsigned long long));

    
    fillTablePsi128<<<N/1024,1024>>>(psi, q, psiinv, psiTable, psiinvTable, log2(N));
    uint128_t mu1 = uint128_t::exp2(q_bit * 2);
    mu = (mu1 / q).low;

    unsigned long long* ntt_in;
    cudaStream_t ntt = 0;
    Check(cudaMalloc((void**)&ntt_in, N * sizeof(unsigned long long)));
    genRandom<<<N/1024,1024>>>(ntt_in);
    double start = cpuSecond();
    for(int i = 0; i < 100000; i++){
        forwardNTT(ntt_in,N,ntt,q,mu,q_bit,psiTable);
    }
    cudaDeviceSynchronize();
    printf("Time :%lf\n",cpuSecond()-start);
}