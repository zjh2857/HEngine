#pragma once
#include "uint128.cuh"
#include "rottable.cuh"
#define BASISSIZE 128
__constant__ unsigned long long q_const[BASISSIZE];
__constant__ unsigned long long mu_const[BASISSIZE];
__constant__ unsigned long long qbit_const[BASISSIZE];

extern int* rot1;
extern int* rot1r;

// __global__ void polymul(unsigned long long* a,unsigned long long* b,unsigned long long* c,int N,unsigned long long q){
//     int tid = blockDim.x * blockIdx.x +threadIdx.x;
//     a[tid] %= q;
//     b[tid] %= q;
//     c[tid] = (a[tid] * b[tid]) % q;
// }

//TODO 64 bits can't use a * b % q
__global__ void polyadd(unsigned long long* a,unsigned long long* b,unsigned long long* c,int N,unsigned long long q){
    int tid = blockDim.x * blockIdx.x +threadIdx.x;
    if(a != 0)c[tid] = (a[tid] + b[tid]) % q;
    else c[tid] = b[tid];
}

__global__ void polyminus(unsigned long long* a,unsigned long long* b,unsigned long long* c,int N,unsigned long long q){
    int tid = blockDim.x * blockIdx.x +threadIdx.x;
    if(a != 0)c[tid] = (a[tid] +  q - b[tid] ) % q;
    else c[tid] = (q - b[tid] ) % q;
}


__global__ void polyminussingle(unsigned long long* a,unsigned long long* b,unsigned long long* c){
    int tid = blockDim.x * blockIdx.x +threadIdx.x;
    int idx = blockIdx.y;
    c[tid + idx * blockDim.x * gridDim.x] = (a[tid + idx * blockDim.x * gridDim.x] +  q_const[idx] - b[tid + idx * blockDim.x * gridDim.x] ) % q_const[idx];
}
// __global__ void polymuladd(unsigned long long* a,unsigned long long* b,unsigned long long* c,unsigned long long* d,int N,unsigned long long q){
//     int tid = blockDim.x * blockIdx.x +threadIdx.x;
//     unsigned long long t = (a[tid] * b[tid]) % q;
//     d[tid] = (t + c[tid]) % q;
// }
__global__ void polymul(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long q, unsigned long long mu, int qbit)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    if (rc.low < q)
        c[i] = rc.low;
    else
        c[i] = rc.low - q;
    // if(i < 10){
    //     printf("%lld*%lld %% %lld==%lld\n",ra,rb,q,c[i]);
    // }
}

__global__ void polymulInteger(unsigned long long a[],unsigned long long P,unsigned long long q, unsigned long long mu, int qbit)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = P;

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    if (rc.low < q)
        a[i] = rc.low;
    else
        a[i] = rc.low - q;
    // if(i < 10){
    //     printf("%lld*%lld %% %lld==%lld\n",ra,rb,q,c[i]);
    // }
}

__global__ void polymuladd(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],unsigned long long q, unsigned long long mu, int qbit)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);
    d[i] = (rc.low + c[i])%q;

}

__global__ void polymulminus(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],unsigned long long q, unsigned long long mu, int qbit)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    register unsigned long long ra = a[i];
    register unsigned long long rb = b[i];

    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);
    d[i] = (rc.low + q - c[i])%q;

}

__global__ void cudaRescale(unsigned long long *a, unsigned long long *b,unsigned long long q,unsigned long long mu,unsigned long long qbit,unsigned long long qinv){
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    register unsigned long long remainder = b[i] % q;
    register unsigned long long ra = (a[i] + q - remainder) % q;
    register unsigned long long rb = qinv;
    // register unsigned long long rc = qinv;
    // if(i == 0){
    //     printf("%lld\n",remainder);
    // }
    uint128_t rc, rx;

    mul64(ra, rb, rc);

    rx = rc >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(rc, rx);

    if (rc.low < q)
        a[i] = rc.low;
    else
        a[i] = rc.low - q;
}

__global__ void cudaRescale_fusion(unsigned long long *a, unsigned long long *b,int depth,int size,int N,unsigned long long* spinv){
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = blockIdx.y;
    if(idx <= depth)return;
    unsigned long long q = q_const[idx];
    unsigned long long mu = mu_const[idx];
    unsigned long long qbit = qbit_const[idx];

    register unsigned long long remainder = b[depth * N + i] ;
    register unsigned long long ra = (a[idx * N + i] + q - remainder) ;
    register unsigned long long rb = spinv[depth*size+idx];
    uint128_t rc, rx;
    mul64(ra, rb, rc);
    rx = rc >> (qbit - 2);
    mul64(rx.low, mu, rx);
    uint128_t::shiftr(rx, qbit + 2);
    mul64(rx.low, q, rx);
    sub128(rc, rx);

    if (rc.low < q)
        a[idx * N + i] = rc.low;
    else
        a[idx * N + i] = rc.low - q;

}


__global__ void cudaRescale_fusion_opt(unsigned long long *a, unsigned long long *b,int depth,int size,int N,unsigned long long* spinv){
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = blockIdx.y;
    if(idx <= depth)return;
    unsigned long long q = q_const[idx];
    unsigned long long mu = mu_const[idx];
    unsigned long long qbit = qbit_const[idx];

    register unsigned long long remainder = b[idx * N + i] ;
    register unsigned long long ra = (a[idx * N + i] + q - remainder);
    register unsigned long long rb = spinv[depth*size+idx];
    uint128_t rc, rx;
    mul64(ra, rb, rc);
    rx = rc >> (qbit - 2);
    mul64(rx.low, mu, rx);
    uint128_t::shiftr(rx, qbit + 2);
    mul64(rx.low, q, rx);
    sub128(rc, rx);

    if (rc.low < q)
        a[idx * N + i] = rc.low;
    else
        a[idx * N + i] = rc.low - q;

}
__global__ void cudaRescale_fusion_opt(unsigned long long *a1, unsigned long long *b1,unsigned long long *a2, unsigned long long *b2,int depth,int size,int N,unsigned long long* spinv){
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = blockIdx.y;
    if(idx <= depth)return;
    unsigned long long q = q_const[idx];
    unsigned long long mu = mu_const[idx];
    unsigned long long qbit = qbit_const[idx];

    register unsigned long long remainder = b1[idx * N + i] ;
    register unsigned long long ra = (a1[idx * N + i] + q - remainder);
    register unsigned long long rb = spinv[depth*size+idx];
    uint128_t rc, rx;
    mul64(ra, rb, rc);
    rx = rc >> (qbit - 2);
    mul64(rx.low, mu, rx);
    uint128_t::shiftr(rx, qbit + 2);
    mul64(rx.low, q, rx);
    sub128(rc, rx);

    if (rc.low < q)
        a1[idx * N + i] = rc.low;
    else
        a1[idx * N + i] = rc.low - q;

    remainder = b2[idx * N + i] ;
    ra = (a2[idx * N + i] + q - remainder);
    rb = spinv[depth*size+idx];
    // uint128_t rc, rx;
    mul64(ra, rb, rc);
    rx = rc >> (qbit - 2);
    mul64(rx.low, mu, rx);
    uint128_t::shiftr(rx, qbit + 2);
    mul64(rx.low, q, rx);
    sub128(rc, rx);

    if (rc.low < q)
        a2[idx * N + i] = rc.low;
    else
        a2[idx * N + i] = rc.low - q;
}
//TODO
//a != buff
__global__ void cudarotation(unsigned long long* a, unsigned long long* buff,unsigned long long q,unsigned long long galois_elt,unsigned long long n){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long index_raw = tid * galois_elt;
    unsigned long long index = index_raw & (n-1);
    unsigned long long rval = a[tid];
    // buff[tid] = a[tid];
    // return ;
    if((index_raw / n) & 1){
        rval = (q - rval) % q;
    }
    buff[index] = rval;
}
__global__ void cudaConvUp(int N,unsigned long long* a,int size,unsigned long long* q,unsigned long long* Qmod,unsigned long long* q_hat_inv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++){
        unsigned long long acc = 0;
        for(int j = 0; j < size;j++){
            unsigned long long temp = a[j*N+tid];
            temp *= q_hat_inv[j];
            temp %= q[j];
            temp *= Qmod[j+i*size];
            temp %= q[i+size];
            acc += temp;
        }   
        acc %= q[i+size];
        a[i*N + size * N + tid] = acc;
    }
}

// __global__ void cudaConvUp_dcomp(int N,unsigned long long* a,int size,unsigned long long* q,unsigned long long* Qmod,unsigned long long* q_hat_inv){
//     unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
//     for(int i = 0; i < 1; i++){
//         unsigned long long acc = 0;
//         for(int j = 0; j < size;j++){
//             unsigned long long temp = a[j*N+tid];
//             temp *= q_hat_inv[j];
//             temp %= q[j];
//             temp *= Qmod[j+i*size];
//             temp %= q[i+size];
//             acc += temp;
//         }   
//         acc %= q[i+size];
//         a[i*N + size * N + tid] = acc;
//     }
// }

__global__ void cudaConvUp_dcomp(int N,unsigned long long* a,int source,int size,unsigned long long* q,unsigned long long* Qmod,unsigned long long* q_hat_inv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size+1; i++){
        a[i*N + tid] = a[source*N + tid];
    }
}

__global__ void cudaConvUp_dcomp_batch(int N,unsigned long long* a,int size,unsigned long long* q,unsigned long long* Qmod,unsigned long long* q_hat_inv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j = 0; j < size; j++){
        for(int i = 0; i < size+1; i++){
            a[j * (size+1) * N + i*N + tid] = a[j * (size+1) * N + j*N + tid];
        }
    }
}

__global__ void cudaConvUp_dcomp_batch_batch(int N,unsigned long long* a,unsigned long long *b,int size,unsigned long long* q,unsigned long long* Qmod,unsigned long long* q_hat_inv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    register unsigned long long source = b[tid];
    int j = tid / N;
    int k = tid % N;
    for(int i = 0; i < size + 1; i++){
        a[j * (size + 1) * N + i * N + k] = source;
    }
}

__global__ void cudaConvDown(int N,unsigned long long* a,unsigned long long* b,int size,unsigned long long* q,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++){
        unsigned long long acc = 0;
        for(int j = 0; j < size;j++){
            unsigned long long temp = b[j*N+tid+size * N];
            temp *= p_hat_inv[j];
            temp %= q[j+size];
            temp *= Pmod[j+i*size];
            temp %= q[i];
            acc += temp;
        }   
        acc %= q[i];
        a[i*N + tid] = (b[i*N+tid] + q[i] - acc)*Pinv[i]%q[i];
    }
}



__global__ void cudaConvDown_dcomp(int N,unsigned long long* a,unsigned long long* b,int size,unsigned long long* q,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv,unsigned long long *spinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++){
        unsigned long long q = q_const[i];
        unsigned long long mu = mu_const[i];
        unsigned long long qbit = qbit_const[i];
        unsigned long long acc = b[size * N + tid] % q;

        register unsigned long long ra = (b[i*N+tid] + q - acc) % q ;
        register unsigned long long rb = spinv[i];
        uint128_t rc, rx;
        mul64(ra, rb, rc);
        rx = rc >> (qbit - 2);
        mul64(rx.low, mu, rx);
        uint128_t::shiftr(rx, qbit + 2);
        mul64(rx.low, q, rx);
        sub128(rc, rx);

        if (rc.low < q)
            a[i*N + tid] = rc.low;
        else
            a[i*N + tid] = rc.low - q;
        // if(tid == 0){
        //     printf("%llu,%llu,%llu,%llu\n",ra,rb,q,a[i*N + tid]);
        // }
    }
}

__global__ void cudaFusionDown(int N,unsigned long long* a,unsigned long long* b,int size,unsigned long long* q,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv,unsigned long long *spinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++){
        unsigned long long q = q_const[i];
        unsigned long long mu = mu_const[i];
        unsigned long long qbit = qbit_const[i];
        unsigned long long acc = b[tid + i * N] % q;

        register unsigned long long ra = (a[i*N+tid] + q - acc) % q ;
        register unsigned long long rb = spinv[i];
        uint128_t rc, rx;
        mul64(ra, rb, rc);
        rx = rc >> (qbit - 2);
        mul64(rx.low, mu, rx);
        uint128_t::shiftr(rx, qbit + 2);
        mul64(rx.low, q, rx);
        sub128(rc, rx);

        if (rc.low < q)
            a[i*N + tid] = rc.low;
        else
            a[i*N + tid] = rc.low - q;
        // if(tid == 0){
        //     printf("%llu,%llu,%llu,%llu\n",ra,rb,q,a[i*N + tid]);
        // }
    }
}
__global__ void cudaConvDown_dcomp_new(int N,unsigned long long* a,unsigned long long* b,int size,unsigned long long* q,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv,unsigned long long *spinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++){
        unsigned long long acc = b[i * N + tid] % q[i];
        if(b[i*N+tid] < acc)b[i*N+tid] += q[i];
        a[i*N + tid] = (b[i*N+tid] - acc)*spinv[i]%q[i];
    }
}
__global__ void cudaConvMov_dcomp(int N,unsigned long long* a,unsigned long long* b,int size,unsigned long long* q,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv,unsigned long long *spinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = b[size * N + tid];
    for(int i = 0; i < size; i++){
        a[i*N + tid] = acc % q_const[i];
    }
}

__global__ void cudaConvMov_rescale(int N,unsigned long long* a,unsigned long long* b,int source,int size){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = b[source * N + tid];
    for(int i = 0; i < size; i++){
        a[i*N + tid] = acc % q_const[i];
    }
}

__global__ void cudaConvMov_rescale(int N,unsigned long long* tmpa,unsigned long long* aa,unsigned long long* tmpb,unsigned long long* bb,int source,int size){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long acc = aa[source * N + tid];
    for(int i = 0; i < size; i++){
        if(acc < q_const[i]){
            tmpa[i*N + tid] = acc;
        }else{
            tmpa[i*N + tid] = acc - q_const[i];

        }
    }
    acc = bb[source * N + tid];
    for(int i = 0; i < size; i++){
        if(acc < q_const[i]){
            tmpb[i*N + tid] = acc;
        }else{
            tmpb[i*N + tid] = acc - q_const[i];

        }
    }
}
__global__ void fusionModDown(int N,unsigned long long* a1,unsigned long long* b1,unsigned long long* a2,unsigned long long* b2,unsigned long long* c1,unsigned long long* c2,int size,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv,unsigned long long *spinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    register unsigned long long q = q_const[i];

    unsigned long long acc = b1[size * N + tid] ;
    register unsigned long long ra1 = (b1[i*N+tid] + q - acc)*spinv[i]%q;
    acc = b2[size * N + tid] ;
    register unsigned long long ra2  = (b2[i*N+tid] + q - acc)*spinv[i]%q;



    // register unsigned long long ra1 = a1[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rb1 = c1[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rc1 = ra1 + rb1;
    if (rc1 > q) {
        rc1 -= q;
    }
    a1[tid+ blockDim.x * gridDim.x * i] = rc1;

    // register unsigned long long ra2 = a2[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rb2 = c2[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rc2 = ra2 + rb2;
    if (rc2 > q) {
        rc2 -= q;
    }
    a2[tid + blockDim.x * gridDim.x * i] = rc2;
}
__global__ void fusionModDownROT(int N,unsigned long long* a1,unsigned long long* b1,unsigned long long* a2,unsigned long long* b2,unsigned long long* c1,unsigned long long* c2,int size,unsigned long long* Pmod,unsigned long long* p_hat_inv,unsigned long long* Pinv,unsigned long long *spinv){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    register unsigned long long q = q_const[i];

    unsigned long long acc = b1[size * N + tid] % q;
    register unsigned long long ra1 = (b1[i*N+tid] + q - acc)*spinv[i]%q;
    acc = b2[size * N + tid] % q;
    register unsigned long long ra2  = (b2[i*N+tid] + q - acc)*spinv[i]%q;



    // register unsigned long long ra1 = a1[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rb1 = c1[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rc1 = ra1 + rb1;
    if (rc1 > q) {
        rc1 -= q;
    }
    a1[tid+ blockDim.x * gridDim.x * i] = rc1;

    // register unsigned long long ra2 = a2[tid + blockDim.x * gridDim.x * i];
    // register unsigned long long rb2 = c2[tid + blockDim.x * gridDim.x * i];
    register unsigned long long rc2 = ra2;
    if (rc2 > q) {
        rc2 -= q;
    }
    a2[tid + blockDim.x * gridDim.x * i] = rc2;
}
__global__ void cudaModRaise(unsigned long long* a,int N,int size,unsigned long long* q){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a1 = a[(size-1)*N + tid];
    unsigned long long a2 = a[(size-2)*N + tid];

    unsigned long long b = 0;
    a1 = a1 * 48883158 % 1156019777784512513;
    uint128_t a1_u128;
    mul64(a1,1075003393,a1_u128);
    a1 = (a1_u128 % 1156019777784512513).low;

    a2 = a2 * 1075363841 % 1156019777784512513;
    uint128_t a2_u128;
    mul64(a2,1026136620,a2_u128);
    a2 = (a2_u128 % 1156019777784512513).low;
    b = (a1 + a2) % 1156019777784512513;
    if(1156019777784512513 - b < b){
        unsigned long long c = 1156019777784512513 - b;

        for(int i = 0; i < size-2; i++){
            a[i*N + tid] = q[i] - (c % q[i]);
        }
    }
    else{
        for(int i = 0; i < size-2; i++){
            a[i*N + tid] = b % q[i];
        }
    }
    
}
__global__ void makezero(unsigned long long *a){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    a[tid] = 0;
}

__global__ void polymuladd_old_old(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],int len)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 8
    for(int idx = 0; idx < len; idx++){
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc % qbit_const[idx];

        d[i+ blockDim.x * gridDim.x * idx] = (rc.low + c[i + blockDim.x * gridDim.x * idx])%q_const[idx];
    }
}


__global__ void polymuladd_old(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],int len)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 8
    for(int idx = 0; idx < len; idx++){
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        d[i+ blockDim.x * gridDim.x * idx] = (rc.low + c[i + blockDim.x * gridDim.x * idx])%q_const[idx];
    }
}

__global__ void polymuladd_new(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    int idx = blockIdx.y;
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        d[i+ blockDim.x * gridDim.x * idx] = (rc.low + c[i + blockDim.x * gridDim.x * idx])%q_const[idx];
}

__global__ void polymul_old(unsigned long long a[], unsigned long long b[], unsigned long long c[],int len)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int idx = blockIdx.y;
    for(int idx = 0; idx < len; idx++){
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        if (rc.low < q_const[idx])
            c[i + blockDim.x * gridDim.x * idx] = rc.low;
        else
            c[i + blockDim.x * gridDim.x * idx] = rc.low - q_const[idx];
    }
}

__global__ void polymul_new(unsigned long long a[], unsigned long long b[], unsigned long long c[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;

        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        if (rc.low < q_const[idx])
            c[i + blockDim.x * gridDim.x * idx] = rc.low;
        else
            c[i + blockDim.x * gridDim.x * idx] = rc.low - q_const[idx];
}

__global__ void polyadd_old(unsigned long long* a1, unsigned long long* b1, unsigned long long* c1,int len)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    // int idx = blockIdx.y;
    for(int idx = 0; idx < len; idx++){
        register unsigned long long q = q_const[idx];
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rc1 = ra1 + rb1;
        if (rc1 > q) {
            rc1 -= q;
        }
        c1[i+ blockDim.x * gridDim.x * idx] = rc1;
    }
}

__global__ void polyaddsingle(unsigned long long* a1, unsigned long long* b1, unsigned long long* c1)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = blockIdx.y;

        register unsigned long long q = q_const[idx];
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rc1 = ra1 + rb1;
        if (rc1 > q) {
            rc1 -= q;
        }
        c1[i+ blockDim.x * gridDim.x * idx] = rc1;
}
__global__ void polyadddouble(unsigned long long* a1, unsigned long long* b1, unsigned long long* c1,unsigned long long* a2, unsigned long long* b2, unsigned long long* c2)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;

        register unsigned long long q = q_const[idx];
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rc1 = ra1 + rb1;
        if (rc1 > q) {
            rc1 -= q;
        }
        c1[i+ blockDim.x * gridDim.x * idx] = rc1;

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rc2 = ra2 + rb2;
        if (rc2 > q) {
            rc2 -= q;
        }
        c2[i+ blockDim.x * gridDim.x * idx] = rc2;
}
__global__ void cudarotation_new(unsigned long long* a, unsigned long long* buff,unsigned long long galois_elt,unsigned long long n,int size){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long index_raw = tid * galois_elt;
    unsigned long long index = index_raw & (n-1);
    for(int i = 0; i < size; i++) {
        unsigned long long rval = a[tid + n * i];
        if((index_raw / n) & 1){
            rval = (q_const[i] - rval) % q_const[i];
        }
        buff[index + n * i] = rval;
    }
}
__global__ void cudarotation_new_table(unsigned long long* a, unsigned long long* buff,int* rot1,unsigned long long n,int size){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++) {
        buff[tid+i*n] = a[rot1[tid]+i*n];
    }
}
__global__ void cudarotation_new_table(unsigned long long* a, unsigned long long* buff,unsigned long long* rot1,unsigned long long n,int size){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++) {
        buff[tid+i*n] = a[rot1[tid]+i*n] % q_const[i];
    }
}
__global__ void cudarotation_new_table(unsigned long long* a, unsigned long long* b, unsigned long long* buff1,unsigned long long* buff2 ,unsigned long long* rot1,unsigned long long n,int size){
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = rot1[tid];
    for(int i = 0; i < size; i++) {
        if(a[t+i*n] < q_const[i]){
            buff1[tid+i*n] = a[t+i*n] ;
        }else{
            buff1[tid+i*n] = a[t+i*n] - q_const[i];
        }
        if(b[t+i*n] < q_const[i]){
            buff2[tid+i*n] = b[t+i*n] ;
        }else{
            buff2[tid+i*n] = b[t+i*n] - q_const[i];
        }
    }
}
__global__ void polymulsingle(unsigned long long a[], unsigned long long b[],unsigned long long d[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;

        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        if (rc.low < q_const[idx]) {
            d[i+ blockDim.x * gridDim.x * idx] = rc.low;
        }
        else{
            d[i+ blockDim.x * gridDim.x * idx] = rc.low - q_const[idx];
        }

}
__global__ void polymuladdsingle(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll

    int idx = blockIdx.y;
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        d[i+ blockDim.x * gridDim.x * idx] = (rc.low + c[i + blockDim.x * gridDim.x * idx])%q_const[idx];
}
__global__ void polymuladdsingle(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],unsigned long long p)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll

    int idx = blockIdx.y;
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        d[i+ blockDim.x * gridDim.x * idx] = p*(rc.low + c[i + blockDim.x * gridDim.x * idx])%q_const[idx];
}

__global__ void polymuladdsingle_batch(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],unsigned long long p,int size)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll

    int idx = blockIdx.y;
    for(int ii = 0; ii < size; ii++){
        register unsigned long long ra = a[i + blockDim.x * gridDim.x * idx + blockDim.x * gridDim.x * size * ii];
        register unsigned long long rb = b[i + blockDim.x * gridDim.x * idx + blockDim.x * gridDim.x * size * ii];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        d[i+ blockDim.x * gridDim.x * idx] = p*(rc.low + c[i + blockDim.x * gridDim.x * idx])%q_const[idx];
    }
}
__global__ void rotation_innerProd(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[],int N,int sumnum)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
    register unsigned long long c1acc = 0;
    register unsigned long long c2acc = 0;
    register unsigned long long q = q_const[idx];
    register unsigned long long mu = mu_const[idx];
    register unsigned long long qbit = qbit_const[idx];
    #pragma unroll
    for(int j = 0; j < sumnum; j++){ 
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];
        if(ra1 != 0)ra1 = q - ra1;


        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit - 2);

        mul64(rx1.low, mu, rx1);

        uint128_t::shiftr(rx1, qbit + 2);

        mul64(rx1.low, q, rx1);

        sub128(rc1, rx1);
        if (rc1.low < q)
            c1acc += rc1.low;
        else
            c1acc += rc1.low - q;

        // c1acc %= q;
        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];
        if(ra2 != 0)ra2 = q - ra2;
        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit - 2);

        mul64(rx2.low, mu, rx2);

        uint128_t::shiftr(rx2, qbit + 2);

        mul64(rx2.low, q, rx2);

        sub128(rc2, rx2);
        if (rc2.low < q)
            c2acc += rc2.low;
        else
            c2acc += rc2.low - q;

        if(c2acc >= q){
            c2acc -= q;
        }
        if(c2acc >= q){
            c2acc -= q;
        }
        // c2acc %= q;
    }
    c1[i + blockDim.x * gridDim.x * idx] = c1acc % q;
    c2[i + blockDim.x * gridDim.x * idx] = c2acc % q;
}
__global__ void rotation_innerProd(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[],int N,int sumnum,int depth)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
    register unsigned long long c1acc = 0;
    register unsigned long long c2acc = 0;
    register unsigned long long q = q_const[idx];
    register unsigned long long mu = mu_const[idx];
    register unsigned long long qbit = qbit_const[idx];
    #pragma unroll
    for(int j = depth; j < sumnum; j++){ 
        
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];
        if(ra1 != 0)ra1 = q - ra1;


        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit - 2);

        mul64(rx1.low, mu, rx1);

        uint128_t::shiftr(rx1, qbit + 2);

        mul64(rx1.low, q, rx1);

        sub128(rc1, rx1);
        if (rc1.low < q)
            c1acc += rc1.low;
        else
            c1acc += rc1.low - q;

        c1acc %= q;
        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];
        if(ra2 != 0)ra2 = q - ra2;
        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit - 2);

        mul64(rx2.low, mu, rx2);

        uint128_t::shiftr(rx2, qbit + 2);

        mul64(rx2.low, q, rx2);

        sub128(rc2, rx2);
        if (rc2.low < q)
            c2acc += rc2.low;
        else
            c2acc += rc2.low - q;

        if(c2acc >= q){
            c2acc -= q;
        }
        if(c2acc >= q){
            c2acc -= q;
        }
        c2acc %= q;
    }
    c1[i + blockDim.x * gridDim.x * idx] = c1acc % q;
    c2[i + blockDim.x * gridDim.x * idx] = c2acc % q;
}
__global__ void polymuladddouble (  unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long d1[],
                                    unsigned long long a2[], unsigned long long b2[], unsigned long long c2[],unsigned long long d2[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
        register unsigned long long ra = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb = b1[i + blockDim.x * gridDim.x * idx];

        uint128_t rc, rx;

        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        register unsigned long long rd = rc.low + c1[i + blockDim.x * gridDim.x * idx];
        if (rd > q_const[idx]) {
            rd -= q_const[idx];
        }
        if (rd > q_const[idx]) {
            rd -= q_const[idx];
        }
        d1[i+ blockDim.x * gridDim.x * idx] = rd;


        ra = a2[i + blockDim.x * gridDim.x * idx];
        rb = b2[i + blockDim.x * gridDim.x * idx];


        mul64(ra, rb, rc);

        rx = rc >> (qbit_const[idx] - 2);

        mul64(rx.low, mu_const[idx], rx);

        uint128_t::shiftr(rx, qbit_const[idx] + 2);

        mul64(rx.low, q_const[idx], rx);

        sub128(rc, rx);
        rd = rc.low + c2[i + blockDim.x * gridDim.x * idx];
        if (rd > q_const[idx]) {
            rd -= q_const[idx];
        }
        if (rd > q_const[idx]) {
            rd -= q_const[idx];
        }
        d2[i+ blockDim.x * gridDim.x * idx] = rd;

}


__global__ void polymuldouble(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        if (rc1.low < q_const[idx])
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low;
        else
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low - q_const[idx];

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx];

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        if (rc2.low < q_const[idx])
            c2[i + blockDim.x * gridDim.x * idx] = rc2.low;
        else
            c2[i + blockDim.x * gridDim.x * idx] = rc2.low - q_const[idx];

}

__global__ void polymuldouble_sum(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
    // for(int k = 0; k < 16; k++){
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        if (rc1.low < q_const[idx])
            c1[i + blockDim.x * gridDim.x * idx] += rc1.low;
        else
            c1[i + blockDim.x * gridDim.x * idx] += rc1.low - q_const[idx];
        if(c1[i + blockDim.x * gridDim.x * idx] >= q_const[idx]){
            c1[i + blockDim.x * gridDim.x * idx] -= q_const[idx];
        }

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx];

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        if (rc2.low < q_const[idx])
            c2[i + blockDim.x * gridDim.x * idx] += rc2.low;
        else
            c2[i + blockDim.x * gridDim.x * idx] += rc2.low - q_const[idx];

        if(c2[i + blockDim.x * gridDim.x * idx] >= q_const[idx]){
            c2[i + blockDim.x * gridDim.x * idx] -= q_const[idx];
        }
    // }
}


__global__ void polymuldouble_sum_batch(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[],int N,int sumnum)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
    register unsigned long long c1acc = 0;
    register unsigned long long c2acc = 0;
    register unsigned long long q = q_const[idx];
    register unsigned long long mu = mu_const[idx];
    register unsigned long long qbit = qbit_const[idx];
    #pragma unroll
    for(int j = 0; j < sumnum; j++){ 
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];
        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit - 2);

        mul64(rx1.low, mu, rx1);

        uint128_t::shiftr(rx1, qbit + 2);

        mul64(rx1.low, q, rx1);

        sub128(rc1, rx1);
        if (rc1.low < q)
            c1acc += rc1.low;
        else
            c1acc += rc1.low - q;
        if(c1acc >= q){
            c1acc -= q;
        }

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];

        // if(idx == 2 && i == 0){
        //     printf("%llu,%llu,%llu,%llu\n",ra1,rb1,ra2,rb2);
        // }

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit - 2);

        mul64(rx2.low, mu, rx2);

        uint128_t::shiftr(rx2, qbit + 2);

        mul64(rx2.low, q, rx2);

        sub128(rc2, rx2);
        if (rc2.low < q)
            c2acc += rc2.low;
        else
            c2acc += rc2.low - q;

        if(c2acc >= q){
            c2acc -= q;
        }
    }
    c1[i + blockDim.x * gridDim.x * idx] = c1acc;
    c2[i + blockDim.x * gridDim.x * idx] = c2acc;
}
__global__ void polymulsingle_sum_batch(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],int N,int sumnum)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
    register unsigned long long c1acc = 0;
    register unsigned long long c2acc = 0;
    register unsigned long long q = q_const[idx];
    register unsigned long long mu = mu_const[idx];
    register unsigned long long qbit = qbit_const[idx];
    #pragma unroll
    for(int j = 0; j < sumnum; j++){ 
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit - 2);

        mul64(rx1.low, mu, rx1);

        uint128_t::shiftr(rx1, qbit + 2);

        mul64(rx1.low, q, rx1);

        sub128(rc1, rx1);
        if (rc1.low < q)
            c1acc += rc1.low;
        else
            c1acc += rc1.low - q;
        if(c1acc >= q){
            c1acc -= q;
        }

        
    }
    c1[i + blockDim.x * gridDim.x * idx] = c1acc;
}

__global__ void polymuldouble_sum_batch_batch(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[],int N,int sumnum)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int i = ii % (N*sumnum);
    int idx = blockIdx.y;
    int j = ii / (N*sumnum); 
    // register unsigned long long c1acc = 0;
    // register unsigned long long c2acc = 0;
    // for(int j = 0; j < sumnum; j++){ 
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        if (rc1.low < q_const[idx])
            c1[i + blockDim.x * gridDim.x * idx] += rc1.low;
        else
            c1[i + blockDim.x * gridDim.x * idx] += rc1.low - q_const[idx];
        if(c1[i + blockDim.x * gridDim.x * idx] >= q_const[idx]){
            c1[i + blockDim.x * gridDim.x * idx] -= q_const[idx];
        }

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx + (sumnum + sumnum) * N * j];

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        if (rc2.low < q_const[idx])
            c2[i + blockDim.x * gridDim.x * idx] += rc2.low;
        else
            c2[i + blockDim.x * gridDim.x * idx] += rc2.low - q_const[idx];

        if(c2[i + blockDim.x * gridDim.x * idx] >= q_const[idx]){
            c2[i + blockDim.x * gridDim.x * idx] -= q_const[idx];
        }
    // }
    // c1[i + blockDim.x * gridDim.x * idx] = c1acc;
    // c2[i + blockDim.x * gridDim.x * idx] = c2acc;
}
__global__ void polyks(int pos,unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[])
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * pos];
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        if (rc1.low < q_const[idx])
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low;
        else
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low - q_const[idx];

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx];

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        if (rc2.low < q_const[idx])
            c2[i + blockDim.x * gridDim.x * idx] = rc2.low;
        else
            c2[i + blockDim.x * gridDim.x * idx] = rc2.low - q_const[idx];

}


__global__ void reline(unsigned long long a1[], unsigned long long b1[], unsigned long long c1[],unsigned long long a2[], unsigned long long b2[], unsigned long long c2[],int len)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y;
    for(int j = 0; j < len; j++){
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx]+j;
        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        if (rc1.low < q_const[idx])
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low;
        else
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low - q_const[idx];

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx];

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        if (rc2.low < q_const[idx])
            c2[i + blockDim.x * gridDim.x * idx] = rc2.low;
        else
            c2[i + blockDim.x * gridDim.x * idx] = rc2.low - q_const[idx];
    }

}
__global__ void polymultriple(unsigned long long a1[], unsigned long long b1[],
                            unsigned long long a2[], unsigned long long b2[], 
                            unsigned long long c1[], unsigned long long c2[],unsigned long long c3[]
                            ){
                 register int i = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = blockIdx.y;
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];

        uint128_t rc1, rx1;

        mul64(ra1, ra2, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        if (rc1.low < q_const[idx])
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low;
        else
            c1[i + blockDim.x * gridDim.x * idx] = rc1.low - q_const[idx];

        register unsigned long long rb1 = b1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = b2[i + blockDim.x * gridDim.x * idx];


        // if(i <= 10){
            // printf("%d,%llu,%llu,%llu,%llu,%llu\n",i,ra1,ra2,rb1,rb2,rc1.low);
        // }
        // if(i == 1){
        //     printf("%d,%llu,%llu,%llu,%llu\n",i,ra1,ra2,rb1,rb2);
        // }
        uint128_t rc2, rx2;

        mul64(rb1, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        if (rc2.low < q_const[idx])
            c3[i + blockDim.x * gridDim.x * idx] = rc2.low;
        else
            c3[i + blockDim.x * gridDim.x * idx] = rc2.low - q_const[idx];

        uint128_t rc3, rx3;

        mul64(ra1, rb2, rc3);

        rx3 = rc3 >> (qbit_const[idx] - 2);

        mul64(rx3.low, mu_const[idx], rx3);

        uint128_t::shiftr(rx3, qbit_const[idx] + 2);

        mul64(rx3.low, q_const[idx], rx3);

        sub128(rc3, rx3);
        register unsigned long long tmp = rc3.low;
        if (rc3.low < q_const[idx])
            tmp = rc3.low;
        else
            tmp = rc3.low - q_const[idx];
        

        mul64(ra2, rb1, rc3);

        rx3 = rc3 >> (qbit_const[idx] - 2);

        mul64(rx3.low, mu_const[idx], rx3);

        uint128_t::shiftr(rx3, qbit_const[idx] + 2);

        mul64(rx3.low, q_const[idx], rx3);

        sub128(rc3, rx3);
        if (rc3.low + tmp < q_const[idx])
            c2[i + blockDim.x * gridDim.x * idx] = rc3.low + tmp;
        else
            c2[i + blockDim.x * gridDim.x * idx] = rc3.low + tmp - q_const[idx];

}


__global__ void polymuladdscalar(unsigned long long a1[], unsigned long long c1[],unsigned long long a2[], unsigned long long c2[],double scale,int batchSize)
{
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i==0)printf("%d\n",scale);
    // scale = 1610809345;

    #pragma unroll
    for(int idx = 0; idx < batchSize; idx++){
        unsigned long long rhs1 = (unsigned long long)scale;
        unsigned long long rhs2 = (unsigned long long)(-scale);
        unsigned long long rhs;
        if (rhs2 ==0) {
            rhs = rhs1 % q_const[idx];
        }else{
            rhs = q_const[idx] - (rhs2%q_const[idx]);
        }
        register unsigned long long ra1 = a1[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb1 = rhs;

        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit_const[idx] - 2);

        mul64(rx1.low, mu_const[idx], rx1);

        uint128_t::shiftr(rx1, qbit_const[idx] + 2);

        mul64(rx1.low, q_const[idx], rx1);

        sub128(rc1, rx1);
        register unsigned long long rd1 = (c1[i + blockDim.x * gridDim.x * idx] + rc1.low);
    
        if (rd1 > q_const[idx])
            rd1 -= q_const[idx];

        if (rd1 > q_const[idx])
            rd1 -= q_const[idx];
        
        c1[i + blockDim.x * gridDim.x * idx] = rd1;

        register unsigned long long ra2 = a2[i + blockDim.x * gridDim.x * idx];
        register unsigned long long rb2 = rhs;

        uint128_t rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit_const[idx] - 2);

        mul64(rx2.low, mu_const[idx], rx2);

        uint128_t::shiftr(rx2, qbit_const[idx] + 2);

        mul64(rx2.low, q_const[idx], rx2);

        sub128(rc2, rx2);
        register unsigned long long rd2 = (c2[i + blockDim.x * gridDim.x * idx] + rc2.low);
    
        if (rd2 > q_const[idx])
            rd2 -= q_const[idx];

        if (rd2 > q_const[idx])
            rd2 -= q_const[idx];
        
        c2[i + blockDim.x * gridDim.x * idx] = rd2;
    }
    // if(i < 10){
    //     printf("%lld*%lld %% %lld==%lld\n",ra,rb,q,c[i]);
    // }
}

__global__ void polyaddcalar(unsigned long long a1[], double scale,int batchSize){
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for(int idx = 0; idx < batchSize; idx++){
        unsigned long long rhs1 = (unsigned long long)scale;
        unsigned long long rhs2 = (unsigned long long)(-scale);
        unsigned long long rhs;
        if (rhs2 ==0) {
            rhs = rhs1;
        }else{
            rhs = rhs2;
        }
        // if (rhs != 0)printf("!!%llu\n",rhs);
        a1[i + blockDim.x * gridDim.x * idx] = (a1[i + blockDim.x * gridDim.x * idx] + rhs) % q_const[idx];

    }
}

