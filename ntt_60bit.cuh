#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "uint128.cuh"

extern __constant__ unsigned long long q_const[128];
extern __constant__ unsigned long long qbit_const[128];
extern __constant__ unsigned long long mu_const[128];
extern __constant__ unsigned long long inv_q_last_mod_q_const[128];
extern __constant__ unsigned long long inv_punctured_q_const[128];
extern __constant__ unsigned long long prod_t_gamma_mod_q_const[128];


__global__ void print11(unsigned long long* a){
    for(int i = 0; i < 16; i++){
        printf("%lld\t",a[i]);
    }printf("\n\n\n");
}
__global__ void fill(unsigned long long* a){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    a[tid] = tid;
}
// --------------------------------------------------------------------------------------------------------------------------------------------------------
// declarations for templated ntt functions
void forwardNTT9(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers,unsigned batchSize);
template<unsigned l, unsigned n>  // single kernel NTT
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);

template<unsigned l, unsigned n>  // single kernel INTT
__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

template<unsigned l, unsigned n>  // multi kernel NTT
__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);

template<unsigned l, unsigned n>  // multi kernel INTT
__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

template<unsigned l, unsigned n>  // single kernel NTT batch
__global__ void CTBasedNTTInnerSingle_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division);

template<unsigned l, unsigned n>  // single kernel INTT batch
__global__ void GSBasedINTTInnerSingle_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

template<unsigned l, unsigned n>  // multi kernel omg are you still reading this
__global__ void CTBasedNTTInner_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division);

template<unsigned l, unsigned n>  // i'm not gonna write this one, figure this out on your own
__global__ void GSBasedINTTInner_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// --------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void singleBarrett(uint128_t& a, unsigned long long& q, unsigned long long& mu, int& qbit)
{
    uint128_t rx;

    rx = a >> (qbit - 2);

    mul64(rx.low, mu, rx);

    uint128_t::shiftr(rx, qbit + 2);

    mul64(rx.low, q, rx);

    sub128(a, rx);

    if (a.low >= q)
        a.low -= q;

}



template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3(unsigned long long a[],unsigned long long psi_powers[])
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    unsigned long long q = q_const[blockIdx.y];
    unsigned long long mu = mu_const[blockIdx.y];
    int qbit = qbit_const[blockIdx.y];
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];


    // register int length = 1;
    #pragma unroll
    for(int length = l; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;


        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = target_result;

        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index + step] = first_target_value - second_target_value;
        __syncthreads();
        
    } 
    a[tid * gridDim.x + blockIdx.x + n * blockIdx.y] = shared_array[tid];
    a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y] = shared_array[tid + blockDim.x];
}


template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3f(unsigned long long a[],unsigned long long psi_powers[])
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    unsigned long long q = q_const[blockIdx.y];
    unsigned long long mu = mu_const[blockIdx.y];
    int qbit = qbit_const[blockIdx.y];
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];


    register int length = l;
    register unsigned long long first_value;
    register unsigned long long second_value;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        // if(gid == 0 && blockIdx.y == 1){
        //     printf("%d\n",step);
        // }

        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;


        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = target_result;

        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index + step] = first_target_value - second_target_value;
        // if(step == 64){
        //     first_value = shared_array[target_index];
        //     second_value = shared_array[target_index + step];
        //     length *= 2;
        //     break;
        // }
        __syncthreads();
        
    } 
    int wrapid = tid % 32;
    register unsigned long long first_back = first_value;
    register unsigned long long second_back = second_value;
    #pragma unroll
    for(; length < n/r; length *= 2){
        register int step = (n / length) / 2;
        register int psi_step = tid / step;
        register int target_index = psi_step * step * 2  + tid % step;
        psi_step = gid / step;
        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];
        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow


        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;   
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) % 2 == 1){
            second_value = second_back;
        }else if(wrapid / (step/2) % 2 == 0){
            first_value = first_back;
        }
    } 
    
    a[gid * 2 + n * blockIdx.y ] = first_back;
    a[gid * 2 + 1 + n * blockIdx.y]  = second_back;
}
template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3g(unsigned long long a[], unsigned long long psi_powers[])
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    unsigned long long q = q_const[blockIdx.y];
    unsigned long long mu = mu_const[blockIdx.y];
    int qbit = qbit_const[blockIdx.y];    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];


    register unsigned long long first_back;
    register unsigned long long second_back;
    int length = l;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;
        // if(gid == 0){
        //     printf("###%d\n",step);
        // }
        if(step <= 32){
            break;
        }
        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);


        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index] = target_result;
        shared_array[target_index + step] = first_target_value - second_target_value;
        __syncthreads();
        
    } 

    register int step = (n / length) / 2 / r;
    register int psi_step = tid / step ;
    // register int length =  blockDim.x / step;
    register int target_index = psi_step * step * 2  + tid % step;
    psi_step = tid / step ;
    
    register unsigned long long first_value = shared_array[target_index ];
    register unsigned long long second_value = shared_array[target_index + step]; 


    int wrapid = tid % 32;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        register int target_index = psi_step * step * 2  + tid % step;

        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];

        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow
        // if(gid == 0){
        //     printf("%llu,%llu\n",first_value,second_value);
        // }
        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;
         
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) % 2 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }    
    } 
    a[tid * 2 * gridDim.x + blockIdx.x + n * blockIdx.y] = first_back;
    a[tid * 2 * gridDim.x + blockIdx.x + gridDim.x + n * blockIdx.y] = second_back;
}
template<unsigned l,unsigned n>
__global__ void secondStep3(unsigned long long a[],unsigned long long psi_powers[])
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;

    unsigned long long q = q_const[blockIdx.y];
    unsigned long long mu = mu_const[blockIdx.y];
    int qbit = qbit_const[blockIdx.y];

    register int length = l;
    register int step = (n / length) / 2;
    register int psi_step = tid / step;
    register int target_index = psi_step * step * 2  + tid % step;
    
    register unsigned long long first_value = a[blockIdx.x * blockDim.x * 2 + tid + n * blockIdx.y];
    register unsigned long long second_value = a[blockIdx.x * blockDim.x * 2 + tid + blockDim.x + n * blockIdx.y];
    
    register int wrapid = tid % 32;
    register unsigned long long first_back = first_value;
    register unsigned long long second_back = second_value;
    #pragma unroll
    for(int length = l; length < n; length *= 2){
        step = (n / length) / 2;
        psi_step = tid / step;
        target_index = psi_step * step * 2  + tid % step;
        psi_step = gid / step;
        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];
        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow


        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;   
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) % 2 == 1){
            second_value = second_back;
        }else{
            first_value = first_back;
        }
    } 
    
    a[gid * 2 + n * blockIdx.y ] = first_back;
    a[gid * 2 + 1 + n * blockIdx.y]  = second_back;
}
template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3(unsigned long long a[],unsigned long long psi_powers[],int moduleQplen)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    int moduleidx = blockIdx.y % moduleQplen;
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];    

    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];


    // register int length = 1;
    #pragma unroll
    for(int length = l; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;


        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = target_result;

        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index + step] = first_target_value - second_target_value;
        __syncthreads();
        
    } 
    a[tid * gridDim.x + blockIdx.x + n * blockIdx.y] = shared_array[tid];
    a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y] = shared_array[tid + blockDim.x];
}
template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3g1(unsigned long long a[], unsigned long long psi_powers[],int moduleQplen)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    int moduleidx = blockIdx.y % moduleQplen;
    
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];


    register unsigned long long first_back;
    register unsigned long long second_back;
    int length = l;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;
        // if(gid == 0){
        //     printf("###%d\n",step);
        // }
        if(step <= 32){
            break;
        }
        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);


        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index] = target_result;
        shared_array[target_index + step] = first_target_value - second_target_value;
        __syncthreads();
        
    } 

    register int step = (n / length) / 2 / r;
    register int psi_step = tid / step ;
    // register int length =  blockDim.x / step;
    register int target_index = psi_step * step * 2  + tid % step;
    psi_step = tid / step ;
    
    register unsigned long long first_value = shared_array[target_index ];
    register unsigned long long second_value = shared_array[target_index + step]; 


    int wrapid = tid % 32;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        register int target_index = psi_step * step * 2  + tid % step;

        // psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];

        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow
        // if(gid == 0){
        //     printf("%llu,%llu\n",first_value,second_value);
        // }
        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;
         
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) & 1 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }    
    } 
    a[tid * 2 * gridDim.x + blockIdx.x + n * blockIdx.y] = first_back;
    a[tid * 2 * gridDim.x + blockIdx.x + gridDim.x + n * blockIdx.y] = second_back;
}
template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3i(unsigned long long a[], unsigned long long b[],unsigned long long psi_powers[],int moduleQplen)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    // int moduleidx = blockIdx.y % moduleQplen;
    register unsigned long long source1 = b[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    register unsigned long long source2 = b[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];
    for(int ii = 0; ii < moduleQplen; ii++){
        unsigned long long q = q_const[ii];
        unsigned long long mu = mu_const[ii];
        int qbit = qbit_const[ii];    
        unsigned long long tmp = source1;
        if(source1 > q){
            tmp -= q;
        }
        shared_array[tid] = tmp;
        tmp = source2;
        if(source2 > q){
            tmp -= q;
        }
        shared_array[tid + blockDim.x] = tmp;


        register unsigned long long first_back;
        register unsigned long long second_back;
        int length = l;
        #pragma unroll
        for(; length < n / r ; length *= 2){
            register int step = (n / length) / 2 / r;
            register int psi_step = tid / step ;
            // register int length =  blockDim.x / step;
            register int target_index = psi_step * step * 2  + tid % step;
            // if(gid == 0){
            //     printf("###%d\n",step);
            // }
            if(step <= 32){
                break;
            }
            psi_step = tid / step ;

            register unsigned long long psi = psi_powers[length + psi_step + n * ii];

            register unsigned long long first_target_value = shared_array[target_index];

            // if(ii == 0 && blockIdx.y == 1 && gid == 0){
            //     printf("@@%d\n",first_target_value);
            // }
            

            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);


            first_target_value += q * (first_target_value < second_target_value);

            shared_array[target_index] = target_result;
            shared_array[target_index + step] = first_target_value - second_target_value;
            __syncthreads();
            
        } 

        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;
        psi_step = tid / step ;
        
        register unsigned long long first_value = shared_array[target_index ];
        register unsigned long long second_value = shared_array[target_index + step]; 

        // if(ii == 1 && blockIdx.y == 1 && gid == 0){
        //     printf("@@%d\n",first_value);
        // }
        int wrapid = tid % 32;
        #pragma unroll
        for(; length < n / r ; length *= 2){
            register int step = (n / length) / 2 / r;
            register int psi_step = tid / step ;
            register int target_index = psi_step * step * 2  + tid % step;

            // psi_step = tid / step ;

            register unsigned long long psi = psi_powers[length + psi_step + n * ii];

            register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow
            // if(gid == 0){
            //     printf("%llu,%llu\n",first_value,second_value);
            // }
            mul64(temp_storage.low, psi, temp_storage);
            // if(ii == 0 && blockIdx.y == 0 && gid == 0){
            //     printf("@@%d\n",first_value);
            // }
            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_value + second_target_value;

            target_result -= q * (target_result >= q);

            second_value = first_value + q * (first_value < second_target_value) - second_target_value;
            
            first_value = target_result;

            first_back = first_value;
            second_back = second_value;
            
            first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
            second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

            if(wrapid / (step / 2) & 1 == 1){
                second_value = second_back;
            }else {
                first_value = first_back;
            }    
        } 
        // if(ii == 0 && blockIdx.y == 0 && gid == 0){
        //     printf("@@%d\n",first_back);
        // }
        a[tid * 2 * gridDim.x + blockIdx.x + n * ii + moduleQplen * n * blockIdx.y] = first_back;
        a[tid * 2 * gridDim.x + blockIdx.x + gridDim.x + n * ii + moduleQplen * n * blockIdx.y] = second_back;
    }
}
template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3h(unsigned long long a[],
unsigned long long psi_powers[])
{
    register int tid = threadIdx.x;
    // register int gid = blockDim.x * blockIdx.x + tid;
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];
    int moduleidx = blockIdx.y;
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];  

    register unsigned long long first_back;
    register unsigned long long second_back;
    register int step;
    register int psi_step;
    register int target_index;
    register unsigned long long first_value;
    register unsigned long long second_value; 
    int length = l;
    #pragma unroll
    for(; length < n / 4096; length *= 2){
        step = (n / length) / 2 / r;
        psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        target_index = psi_step * step * 2  + tid % step;
        // if(gid == 0){
        //     printf("%d\n",step);
        // }


        // if(step <= 32){
        //     break;
        // }
        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);


        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index] = target_result;
        shared_array[target_index + step] = first_target_value - second_target_value;
        __syncthreads();
        
    } 

    step = (n / length) / 2 / r;
    psi_step = tid / step ;
    // register int length =  blockDim.x / step;
    target_index = psi_step * step * 2  + tid % step;
    psi_step = tid / step ;
    
    first_value = shared_array[target_index ];
    second_value = shared_array[target_index + step]; 


    int wrapid = tid % 32;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        step = (n / length) / 2 / r;
        psi_step = tid / step ;
        target_index = psi_step * step * 2  + tid % step;

        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * blockIdx.y];

        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow
        // if(gid == 0){
        //     printf("%llu,%llu\n",first_value,second_value);
        // }
        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;
         
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) & 1 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }    
    } 
    a[tid * 2 * gridDim.x + blockIdx.x + n * blockIdx.y] = first_back;
    a[tid * 2 * gridDim.x + blockIdx.x + gridDim.x + n * blockIdx.y] = second_back;
}

template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3h2(unsigned long long a[],
 unsigned long long psi_powers[],int moduleQplen)
{
    register int tid = threadIdx.x;
    // register int gid = blockDim.x * blockIdx.x + tid;
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2
    
    shared_array[tid] = a[tid * gridDim.x + blockIdx.x + n * blockIdx.y];
    shared_array[tid + blockDim.x] = a[tid * gridDim.x + blockIdx.x + n / 2 + n * blockIdx.y];

    int moduleidx = blockIdx.y % moduleQplen;
    
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];  

    register unsigned long long first_back;
    register unsigned long long second_back;
    register int step;
    register int psi_step;
    register int target_index;
    register unsigned long long first_value;
    register unsigned long long second_value; 
    int length = l;
    #pragma unroll
    for(; length < n / 4096; length *= 2){
        step = (n / length) / 2 / r;
        psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        target_index = psi_step * step * 2  + tid % step;
        // if(gid == 0){
        //     printf("%d\n",step);
        // }


        // if(step <= 32){
        //     break;
        // }
        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];

        register unsigned long long first_target_value = shared_array[target_index];
        register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);


        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index] = target_result;
        shared_array[target_index + step] = first_target_value - second_target_value;
        __syncthreads();
        
    } 

    step = (n / length) / 2 / r;
    psi_step = tid / step ;
    // register int length =  blockDim.x / step;
    target_index = psi_step * step * 2  + tid % step;
    psi_step = tid / step ;
    
    first_value = shared_array[target_index ];
    second_value = shared_array[target_index + step]; 


    int wrapid = tid % 32;
    #pragma unroll
    for(; length < n / r ; length *= 2){
        step = (n / length) / 2 / r;
        psi_step = tid / step ;
        target_index = psi_step * step * 2  + tid % step;

        psi_step = tid / step ;

        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];

        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow
        // if(gid == 0){
        //     printf("%llu,%llu\n",first_value,second_value);
        // }
        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;
         
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) & 1 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }    
    } 
    a[tid * 2 * gridDim.x + blockIdx.x + n * blockIdx.y] = first_back;
    a[tid * 2 * gridDim.x + blockIdx.x + gridDim.x + n * blockIdx.y] = second_back;
}



template<unsigned l,unsigned n>
__global__ void secondStep3(unsigned long long a[],unsigned long long psi_powers[],int moduleQplen)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;

    int moduleidx = blockIdx.y % moduleQplen;
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];

    register int length = l;
    register int step = (n / length) / 2;
    register int psi_step = tid / step;
    register int target_index = psi_step * step * 2  + tid % step;
    
    register unsigned long long first_value = a[blockIdx.x * blockDim.x * 2 + tid + n * blockIdx.y];
    register unsigned long long second_value = a[blockIdx.x * blockDim.x * 2 + tid + blockDim.x + n * blockIdx.y];
    
    register int wrapid = tid % 32;
    register unsigned long long first_back = first_value;
    register unsigned long long second_back = second_value;
    #pragma unroll
    for(int length = l; length < n; length *= 2){
        step = (n / length) / 2;
        psi_step = tid / step;
        target_index = psi_step * step * 2  + tid % step;
        psi_step = gid / step;
        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];
        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow


        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;   
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) & 1 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }
    } 
    
    a[gid * 2 + n * blockIdx.y ] = first_back;
    a[gid * 2 + 1 + n * blockIdx.y]  = second_back;
}

template<unsigned l,unsigned n>
__global__ void secondStep3i(unsigned long long a[],unsigned long long psi_powers[],int moduleQplen,unsigned long long* keya,unsigned long long* keyb,unsigned long long* suma,unsigned long long* sumb,bool isrot)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;

    int moduleidx = blockIdx.y % moduleQplen;
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];
    register unsigned long long c1a=0;
    register unsigned long long c1b=0;
    register unsigned long long c2a=0;
    register unsigned long long c2b=0;

    #pragma unroll
    for(int pcnt = 0; pcnt < moduleQplen - 1; pcnt++){
        register int length = l;
        register int step = (n / length) / 2;
        register int psi_step = tid / step;
        register int target_index = psi_step * step * 2  + tid % step;
        
        register unsigned long long first_value = a[blockIdx.x * blockDim.x * 2 + tid + n * blockIdx.y + pcnt * moduleQplen * n];
        register unsigned long long second_value = a[blockIdx.x * blockDim.x * 2 + tid + blockDim.x + n * blockIdx.y + pcnt * moduleQplen * n];

        register int wrapid = tid % 32;
        register unsigned long long first_back = first_value;
        register unsigned long long second_back = second_value;
        #pragma unroll
        for(int length = l; length < n; length *= 2){
            step = (n / length) / 2;
            psi_step = tid / step;
            target_index = psi_step * step * 2  + tid % step;
            psi_step = gid / step;



            register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];
            register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow


            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_value + second_target_value;

            target_result -= q * (target_result >= q);

            second_value = first_value + q * (first_value < second_target_value) - second_target_value;
            
            first_value = target_result;

            first_back = first_value;
            second_back = second_value;   
            first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
            second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

            if(wrapid / (step / 2) & 1 == 1){
                second_value = second_back;
            }else {
                first_value = first_back;
            }

        } 

        int sumnum = moduleQplen - 1;
        register unsigned long long ra1 = first_back;

        if(isrot && ra1) ra1 = q - ra1; 
        // register unsigned long long ra1 = a1[gid * 2 + n * blockIdx.y];
        ulong2 keyadata = ((ulong2*)keya)[gid + pcnt * (sumnum ) * n + moduleidx * n / 2];
        ulong2 keybdata = ((ulong2*)keyb)[gid + pcnt * (sumnum ) * n + moduleidx * n / 2];


        register unsigned long long rb1 = keyadata.x;

        register unsigned long long rb2 = keybdata.x;
        uint128_t rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit - 2);

        mul64(rx1.low, mu, rx1);

        uint128_t::shiftr(rx1, qbit + 2);

        mul64(rx1.low, q, rx1);

        sub128(rc1, rx1);
    
        if (rc1.low < q)
            c1a += rc1.low;
        else
            c1a += rc1.low - q;
        if(c1a >= q){
            c1a -= q;
        }

        register unsigned long long ra2 = ra1;//a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];
        // if(blockIdx.y == 2 && gid == 0){
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
            c1b += rc2.low;
        else
            c1b += rc2.low - q;

        if(c1b >= q){
            c1b -= q;
        }

        ra1 = second_back;
        // register unsigned long long ra1 = a1[gid * 2 + n * blockIdx.y];
        if(isrot && ra1) ra1 = q - ra1; 
        rb1 = keyadata.y;

        rb2 = keybdata.y;

        rc1, rx1;

        mul64(ra1, rb1, rc1);

        rx1 = rc1 >> (qbit - 2);

        mul64(rx1.low, mu, rx1);

        uint128_t::shiftr(rx1, qbit + 2);

        mul64(rx1.low, q, rx1);

        sub128(rc1, rx1);
        if (rc1.low < q)
            c2a += rc1.low;
        else
            c2a += rc1.low - q;
        if(c2a >= q){
            c2a -= q;
        }

        ra2 = ra1;//a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];

        rc2, rx2;

        mul64(ra2, rb2, rc2);

        rx2 = rc2 >> (qbit - 2);

        mul64(rx2.low, mu, rx2);

        uint128_t::shiftr(rx2, qbit + 2);

        mul64(rx2.low, q, rx2);

        sub128(rc2, rx2);
        if (rc2.low < q)
            c2b += rc2.low;
        else
            c2b += rc2.low - q;

        if(c2b >= q){
            c2b -= q;
        }
        
    }
    // if(blockIdx.y == 1 && gid == 0){
    //     printf("LL%d\n",c1a);
    // }
    
    ulong2 tmp;
    tmp.x = c1a;
    tmp.y = c2a;
    ((ulong2*)suma)[gid  + n / 2 * moduleidx ] = tmp;
    // suma[gid * 2 + 1 + n * moduleidx]  = c2a;
    tmp.x = c1b;
    tmp.y = c2b;    
    ((ulong2*)sumb)[gid  + n/2 * moduleidx ] = tmp;
    // sumb[gid * 2 + 1 + n * moduleidx]  = c2b;

}

template<unsigned l,unsigned n>
__global__ void secondStep3Fusion(unsigned long long a[],unsigned long long psi_powers[],int moduleQplen,unsigned long long* keya,unsigned long long* keyb,unsigned long long* suma,unsigned long long* sumb)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;

    int moduleidx = blockIdx.y % moduleQplen;
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];

    register int length = l;
    register int step = (n / length) / 2;
    register int psi_step = tid / step;
    register int target_index = psi_step * step * 2  + tid % step;
    
    register unsigned long long first_value = a[blockIdx.x * blockDim.x * 2 + tid + n * blockIdx.y];
    register unsigned long long second_value = a[blockIdx.x * blockDim.x * 2 + tid + blockDim.x + n * blockIdx.y];
    
    register int wrapid = tid % 32;
    register unsigned long long first_back = first_value;
    register unsigned long long second_back = second_value;
    #pragma unroll
    for(int length = l; length < n; length *= 2){
        step = (n / length) / 2;
        psi_step = tid / step;
        target_index = psi_step * step * 2  + tid % step;
        psi_step = gid / step;
        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];
        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow


        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;   
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) & 1 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }
    } 
    int sumnum = moduleQplen - 1;

    register unsigned long long c1acc;
    register unsigned long long c2acc;


    register unsigned long long ra1 = first_back;
    // register unsigned long long ra1 = a1[gid * 2 + n * blockIdx.y];

    register unsigned long long rb1 = keya[gid * 2 + blockIdx.y/(sumnum + 1) * (sumnum + sumnum) * n + moduleidx * n];

    register unsigned long long rb2 = keyb[gid * 2 + blockIdx.y/(sumnum + 1) * (sumnum + sumnum) * n + moduleidx * n];
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

    register unsigned long long ra2 = ra1;//a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];

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
    suma[gid * 2 + n * moduleidx ] = c1acc;
    sumb[gid * 2 + n * moduleidx ] = c2acc;

    ra1 = second_back;
    // register unsigned long long ra1 = a1[gid * 2 + n * blockIdx.y];

    rb1 = keya[gid * 2 + 1 + blockIdx.y/(sumnum + 1) * (sumnum + sumnum) * n + moduleidx * n];

    rb2 = keyb[gid * 2 + 1 + blockIdx.y/(sumnum + 1) * (sumnum + sumnum) * n + moduleidx * n];
    rc1, rx1;

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

    ra2 = ra1;//a2[i + blockDim.x * gridDim.x * idx + (sumnum + 1) * N * j];

    rc2, rx2;

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

    suma[gid * 2 + 1 + n * moduleidx]  = c1acc;
    sumb[gid * 2 + 1 + n * moduleidx]  = c2acc;
}

template<unsigned l,unsigned n,unsigned r>
__global__ void firstStep3j(unsigned long long a[], unsigned long long psi_powers[],int batchsize)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;
    extern __shared__ unsigned long long shared_array[]; // blockDim.x * 2


    register unsigned long long first_back;
    register unsigned long long second_back;
    register unsigned long long source1 = a[tid * gridDim.x + blockIdx.x + n * batchsize];
    register unsigned long long source2 = a[tid * gridDim.x + blockIdx.x + n / 2 + n * batchsize];
    for(int ii = 0; ii < batchsize; ii++){
        int length = l;
        unsigned long long q = q_const[ii];
        unsigned long long mu = mu_const[ii];
        int qbit = qbit_const[ii];    
        unsigned long long tmp = source1;
        if(source1 > q){
            tmp -= q;
        }
        shared_array[tid] = tmp;

        tmp = source2;
        if(source2 > q){
            tmp -= q;
        }
        shared_array[tid + blockDim.x] = tmp;
        #pragma unroll
        for(; length < n / r ; length *= 2){
            register int step = (n / length) / 2 / r;
            register int psi_step = tid / step ;
            // register int length =  blockDim.x / step;
            register int target_index = psi_step * step * 2  + tid % step;
            // if(gid == 0){
            //     printf("###%d\n",step);
            // }
            if(step <= 32){
                break;
            }
            psi_step = tid / step ;

            register unsigned long long psi = psi_powers[length + psi_step + n * ii];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);


            first_target_value += q * (first_target_value < second_target_value);

            shared_array[target_index] = target_result;
            shared_array[target_index + step] = first_target_value - second_target_value;
            __syncthreads();
            
        } 

        register int step = (n / length) / 2 / r;
        register int psi_step = tid / step ;
        // register int length =  blockDim.x / step;
        register int target_index = psi_step * step * 2  + tid % step;
        psi_step = tid / step ;
        
        register unsigned long long first_value = shared_array[target_index ];
        register unsigned long long second_value = shared_array[target_index + step]; 


        int wrapid = tid % 32;
        #pragma unroll
        for(; length < n / r ; length *= 2){
            register int step = (n / length) / 2 / r;
            register int psi_step = tid / step ;
            register int target_index = psi_step * step * 2  + tid % step;

            psi_step = tid / step ;

            register unsigned long long psi = psi_powers[length + psi_step + n * ii];

            register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow
            // if(gid == 0){
            //     printf("%llu,%llu\n",first_value,second_value);
            // }
            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_value + second_target_value;

            target_result -= q * (target_result >= q);

            second_value = first_value + q * (first_value < second_target_value) - second_target_value;
            
            first_value = target_result;

            first_back = first_value;
            second_back = second_value;
            
            first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
            second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

            if((wrapid / (step / 2) )& 1 == 1){
                second_value = second_back;
            }else {
                first_value = first_back;
            }    
        } 
        a[tid * 2 * gridDim.x + blockIdx.x + n * ii] = first_back;
        a[tid * 2 * gridDim.x + blockIdx.x + gridDim.x + n * ii] = second_back;
    }
}


template<unsigned l,unsigned n>
__global__ void secondStep3j(unsigned long long a[],unsigned long long b[],unsigned long long psi_powers[],int moduleQplen,unsigned long long* spinv)
{
    register int tid = threadIdx.x;
    register int gid = blockDim.x * blockIdx.x + tid;

    int moduleidx = blockIdx.y % moduleQplen;
    unsigned long long q = q_const[moduleidx];
    unsigned long long mu = mu_const[moduleidx];
    int qbit = qbit_const[moduleidx];

    register int length = l;
    register int step = (n / length) / 2;
    register int psi_step = tid / step;
    register int target_index = psi_step * step * 2  + tid % step;
    
    register unsigned long long first_value = a[blockIdx.x * blockDim.x * 2 + tid + n * blockIdx.y];
    register unsigned long long second_value = a[blockIdx.x * blockDim.x * 2 + tid + blockDim.x + n * blockIdx.y];
    
    register int wrapid = tid % 32;
    register unsigned long long first_back = first_value;
    register unsigned long long second_back = second_value;
    #pragma unroll
    for(int length = l; length < n; length *= 2){
        step = (n / length) / 2;
        psi_step = tid / step;
        target_index = psi_step * step * 2  + tid % step;
        psi_step = gid / step;
        register unsigned long long psi = psi_powers[length + psi_step + n * moduleidx];
        register uint128_t temp_storage = second_value;  // this is for eliminating the possibility of overflow


        mul64(temp_storage.low, psi, temp_storage);

        singleBarrett(temp_storage, q, mu, qbit);
        register unsigned long long second_target_value = temp_storage.low;

        register unsigned long long target_result = first_value + second_target_value;

        target_result -= q * (target_result >= q);

        second_value = first_value + q * (first_value < second_target_value) - second_target_value;
        
        first_value = target_result;

        first_back = first_value;
        second_back = second_value;   
        first_value = __shfl_xor_sync(0xffffffff,second_back,step/2);
        second_value = __shfl_xor_sync(0xffffffff,first_back,step/2);

        if(wrapid / (step / 2) & 1 == 1){
            second_value = second_back;
        }else {
            first_value = first_back;
        }
    } 
    unsigned long long acc = first_back;
    register ulong2 bdouble = ((ulong2*)b)[gid + n / 2 * blockIdx.y];
    register unsigned long long ra = (bdouble.x + q - acc) ;
    if(ra >= q)ra-=q;

    register unsigned long long rb = spinv[blockIdx.y];
    uint128_t rc, rx;
    mul64(ra, rb, rc);
    rx = rc >> (qbit - 2);
    mul64(rx.low, mu, rx);
    uint128_t::shiftr(rx, qbit + 2);
    mul64(rx.low, q, rx);
    sub128(rc, rx);
    unsigned long long res1 = rc.low;
    if(res1 > q){
        res1 -= q;
    }
    // b[gid * 2 + n * blockIdx.y] = res1;



    acc = second_back;

    ra = (b[gid * 2 + 1 + n * blockIdx.y] + q - acc)  ;
    if(ra >= q)ra-=q;

    rc, rx;
    mul64(ra, rb, rc);
    rx = rc >> (qbit - 2);
    mul64(rx.low, mu, rx);
    uint128_t::shiftr(rx, qbit + 2);
    mul64(rx.low, q, rx);
    sub128(rc, rx);

    unsigned long long res2 = rc.low;
    if(res2 > q){
        res2 -= q;
    }
    // b[gid * 2 + 1 + n * blockIdx.y] = res2;
    ulong2 res;
    res.x = res1;
    res.y = res2;
    ((ulong2*)b)[gid + n / 2 * blockIdx.y] = res;
    // if (rc.low < q)
    //     b[gid * 2 + 1 + n * blockIdx.y] = rc.low;
    // else
    //     b[gid * 2 + 1 + n * blockIdx.y] = rc.low - q;

    // a[gid * 2 + n * blockIdx.y ] = first_back;
    // a[gid * 2 + 1 + n * blockIdx.y]  = second_back;
}
__host__ void fusionModdown(unsigned long long* device_a,unsigned long long* device_b, unsigned n, unsigned long long* psi_powers,int batchSize,unsigned long long* spinv){
    if(n == 2048){
        const int sizeScale = 1;

        dim3 dim1(64,1);
        dim3 dim2(32*sizeScale,batchSize);

        firstStep3j<1,2048*sizeScale,64><<<dim1, 16, 32 * sizeof(unsigned long long)>>>(device_a,psi_powers,batchSize);
        // print1<<<1,1>>>(device_a);

        secondStep3j<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,device_b,psi_powers,batchSize,spinv);
    }
    else if(n == 4096){
        const int sizeScale = 2;
        dim3 dim1(64,1);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3j<1,2048*sizeScale,64><<<dim1, 32, 64 * sizeof(unsigned long long)>>>(device_a,psi_powers,batchSize);
        secondStep3j<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,device_b,psi_powers,batchSize,spinv);
    }
    else if(n == 8192){
        const int sizeScale = 4;
        dim3 dim1(64,1);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3j<1,2048*sizeScale,64><<<dim1, 64, 128 * sizeof(unsigned long long)>>>(device_a,psi_powers,batchSize);
        secondStep3j<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,device_b,psi_powers,batchSize,spinv);
    }
    else if(n == 8192 * 2){
        const int sizeScale = 8;
        dim3 dim1(64,1);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3j<1,2048*sizeScale,64><<<dim1, 128, 256 * sizeof(unsigned long long)>>>(device_a,psi_powers,batchSize);
        secondStep3j<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,device_b,psi_powers,batchSize,spinv);
    }
    else if(n == 8192 * 4){
        const int sizeScale = 16;
        dim3 dim1(64,1);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3j<1,2048*sizeScale,64><<<dim1, 256, 512 * sizeof(unsigned long long)>>>(device_a,psi_powers,batchSize);

        secondStep3j<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,device_b,psi_powers,batchSize,spinv);
    }else if(n == 8192 * 8){
        const int sizeScale = 32;
        dim3 dim1(64,1);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3j<1,2048*sizeScale,64><<<dim1, 512, 1024 * sizeof(unsigned long long)>>>(device_a,psi_powers,batchSize);

        secondStep3j<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,device_b,psi_powers,batchSize,spinv);        
    }
    else{
        printf("not imp\n");
    } 
}
__host__ void forwardNTT3(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers,int batchSize){
    if(n == 2048){
        const int sizeScale = 1;

        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);

        firstStep3g<1,2048*sizeScale,64><<<dim1, 16, 32 * sizeof(unsigned long long)>>>(device_a,psi_powers);
        // print1<<<1,1>>>(device_a);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers);
    }
    else if(n == 4096){
        const int sizeScale = 2;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 32, 64 * sizeof(unsigned long long)>>>(device_a,psi_powers);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers);
    }
    else if(n == 8192){
        const int sizeScale = 4;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 64, 128 * sizeof(unsigned long long)>>>(device_a,psi_powers);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers);
    }
    else if(n == 8192 * 2){
        const int sizeScale = 8;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 128, 256 * sizeof(unsigned long long)>>>(device_a,psi_powers);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers);
    }
    else if(n == 8192 * 4){
        const int sizeScale = 16;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 256, 512 * sizeof(unsigned long long)>>>(device_a,psi_powers);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers);
    }else if(n == 8192 * 8){
        const int sizeScale = 32;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 512, 1024 * sizeof(unsigned long long)>>>(device_a,psi_powers);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers);        
    }
    else{
        printf("not imp\n");
    } 
}



__host__ void forwardNTT3(cudaStream_t& stream1,unsigned long long* device_a, unsigned n, unsigned long long* psi_powers,int batchSize){
    if(n == 2048){
        const int sizeScale = 1;

        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);

        firstStep3g<1,2048*sizeScale,64><<<dim1, 16, 32 * sizeof(unsigned long long),stream1>>>(device_a,psi_powers);
        // print1<<<1,1>>>(device_a);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32,0,stream1>>>(device_a,psi_powers);
    }
    else if(n == 4096){
        const int sizeScale = 2;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 32, 64 * sizeof(unsigned long long),stream1>>>(device_a,psi_powers);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32,0,stream1>>>(device_a,psi_powers);
    }
    else if(n == 8192){
        const int sizeScale = 4;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 64, 128 * sizeof(unsigned long long),stream1>>>(device_a,psi_powers);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32,0,stream1>>>(device_a,psi_powers);
    }
    else if(n == 8192 * 2){
        const int sizeScale = 8;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 128, 256 * sizeof(unsigned long long),stream1>>>(device_a,psi_powers);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32,0,stream1>>>(device_a,psi_powers);
    }
    else if(n == 8192 * 4){
        const int sizeScale = 16;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g<1,2048*sizeScale,64><<<dim1, 256, 512 * sizeof(unsigned long long),stream1>>>(device_a,psi_powers);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32,0,stream1>>>(device_a,psi_powers);
    }
    else{
        printf("not imp\n");
    } 
}



__host__ void forwardNTT3(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers,int batchSize,int moduleQplen){
    if(n == 2048){
        const int sizeScale = 1;

        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);

        firstStep3g1<1,2048*sizeScale,64><<<dim1, 16, 32 * sizeof(unsigned long long)>>>(device_a,psi_powers,moduleQplen);
        // print1<<<1,1>>>(device_a);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen);
    }
    else if(n == 4096){
        const int sizeScale = 2;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g1<1,2048*sizeScale,64><<<dim1, 32, 64 * sizeof(unsigned long long)>>>(device_a,psi_powers,moduleQplen);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen);
    }
    else if(n == 8192){
        const int sizeScale = 4;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g1<1,2048*sizeScale,64><<<dim1, 64, 128 * sizeof(unsigned long long)>>>(device_a,psi_powers,moduleQplen);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen);
    }
    else if(n == 8192 * 2){
        const int sizeScale = 8;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g1<1,2048*sizeScale,64><<<dim1, 128, 256 * sizeof(unsigned long long)>>>(device_a,psi_powers,moduleQplen);
        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen);
    }
    else if(n == 8192 * 4){
        const int sizeScale = 16;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g1<1,2048*sizeScale,64><<<dim1, 256, 512 * sizeof(unsigned long long)>>>(device_a,psi_powers,moduleQplen);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen);
    }
    else if(n == 8192 * 8){
        const int sizeScale = 32;
        dim3 dim1(64,batchSize);
        dim3 dim2(32*sizeScale,batchSize);
        firstStep3g1<1,2048*sizeScale,64><<<dim1, 512, 1024 * sizeof(unsigned long long)>>>(device_a,psi_powers,moduleQplen);

        secondStep3<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen);        
    }
    else{
        printf("not imp\n");
    } 
}

__host__ void forwardNTT3_fusion(unsigned long long* device_a,unsigned long long* device_b, unsigned n, unsigned long long* psi_powers,int batchSize,int moduleQplen,unsigned long long* a,unsigned long long* b,unsigned long long* suma,unsigned long long* sumb,bool isrot){
    if(n == 2048){
        const int sizeScale = 1;

        dim3 dim1(64,moduleQplen-1);
        dim3 dim2(32*sizeScale,moduleQplen);

        firstStep3i<1,2048*sizeScale,64><<<dim1, 16, 32 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);
        // print1<<<1,1>>>(device_a);

        secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen,a,b,suma,sumb,isrot);
    }
    else if(n == 4096){
        const int sizeScale = 2;
        dim3 dim1(64,moduleQplen-1);
        dim3 dim2(32*sizeScale,moduleQplen);
        firstStep3i<1,2048*sizeScale,64><<<dim1, 32, 64 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);
        secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen,a,b,suma,sumb,isrot);
    }
    else if(n == 8192){
        const int sizeScale = 4;
        dim3 dim1(64,moduleQplen-1);
        dim3 dim2(32*sizeScale,moduleQplen);
        firstStep3i<1,2048*sizeScale,64><<<dim1, 64, 128 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);
        secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen,a,b,suma,sumb,isrot);
    }
    else if(n == 8192 * 2){
        const int sizeScale = 8;
        dim3 dim1(64,moduleQplen-1);
        dim3 dim2(32*sizeScale,moduleQplen);
        firstStep3i<1,2048*sizeScale,64><<<dim1, 128, 256 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);
        secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen,a,b,suma,sumb,isrot);
    }
    else if(n == 8192 * 4){
        const int sizeScale = 16;
        dim3 dim1(64,moduleQplen-1);
        dim3 dim2(32*sizeScale,moduleQplen);
        firstStep3i<1,2048*sizeScale,64><<<dim1, 256, 512 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);

        secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen,a,b,suma,sumb,isrot);
    }
    else if(n == 8192 * 8){
        const int sizeScale = 32;
        dim3 dim1(64,moduleQplen-1);
        dim3 dim2(32*sizeScale,moduleQplen);
        firstStep3i<1,2048*sizeScale,64><<<dim1, 512, 1024 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);

        secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_a,psi_powers,moduleQplen,a,b,suma,sumb,isrot);        
    }
    else{
        printf("not imp\n");
    } 
}
template<unsigned l, unsigned n>
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[])
{
    register int local_tid = threadIdx.x; 

    extern __shared__ unsigned long long shared_array[];  // declaration of shared_array

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copying to shared from global
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
    }

#pragma unroll
    for (int length = l; length < n; length *= 2)
    {  // for loops are required since we are handling all the remaining iterations in this kernel
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = target_result;

            first_target_value += q * (first_target_value < second_target_value);

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copy back to global from shared
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid] % q;
    }

}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[])
{
    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];  // declaration of shared_array

    register unsigned long long q2 = (q + 1) >> 1;

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copying to shared from global
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
    }

    __syncthreads();

#pragma unroll
    for (int length = (n / 2); length >= l; length /= 2)
    {  // for loops are required since we are handling all the remaining iterations in this kernel
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

            first_target_value += q * (first_target_value < second_target_value);

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);

            register unsigned long long temp_storage_low = temp_storage.low;

            shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {  // copy back to global from shared
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid] % q;
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[])
{
    // no shared memory - handling only one iteration in here

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    register unsigned long long psi = psi_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];
    // if(global_tid == 8191){
    //     printf("@@@@@@@@@@@@@@%llu,%llu\n",a[target_index],a[target_index + step]);
    // }
    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result % q;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = (first_target_value - second_target_value)%q;
    // if(a[target_index + step] == 442763871){
    //     printf("!!!!!!!!!!!!!!!!!!!!!!%d\n\n\n\n",global_tid);
    // }
}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInner(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[])
{
    // no shared memory - handling only one iteration in here

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step;

    
    
    register unsigned long long psiinv = psiinv_powers[length + psi_step];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    register unsigned long long q2 = (q + 1) >> 1;

    target_result = (target_result >> 1) + q2 * (target_result & 1);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);

    register unsigned long long temp_storage_low = temp_storage.low;

    temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

    a[target_index + step] = temp_storage_low % q;
}

__host__ void forwardNTTdouble(unsigned long long* device_a, unsigned long long* device_b, unsigned n, cudaStream_t& stream1, cudaStream_t& stream2, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{  // performs 2 NTT operations together, check the forwardNTT function below for detailed comments
    if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 8192)
    {
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream2 >> > (device_b, q, mu, bit_length, psi_powers);

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }
    else if (n == 2048)
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream2 >> > (device_b, q, mu, bit_length, psi_powers);
    }else {
        printf("NONONO1\n");
    }
}
__global__ void transpose(unsigned long long *vec,int row, int col){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int x = tid % col;
    int y = tid / col;
    vec[x * row + y] = vec[y * col + x]; 
}
__host__ void forwardNTT(unsigned long long* device_a, unsigned n, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{  // performs single NTT operation
    if (n == 65536)
    {
        CTBasedNTTInner<1, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInner<2, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)

        CTBasedNTTInner<4, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // third iter (multi-kernel)

        CTBasedNTTInner<8, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // third iter (multi-kernel)

        // transpose<<<n/1024,1024>>>(device_a,8,n/8);
        CTBasedNTTInnerSingle<16, 65536> << <16, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // third iter (multi-kernel)
        // transpose<<<n/1024,1024>>>(device_a,8,n/8);
        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)

        CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 8192)
    {
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // all iterations, single-kernel
    }
    else if (n == 2048)
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // all iterations, single-kernel
    }else {
        printf("NONONO2\n");
    }
}
__host__ void forwardNTT_TEST(unsigned long long* device_a, unsigned n, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers)
{  // performs single NTT operation
    if (n == 32768)
    {
        CTBasedNTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)

        CTBasedNTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // third iter (multi-kernel)
        // transpose<<<n/1024,1024>>>(device_a,8,n/8);
        CTBasedNTTInnerSingle<8, 32768> << <8, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 16384)
    {
        CTBasedNTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)
        // printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
        // CTBasedNTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // second iter (mult-kernel)


        // CTBasedNTTInnerSingle<4, 16384> << <4, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    
    }
    else if (n == 8192)
    {
        CTBasedNTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // first iteration (multi-kernel)

        CTBasedNTTInnerSingle<2, 8192> << <2, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // rest of the iterations, single-kernel
    }
    else if (n == 4096)
    {
        CTBasedNTTInnerSingle<1, 4096> << <1, 1024, 4096 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // all iterations, single-kernel
    }
    else if (n == 2048)
    {
        CTBasedNTTInnerSingle<1, 2048> << <1, 1024, 2048 * sizeof(unsigned long long), stream1 >> > (device_a, q, mu, bit_length, psi_powers);  // all iterations, single-kernel
    }else {
        printf("NONONO3\n");
    }
}
__host__ void inverseNTT(unsigned long long* device_a, unsigned n, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psiinv_powers)
{
    if(n == 65536) 
    {
        GSBasedINTTInnerSingle<32, 65536> << <32, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 4 iterations
        
        GSBasedINTTInner<16, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 4th iter
        GSBasedINTTInner<8, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 3rd iter
        GSBasedINTTInner<4, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<2, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
        GSBasedINTTInner<1, 65536> << <65536 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter

    }
    else if (n == 32768)
    {
        GSBasedINTTInnerSingle<16, 32768> << <16, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 4 iterations
        
        GSBasedINTTInner<8, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 4th iter
        GSBasedINTTInner<4, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 3rd iter
        GSBasedINTTInner<2, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<1, 32768> << <32768 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
    }
    else if (n == 16384)
    {
        GSBasedINTTInnerSingle<8, 16384> << <8, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 3 iterations

        GSBasedINTTInner<4, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 3rd iter
        GSBasedINTTInner<2, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<1, 16384> << <16384 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
    }
    else if (n == 8192)
    {
        GSBasedINTTInnerSingle<4, 8192> << <4, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till last 2 iterations

        GSBasedINTTInner<2, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last 2nd iter
        GSBasedINTTInner<1, 8192> << <8192 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for last last iter
    }
    else if (n == 4096)
    {
        GSBasedINTTInnerSingle<2, 4096> << <2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel till the last iteration

        GSBasedINTTInner<1, 4096> << <4096 / 1024 / 2, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // multi-kernel for the last iter
    }
    else if (n == 2048)
    {
        GSBasedINTTInnerSingle<1, 2048> << <1, 1024, 0, stream1 >> > (device_a, q, mu, bit_length, psiinv_powers);  // single-kernel for all iterations
    }else {
        printf("NONONO4\n");
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInnerSingle_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_const[index];
    unsigned long long mu = mu_const[index];
    int qbit = qbit_const[index];

    register int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];
    }

#pragma unroll
    for (int length = l; length < n; length *= 2)
    {
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {

            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psi = psi_powers[length + psi_step + index * n];

            register unsigned long long first_target_value = shared_array[target_index];
            register uint128_t temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

            mul64(temp_storage.low, psi, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            register unsigned long long second_target_value = temp_storage.low;

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = target_result;

            first_target_value += q * (first_target_value < second_target_value);

            shared_array[target_index + step] = first_target_value - second_target_value;
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
    }

}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInnerSingle_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_const[index];
    unsigned long long mu = mu_const[index];
    int qbit = qbit_const[index];

    register int local_tid = threadIdx.x;

    __shared__ unsigned long long shared_array[2048];

    register unsigned long long q2 = (q + 1) >> 1;
    // if (threadIdx.x <= 32) {
    //     printf("%d,%lld\n",index,q_const[index]);
    // }
#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];
    }

    __syncthreads();

#pragma unroll
    for (int length = (n / 2); length >= l; length /= 2)
    {
        register int step = (n / length) / 2;

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
        {
            register int global_tid = local_tid + iteration_num * 1024;
            register int psi_step = global_tid / step;
            register int target_index = psi_step * step * 2 + global_tid % step;

            psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

            register unsigned long long psiinv = psiinv_powers[length + psi_step + index * n];

            register unsigned long long first_target_value = shared_array[target_index];
            register unsigned long long second_target_value = shared_array[target_index + step];

            register unsigned long long target_result = first_target_value + second_target_value;

            target_result -= q * (target_result >= q);

            shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

            first_target_value += q * (first_target_value < second_target_value);

            register uint128_t temp_storage = first_target_value - second_target_value;

            mul64(temp_storage.low, psiinv, temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);

            register unsigned long long temp_storage_low = temp_storage.low;

            shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        }

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
    {
        register int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
    }
}

template<unsigned l, unsigned n>
__global__ void CTBasedNTTInner_batch(unsigned long long a[], unsigned long long psi_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_const[index];
    unsigned long long mu = mu_const[index];
    int qbit = qbit_const[index];

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.y * n;

    register unsigned long long psi = psi_powers[length + psi_step + index * n];

    register unsigned long long first_target_value = a[target_index];
    register uint128_t temp_storage = a[target_index + step];

    mul64(temp_storage.low, psi, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);
    register unsigned long long second_target_value = temp_storage.low;

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;
}

template<unsigned l, unsigned n>
__global__ void GSBasedINTTInner_batch(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division)
{
    unsigned index = blockIdx.y % division;
    unsigned long long q = q_const[index];
    unsigned long long mu = mu_const[index];
    int qbit = qbit_const[index];

    int length = l;

    register int global_tid = blockIdx.x * 1024 + threadIdx.x;
    register int step = (n / length) / 2;
    register int psi_step = global_tid / step;
    register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.y * n;

    register unsigned long long psiinv = psiinv_powers[length + psi_step + index * n];

    register unsigned long long first_target_value = a[target_index];
    register unsigned long long second_target_value = a[target_index + step];

    register unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    register unsigned long long q2 = (q + 1) >> 1;

    target_result = (target_result >> 1) + q2 * (target_result & 1);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    register uint128_t temp_storage = first_target_value - second_target_value;

    mul64(temp_storage.low, psiinv, temp_storage);

    singleBarrett(temp_storage, q, mu, qbit);

    register unsigned long long temp_storage_low = temp_storage.low;

    temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

    a[target_index + step] = temp_storage_low;
}

__host__ void forwardNTT_batch(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers, unsigned num, unsigned division)
{
            forwardNTT3(device_a,n,psi_powers,num);
        return ;
    if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        forwardNTT3(device_a,n,psi_powers,num);
        return ;
        CTBasedNTTInner_batch<1, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInner_batch<2, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInner_batch<4, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInnerSingle_batch<8, 32768> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        forwardNTT3(device_a,n,psi_powers,num);
        return ;
        CTBasedNTTInner_batch<1, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInner_batch<2, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInnerSingle_batch<4, 16384> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 8192)
    {
        // fill<<<n*division/1024,1024>>>(device_a);
        forwardNTT3(device_a,n,psi_powers,num);
        return ;
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(2, num);
        CTBasedNTTInner_batch<1, 8192> << <multi_dim, 1024, 0, 0 >> > (device_a, psi_powers, division);

        CTBasedNTTInnerSingle_batch<2, 8192> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
        // print11<<<1,1>>>(device_a);
        // cudaDeviceSynchronize();
        // exit(0);
    }
    else if (n == 4096)
    {
        forwardNTT3(device_a,n,psi_powers,num);
        return ;
        dim3 single_dim(1, num);
        CTBasedNTTInnerSingle_batch<1, 4096> << <single_dim, 1024, 4096 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }
    else if (n == 2048)
    {
        forwardNTT3(device_a,n,psi_powers,num);
        return ;
        dim3 single_dim(1, num);
        CTBasedNTTInnerSingle_batch<1, 2048> << <single_dim, 1024, 2048 * sizeof(unsigned long long), 0 >> > (device_a, psi_powers, division);
    }else {
        printf("NONONO5\n");
    }
}
__host__ void forwardNTT_batch(cudaStream_t& stream1,unsigned long long* device_a, unsigned n, unsigned long long* psi_powers, unsigned num, unsigned division)
{
            forwardNTT3(stream1,device_a,n,psi_powers,num);
        return ;
    if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        forwardNTT3(stream1,device_a,n,psi_powers,num);
        return ;
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        forwardNTT3(stream1,device_a,n,psi_powers,num);
        return ;
    }
    else if (n == 8192)
    {
        // fill<<<n*division/1024,1024>>>(device_a);
        forwardNTT3(stream1,device_a,n,psi_powers,num);
        return ;
    }
    else if (n == 4096)
    {
        forwardNTT3(stream1,device_a,n,psi_powers,num);
        return ;
    }
    else if (n == 2048)
    {
        forwardNTT3(stream1,device_a,n,psi_powers,num);
        return ;
    }
    else {
        printf("NONONO6\n");
    }
}

__host__ void forwardNTT_batch_batch(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers, unsigned num, unsigned division,int moduleQplen)
{   
    if(n == 32768 * 2){
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        forwardNTT3(device_a,n,psi_powers,num,moduleQplen);
        return ;        
    }
    else if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        forwardNTT3(device_a,n,psi_powers,num,moduleQplen);
        return ;
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        forwardNTT3(device_a,n,psi_powers,num,moduleQplen);
        return ;
    }
    else if (n == 8192)
    {
        // fill<<<n*division/1024,1024>>>(device_a);
        forwardNTT3(device_a,n,psi_powers,num,moduleQplen);
        return ;
    }
    else if (n == 4096)
    {
        forwardNTT3(device_a,n,psi_powers,num,moduleQplen);
        return ;
    }
    else if (n == 2048)
    {
        forwardNTT3(device_a,n,psi_powers,num,moduleQplen);
        return ;
    }else {
        printf("NONONO7\n");
    }
}

__host__ void forwardNTT_batch_batch_fusion(unsigned long long* device_a,unsigned long long* device_b, unsigned n, unsigned long long* psi_powers, unsigned num, unsigned division,int moduleQplen,unsigned long long* a,unsigned long long* b,unsigned long long* suma,unsigned long long* sumb,bool isrot)
{   
    if(n == 32768 * 2){
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        forwardNTT3_fusion(device_a,device_b,n,psi_powers,num,moduleQplen,a,b,suma,sumb,isrot);
        return ;        
    }
    else if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        forwardNTT3_fusion(device_a,device_b,n,psi_powers,num,moduleQplen,a,b,suma,sumb,isrot);
        return ;
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        forwardNTT3_fusion(device_a,device_b,n,psi_powers,num,moduleQplen,a,b,suma,sumb,isrot);
        return ;
    }
    else if (n == 8192)
    {
        // fill<<<n*division/1024,1024>>>(device_a);
        forwardNTT3_fusion(device_a,device_b,n,psi_powers,num,moduleQplen,a,b,suma,sumb,isrot);
        return ;
    }
    else if (n == 4096)
    {
        forwardNTT3_fusion(device_a,device_b,n,psi_powers,num,moduleQplen,a,b,suma,sumb,isrot);
        return ;
    }
    else if (n == 2048)
    {
        forwardNTT3_fusion(device_a,device_b,n,psi_powers,num,moduleQplen,a,b,suma,sumb,isrot);
        return ;
    }else {
        printf("NONONO8\n");
    }
}

__host__ void inverseNTT_batch(unsigned long long* device_a, unsigned n, unsigned long long* psiinv_powers, unsigned num, unsigned division)
{   
    if(n == 32768 * 2){
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(32, num);

        GSBasedINTTInnerSingle_batch<32, 32768*2> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<16, 32768*2> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<8, 32768*2> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<4, 32768*2> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<2, 32768*2> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 32768*2> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division); 
    }
    else if (n == 32768)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(16, num);
        // forwardNTT3(device_a,n,psiinv_powers,num);
        // return ;
        GSBasedINTTInnerSingle_batch<16, 32768> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<8, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<4, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<2, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 32768> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 16384)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(8, num);
        // forwardNTT3(device_a,n,psiinv_powers,num);
        // return ;
        GSBasedINTTInnerSingle_batch<8, 16384> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<4, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<2, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 16384> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 8192)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(4, num);
        // forwardNTT3(device_a,n,psiinv_powers,num);
        // return ;
        GSBasedINTTInnerSingle_batch<4, 8192> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<2, 8192> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
        GSBasedINTTInner_batch<1, 8192> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 4096)
    {
        dim3 multi_dim(n / 1024 / 2, num);
        dim3 single_dim(2, num);
        // forwardNTT3(device_a,n,psiinv_powers,num);
        // return ;
        GSBasedINTTInnerSingle_batch<2, 4096> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);

        GSBasedINTTInner_batch<1, 4096> << <multi_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }
    else if (n == 2048)
    {
        // forwardNTT3(device_a,n,psiinv_powers,num);
        // return ;
        dim3 single_dim(1, num);
        GSBasedINTTInnerSingle_batch<1, 2048> << <single_dim, 1024, 0, 0 >> > (device_a, psiinv_powers, division);
    }else {
        printf("NONONO9\n");
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// explicit template instantiations
// all these are required for the program to compile

// n = 2048
template __global__ void CTBasedNTTInnerSingle<1, 2048>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInnerSingle<1, 2048>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 4096
template __global__ void CTBasedNTTInnerSingle<1, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<2, 4096>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 8192
template __global__ void CTBasedNTTInner<1, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<2, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<4, 8192>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 16384
template __global__ void CTBasedNTTInner<1, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<2, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<4, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<8, 16384>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// n = 32768
template __global__ void CTBasedNTTInner<1, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<2, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInner<4, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void CTBasedNTTInnerSingle<8, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[]);
template __global__ void GSBasedINTTInner<1, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<2, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<4, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInner<8, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);
template __global__ void GSBasedINTTInnerSingle<16, 32768>(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[]);

// --------------------------------------------------------------------------------------------------------------------------------------------------------
// explicit template instantiations for batch ntt
// all these are required for the program to compile

// n = 2048
template __global__ void CTBasedNTTInnerSingle_batch<1, 2048>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<1, 2048>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 4096
template __global__ void CTBasedNTTInnerSingle_batch<1, 4096>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 4096>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<2, 4096>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 8192
template __global__ void CTBasedNTTInner_batch<1, 8192>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInnerSingle_batch<2, 8192>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 8192>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<2, 8192>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<4, 8192>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 16384
template __global__ void CTBasedNTTInner_batch<1, 16384>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInner_batch<2, 16384>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInnerSingle_batch<4, 16384>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<2, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<4, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<8, 16384>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);

// n = 32768
template __global__ void CTBasedNTTInner_batch<1, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInner_batch<2, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInner_batch<4, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void CTBasedNTTInnerSingle_batch<8, 32768>(unsigned long long a[], unsigned long long psi_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<1, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<2, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<4, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInner_batch<8, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);
template __global__ void GSBasedINTTInnerSingle_batch<16, 32768>(unsigned long long a[], unsigned long long psiinv_powers[], unsigned division);





// --------------------------------------------------------------------------------------------------------------------------------------------------------


template <typename T>
__device__ __forceinline__ void myswap(T &a, T &b){ T s = a;  a = b; b = s;}
template <typename T, int s>
__device__ __forceinline__ void mymove(T (&u)[8]){
  const int s1 = 2*s;
  // step 1:
  if (!(threadIdx.x&s)) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++){
      int i1 = i%s;
      int i2 = i/s;
      int i3 = i2*s1;
      myswap(u[i3+i1], u[i3+i1+s]);}}
  // step 2:
  #pragma unroll 4
  for (int i = 0; i < 4; i++){
    int i1 = i%s;
    int i2 = i/s;
    int i3 = i2*s1;
    u[i3+i1] = __shfl_xor_sync(0xFFFFFFFF, u[i3+i1], s);}
  // step 3:
  if (!(threadIdx.x&s)) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++){
      int i1 = i%s;
      int i2 = i/s;
      int i3 = i2*s1;
      myswap(u[i3+i1], u[i3+i1+s]);}}
}

template <typename T>
__device__ __forceinline__ void myswap1(T &a, T &b){ T s = a;  a = b; b = s;}
template <typename T, int s>
__device__ __forceinline__ void mymove1(T (&u)[8]){
  const int s1 = 2*s;
  // step 1:
  if (!(threadIdx.x&(4 * s))) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++){
      int i1 = i%s;
      int i2 = i/s;
      int i3 = i2*s1;
      myswap1(u[i3+i1], u[i3+i1+s]);}}
  // step 2:
  #pragma unroll 4
  for (int i = 0; i < 4; i++){
    int i1 = i%s;
    int i2 = i/s;
    int i3 = i2*s1;
    u[i3+i1] = __shfl_xor_sync(0xFFFFFFFF, u[i3+i1], (4 * s));}
  // step 3:
  if (!(threadIdx.x&(4 * s))) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++){
      int i1 = i%s;
      int i2 = i/s;
      int i3 = i2*s1;
      myswap1(u[i3+i1], u[i3+i1+s]);}}
}

template <typename T>
__device__ __forceinline__ void myswap2(T &a, T &b){ T s = a;  a = b; b = s;}
template <typename T, int s>
__device__ __forceinline__ void mymove2(T (&u)[8]){
  const int s1 = 2*s;
  // step 1:
  if (!(threadIdx.x&(2 * s))) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++){
      int i1 = i%s;
      int i2 = i/s;
      int i3 = i2*s1;
      myswap1(u[i3+i1], u[i3+i1+s]);}}
  // step 2:
  #pragma unroll 4
  for (int i = 0; i < 4; i++){
    int i1 = i%s;
    int i2 = i/s;
    int i3 = i2*s1;
    u[i3+i1] = __shfl_xor_sync(0xFFFFFFFF, u[i3+i1], (2 * s));}
  // step 3:
  if (!(threadIdx.x&(2 * s))) {
    #pragma unroll 4
    for (int i = 0; i < 4; i++){
      int i1 = i%s;
      int i2 = i/s;
      int i3 = i2*s1;
      myswap1(u[i3+i1], u[i3+i1+s]);}}
}



const int reg_num = 8;


template <int n, int nlog>
__global__ void ntt_forward_reg_first3(unsigned long long tmp[], unsigned long long arr[], unsigned long long psiTable[], int moduleQplen) {

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    const int warpSize = 32;
    int step1 = n / reg_num / warpSize;
    int step2 = n / reg_num;


    gid &= step2-1;
    unsigned long long reg[reg_num];


    for(int ii = 0; ii < moduleQplen; ii++){
        unsigned long long q = q_const[ii];
        unsigned long long mu = mu_const[ii];
        int qbit = qbit_const[ii];  


        for(int i = 0; i < reg_num; i++){
            reg[i] = arr[i * step2 + (gid & 31) * step1 + (gid >> 5) + gid / step2 * n / reg_num + n * blockIdx.y];
            if(reg[i] > q){
                reg[i] -= q;
            }
        }
        int step = reg_num;

        int gstepidx = nlog;
        #pragma unroll
        for(int l = 0; l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){

                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                
                int gloc = loc * step2 + (gid & 31) * step1 + (gid >> 5);

                unsigned long long psi = psiTable[(1 << l) + (gloc >> (gstepidx + 1)) + ii * n];
                unsigned long long U = reg[loc];
                // if(reg[loc] == 46365438){
                //     printf("&&%d,%d,%d,%d,%d,%d,%d\n",l,ii,blockIdx.y,gid,bf_idx,l,blockIdx.x);
                // }
                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }

        }

        mymove1<unsigned long long, 4>(reg);
        mymove1<unsigned long long, 2>(reg);
        mymove1<unsigned long long, 1>(reg);


        step = reg_num;
        #pragma unroll
        for(int l = 0; l < 3; l++) {
            step >>= 1;
            gstepidx--;

            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){

                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));

                int gloc = loc * (step2/8) + (gid & 3) * step1 + (gid >> 5) + ((gid & 31) >> 2) * step2;

                unsigned long long psi = psiTable[(8 << l) + (gloc >> (gstepidx + 1)) + ii * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
            // for(int i = 0; i < reg_num; i++){
            //     if(reg[i] == 549070729){
            //         printf("!!%d,%d,%d,%d,%d\n",l,i,gid,blockIdx.x,blockIdx.y);
            //     }
            // }
        }

        mymove<unsigned long long, 2>(reg);
        mymove<unsigned long long, 1>(reg);

        step = reg_num/2;
        #pragma unroll
        for(int l = 0; l < 2; l++) {
            step >>= 1;
            gstepidx--;

            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){

                int loc = (bf_idx >> (2 - l - 1)) * 2 * step + (bf_idx & (step - 1));

                int gloc = (loc >> 2) * n / (4096 / 256) + (loc & (3)) * n / (4096 / 16) + (gid & 3) * (n / (4096 / 64)) + ((gid & 31) >> 2) * (n / (4096 / 512)) + (gid >> 5);

                unsigned long long psi = psiTable[(64 << l) + (gloc >> (gstepidx + 1)) + ii * n];
                unsigned long long U = reg[loc];
                
                uint128_t temp_storage = reg[loc + step];  

                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }

        }

        #pragma unroll
        for(int i = 0; i < reg_num; i++){
            // if(reg[i] == 1010783535){
            //     printf("^^%d,%d,%d,%d\n",i,gid,blockIdx.x,blockIdx.y);
            // }
            tmp[i * n / reg_num + gid + gid / step2 * n / reg_num + n * ii + moduleQplen * n * blockIdx.y] = reg[i] ;
        }
    }
    
}



template <int n, int r,int nlog>
__global__ void ntt_forward_reg_second121(unsigned long long tmp[],unsigned long long psiTable[],int moduleQplen,unsigned long long* keya,unsigned long long* keyb,unsigned long long* suma,unsigned long long* sumb,bool isrot){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    int step1 = n / reg_num / reg_num;
    int step2 = n / reg_num;
    gid %= step2;
    
    int moduleidx = blockIdx.y % moduleQplen;


    unsigned long long q = q_const[blockIdx.y];
    unsigned long long mu = mu_const[blockIdx.y];
    int qbit = qbit_const[blockIdx.y];  
    unsigned long long ca[reg_num];
    unsigned long long cb[reg_num];
    for(int i = 0; i < reg_num; i++){
        ca[i] = 0;
        cb[i] = 0;
    }
    for(int pcnt = 0; pcnt < moduleQplen - 1; pcnt++){
        unsigned long long reg[reg_num];
        int gstep = r;

        for(int i = 0; i < reg_num; i++){

            int nloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;
            int t = n / 256;
            int nlocx = nloc / t;
            int nloczz = nloc % t;

            int nlocxx = nlocx % 4 + nlocx % 32 / 16 * 4;
            int nlocyy = (nlocx % 16)/ 4 + (nlocx / 32) * 4 + 32 * nloczz ;

            int loc = nlocxx * n / reg_num + nlocyy ;

            reg[i] = tmp[gid / step2 * n / reg_num + loc + pcnt * moduleQplen * n + blockIdx.y * n];
            // if(reg[i] == 1010783535){
            //     printf("$$%d,%d,%d,%d,%d\n",i,pcnt,gid,blockIdx.x,blockIdx.y);
            // }
        }




        int gstepidx = nlog - 8;
        int step = 8;
        #pragma unroll
        for(int l = 0; l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                
                int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 

                unsigned long long psi = psiTable[(256 << l) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
            // for(int i = 0; i < reg_num; i++){
            //     if(reg[i] == 738988121){
            //         printf("**%d,%d,%d,%d,%d\n",l,i,gid,blockIdx.x,blockIdx.y);
            //     }
            // }
        }

        if(nlog == 15){
            step = n >> 11;
            #pragma unroll
            for(int l = 0; l < 16 - nlog; l++) {
                step >>= 1;
                gstepidx--;
                #pragma unroll
                for(int loc = 0; loc < reg_num; loc++){
                    int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 

                    unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                    unsigned long long U = reg[loc];
                    uint128_t temp_storage = __shfl_xor_sync(0xFFFFFFFF, reg[loc], step);


                    unsigned long long V;
                    if(!(gid & step)){
                        mul64(temp_storage.low, psi, temp_storage);
                        singleBarrett(temp_storage, q, mu, qbit);
                        V = temp_storage.low;
                        reg[loc] = U + V;
                        if(reg[loc] > q){
                            reg[loc] -= q;
                        }
                    }

                    
                    V = __shfl_xor_sync(0xFFFFFFFF, q + U - V, step);
                    if((gid & step)){
                        if(V > q){
                            reg[loc] = V - q;
                        }
                        else{
                            reg[loc] = V;
                        }
                    }

                }
            }
        }

        if(nlog == 16){
            step = n >> 11;
            #pragma unroll
            for(int l = 0; l < 2; l++) {
                step >>= 1;
                gstepidx--;
                #pragma unroll
                for(int loc = 0; loc < reg_num; loc++){
                    int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 
                    // int gstep = step * 
                    unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + + moduleidx * n];
                    unsigned long long U = reg[loc];
                    uint128_t temp_storage = __shfl_xor_sync(0xFFFFFFFF, reg[loc], step);
                    // if(gid == 0){
                    //     printf("^^^%4d,%4d,%4lld,%4lld,%4d,%4d,%4d,%4d\n",loc,loc+step,reg[loc],temp_storage.low,gloc,gloc + step,(n >> (gstepidx + 1)),(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)));
                    // }

                    unsigned long long V;
                    if(!(gid & step)){
                        mul64(temp_storage.low, psi, temp_storage);
                        singleBarrett(temp_storage, q, mu, qbit);
                        V = temp_storage.low;
                        reg[loc] = U + V;
                        if(reg[loc] > q){
                            reg[loc] -= q;
                        }
                    }

                    
                    V = __shfl_xor_sync(0xFFFFFFFF, q + U - V, step);
                    if((gid & step)){
                        if(V > q){
                            reg[loc] = V - q;
                        }
                        else{
                            reg[loc] = V;
                        }
                    }

                }
            }
        }


        mymove<unsigned long long, 1>(reg);
        if(nlog >= 13)mymove<unsigned long long, 2>(reg);
        if(nlog >= 14)mymove<unsigned long long, 4>(reg);


        
        step = 8 >> (3 - min(gstepidx, 3));
        #pragma unroll
        for(int l = (3 - min(gstepidx, 3)); l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                
                int gloc = gid % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 
                if(nlog == 15){
                    gloc = gid % 16 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 
                }
                if(nlog == 16){
                    gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 
                }
                unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
            // for(int i = 0; i < reg_num; i++){
            //     if(reg[i] == 524155228){
            //         printf("**%d,%d,%d,%d,%d\n",l,i,gid,blockIdx.x,blockIdx.y);
            //     }
            // }
        }

        mymove<unsigned long long, 1>(reg);
        if(nlog >= 13)mymove<unsigned long long, 2>(reg);
        if(nlog >= 14)mymove<unsigned long long, 4>(reg);
        
        for(int i = 0; i < reg_num; i++){

            int gloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;

            int sumnum = moduleQplen - 1;
            register unsigned long long ra1 = reg[i];
            // if(ra1 == 206999613){
            //     printf("%d,%d,%d\n",pcnt,blockIdx.y,gid);
            // }
            if(isrot && ra1) ra1 = q - ra1; 

            unsigned long long keyadata = keya[gloc + gid / step2 * n / reg_num + pcnt * (sumnum * 2) * n + blockIdx.y * n];
            unsigned long long keybdatb = keyb[gloc + gid / step2 * n / reg_num + pcnt * (sumnum * 2) * n + blockIdx.y * n];

            register unsigned long long rb1 = keyadata;
            register unsigned long long rb2 = keybdatb;
            uint128_t rc1, rx1;

            mul64(ra1, rb1, rc1);

            rx1 = rc1 >> (qbit - 2);

            mul64(rx1.low, mu, rx1);

            uint128_t::shiftr(rx1, qbit + 2);

            mul64(rx1.low, q, rx1);

            sub128(rc1, rx1);
            if (rc1.low < q)
                ca[i] += rc1.low;
            else
                ca[i] += rc1.low - q;
            if(ca[i] >= q){
                ca[i] -= q;
            }

            register unsigned long long ra2 = ra1;

            uint128_t rc2, rx2;

            mul64(ra2, rb2, rc2);

            rx2 = rc2 >> (qbit - 2);

            mul64(rx2.low, mu, rx2);

            uint128_t::shiftr(rx2, qbit + 2);

            mul64(rx2.low, q, rx2);

            sub128(rc2, rx2);
            if (rc2.low < q)
                cb[i] += rc2.low;
            else
                cb[i] += rc2.low - q;

            if(cb[i] >= q){
                cb[i] -= q;
            }          
        }
    }
    for(int i = 0; i < reg_num; i++){
        int gloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;

        suma[gloc + gid / step2 * n / reg_num + blockIdx.y * n] = ca[i];
        sumb[gloc + gid / step2 * n / reg_num + blockIdx.y * n] = cb[i];
    }
}

template <int n, int r,int nlog>
__global__ void ntt_forward_reg_second12(unsigned long long tmp[],unsigned long long psiTable[],int moduleQplen,unsigned long long* keya,unsigned long long* keyb,unsigned long long* suma,unsigned long long* sumb,bool isrot){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    int step1 = n / reg_num / reg_num;
    int step2 = n / reg_num;
    gid %= step2;
    
    int moduleidx = blockIdx.y % moduleQplen;


    unsigned long long q = q_const[blockIdx.y];
    unsigned long long mu = mu_const[blockIdx.y];
    int qbit = qbit_const[blockIdx.y];  
    unsigned long long ca[reg_num];
    unsigned long long cb[reg_num];
    for(int i = 0; i < reg_num; i++){
        ca[i] = 0;
        cb[i] = 0;
    }
    for(int pcnt = 0; pcnt < moduleQplen - 1; pcnt++){
        unsigned long long reg[reg_num];
        int gstep = r;

        for(int i = 0; i < reg_num; i++){

            int nloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;
            int t = n / 256;
            int nlocx = nloc / t;
            int nloczz = nloc % t;

            int nlocxx = nlocx % 4 + nlocx % 32 / 16 * 4;
            int nlocyy = (nlocx % 16)/ 4 + (nlocx / 32) * 4 + 32 * nloczz ;

            int loc = nlocxx * n / reg_num + nlocyy ;

            reg[i] = tmp[gid / step2 * n / reg_num + loc + pcnt * moduleQplen * n + blockIdx.y * n];
            // if(reg[i] == 1010783535){
            //     printf("$$%d,%d,%d,%d,%d\n",i,pcnt,gid,blockIdx.x,blockIdx.y);
            // }
        }




        int gstepidx = nlog - 8;
        int step = 8;
        #pragma unroll
        for(int l = 0; l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                
                int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 

                unsigned long long psi = psiTable[(256 << l) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
            // for(int i = 0; i < reg_num; i++){
            //     if(reg[i] == 738988121){
            //         printf("**%d,%d,%d,%d,%d\n",l,i,gid,blockIdx.x,blockIdx.y);
            //     }
            // }
        }
        if(nlog == 16){
            mymove1<unsigned long long, 1>(reg);
            mymove1<unsigned long long, 2>(reg);
            mymove1<unsigned long long, 4>(reg);
        }

        if(nlog == 15){
            mymove1<unsigned long long, 1>(reg);
            mymove1<unsigned long long, 2>(reg);
        }
    if(nlog == 15){
        step = 4;
        #pragma unroll
        for(int l = 0; l < 2; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3-l-2)) * 2 * step + (bf_idx & (step - 1));
                // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 

                // int gloc = loc * 4 + gid % 32 % 4 + gid % 32 / 4 * 32 + gid / 32 * 256;
                // int gloc = loc % 2 * 16 + loc / 4 * 64 + gid % 32 % 8 + gid % 16 / 8 * 32 + gid % 32 / 16 * 128 + gid / 32 * 256;
                int gloc = loc % 4 * 4 + loc / 4 * 64 + gid % 32 % 4 + gid % 16 / 4 * 16 + gid % 32 / 16 * 128 + gid / 32 * 256;
                // int gstep = step * 
                unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
                // if(gloc == 1 && ggid == gid){
                //     printf("%d,%lld,%d,%d\n",l,reg[loc],loc,gid);
                // }
            }
        }
    }
    if(nlog == 16){
        step = 8;
        #pragma unroll
        for(int l = 0; l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 

                int gloc = loc * 4 + gid % 32 % 4 + gid % 32 / 4 * 32 + gid / 32 * 256;
                // int gstep = step * 
                unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
        }
    }
        // if(nlog == 15){
        //     step = n >> 11;
        //     #pragma unroll
        //     for(int l = 0; l < 16 - nlog; l++) {
        //         step >>= 1;
        //         gstepidx--;
        //         #pragma unroll
        //         for(int loc = 0; loc < reg_num; loc++){
        //             int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 

        //             unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + moduleidx * n];
        //             unsigned long long U = reg[loc];
        //             uint128_t temp_storage = __shfl_xor_sync(0xFFFFFFFF, reg[loc], step);


        //             unsigned long long V;
        //             if(!(gid & step)){
        //                 mul64(temp_storage.low, psi, temp_storage);
        //                 singleBarrett(temp_storage, q, mu, qbit);
        //                 V = temp_storage.low;
        //                 reg[loc] = U + V;
        //                 if(reg[loc] > q){
        //                     reg[loc] -= q;
        //                 }
        //             }

                    
        //             V = __shfl_xor_sync(0xFFFFFFFF, q + U - V, step);
        //             if((gid & step)){
        //                 if(V > q){
        //                     reg[loc] = V - q;
        //                 }
        //                 else{
        //                     reg[loc] = V;
        //                 }
        //             }

        //         }
        //     }
        // }

        // if(nlog == 16){
        //     step = n >> 11;
        //     #pragma unroll
        //     for(int l = 0; l < 2; l++) {
        //         step >>= 1;
        //         gstepidx--;
        //         #pragma unroll
        //         for(int loc = 0; loc < reg_num; loc++){
        //             int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 
        //             // int gstep = step * 
        //             unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + + moduleidx * n];
        //             unsigned long long U = reg[loc];
        //             uint128_t temp_storage = __shfl_xor_sync(0xFFFFFFFF, reg[loc], step);
        //             // if(gid == 0){
        //             //     printf("^^^%4d,%4d,%4lld,%4lld,%4d,%4d,%4d,%4d\n",loc,loc+step,reg[loc],temp_storage.low,gloc,gloc + step,(n >> (gstepidx + 1)),(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)));
        //             // }

        //             unsigned long long V;
        //             if(!(gid & step)){
        //                 mul64(temp_storage.low, psi, temp_storage);
        //                 singleBarrett(temp_storage, q, mu, qbit);
        //                 V = temp_storage.low;
        //                 reg[loc] = U + V;
        //                 if(reg[loc] > q){
        //                     reg[loc] -= q;
        //                 }
        //             }

                    
        //             V = __shfl_xor_sync(0xFFFFFFFF, q + U - V, step);
        //             if((gid & step)){
        //                 if(V > q){
        //                     reg[loc] = V - q;
        //                 }
        //                 else{
        //                     reg[loc] = V;
        //                 }
        //             }

        //         }
        //     }
        // }

    if(nlog == 16){
    mymove<unsigned long long, 1>(reg);
    mymove<unsigned long long, 2>(reg);

    }

    if(nlog == 15){

        mymove<unsigned long long, 1>(reg);
        mymove<unsigned long long, 2>(reg);

    }
    if(nlog <= 14){

        mymove<unsigned long long, 1>(reg);
        if(nlog >= 13)mymove<unsigned long long, 2>(reg);
        if(nlog >= 14)mymove<unsigned long long, 4>(reg);
    }



        
        step = 8 >> (3 - min(gstepidx, 3));
        #pragma unroll
        for(int l = (3 - min(gstepidx, 3)); l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                
                int gloc = gid % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 
                // if(nlog == 15){
                //     gloc = gid % 16 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 
                // }
                // if(nlog == 16){
                //     gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 
                // }
                if(nlog == 15){
                    // gloc = gid % 16 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 

                    gloc = loc % 4 + loc / 4 * 64 + gid % 32 % 16 * 4 + gid % 32 / 16 * 128 + gid / 32 * 256; 

                }
                if(nlog == 16){
                    // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 
                    gloc = loc % 4 + loc / 4 * 16 + gid % 32 % 4 * 4 +  gid % 32 / 4 * 32 + gid / 32 * 256;

                }
                unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1)) + moduleidx * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
            // for(int i = 0; i < reg_num; i++){
            //     if(reg[i] == 524155228){
            //         printf("**%d,%d,%d,%d,%d\n",l,i,gid,blockIdx.x,blockIdx.y);
            //     }
            // }
        }

        // mymove<unsigned long long, 1>(reg);
        // if(nlog >= 13)mymove<unsigned long long, 2>(reg);
        // if(nlog >= 14)mymove<unsigned long long, 4>(reg);
        
        for(int i = 0; i < reg_num; i++){

            // int gloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;
            int gloc = gid % (r / reg_num) * r / reg_num+ i / (r / 8 ) * (r / 8) * (r / 8) +  i % (r / 8 ) + gid / (r / reg_num) * r  ; 
                if(nlog == 15){
                    // gloc = gid % 16 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 

                    gloc = i % 4 + i / 4 * 64 + gid % 32 % 16 * 4 + gid % 32 / 16 * 128 + gid / 32 * 256; 

                }
                if(nlog == 16){
                    // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 
                    gloc = i % 4 + i / 4 * 16 + gid % 32 % 4 * 4 +  gid % 32 / 4 * 32 + gid / 32 * 256;

                }
            int sumnum = moduleQplen - 1;
            register unsigned long long ra1 = reg[i];
            // if(ra1 == 206999613){
            //     printf("%d,%d,%d\n",pcnt,blockIdx.y,gid);
            // }
            if(isrot && ra1) ra1 = q - ra1; 

            unsigned long long keyadata = keya[gloc + gid / step2 * n / reg_num + pcnt * (sumnum * 2) * n + blockIdx.y * n];
            unsigned long long keybdatb = keyb[gloc + gid / step2 * n / reg_num + pcnt * (sumnum * 2) * n + blockIdx.y * n];

            register unsigned long long rb1 = keyadata;
            register unsigned long long rb2 = keybdatb;
            uint128_t rc1, rx1;

            mul64(ra1, rb1, rc1);

            rx1 = rc1 >> (qbit - 2);

            mul64(rx1.low, mu, rx1);

            uint128_t::shiftr(rx1, qbit + 2);

            mul64(rx1.low, q, rx1);

            sub128(rc1, rx1);
            if (rc1.low < q)
                ca[i] += rc1.low;
            else
                ca[i] += rc1.low - q;
            if(ca[i] >= q){
                ca[i] -= q;
            }

            register unsigned long long ra2 = ra1;

            uint128_t rc2, rx2;

            mul64(ra2, rb2, rc2);

            rx2 = rc2 >> (qbit - 2);

            mul64(rx2.low, mu, rx2);

            uint128_t::shiftr(rx2, qbit + 2);

            mul64(rx2.low, q, rx2);

            sub128(rc2, rx2);
            if (rc2.low < q)
                cb[i] += rc2.low;
            else
                cb[i] += rc2.low - q;

            if(cb[i] >= q){
                cb[i] -= q;
            }          
        }
    }
    for(int i = 0; i < reg_num; i++){
        // int gloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;
        int gloc = gid % (r / reg_num) * r / reg_num+ i / (r / 8 ) * (r / 8) * (r / 8) +  i % (r / 8 ) + gid / (r / reg_num) * r  ; 
        if(nlog == 15){
            // gloc = gid % 16 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 

            gloc = i % 4 + i / 4 * 64 + gid % 32 % 16 * 4 + gid % 32 / 16 * 128 + gid / 32 * 256; 

        }
        if(nlog == 16){
            // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 
            gloc = i % 4 + i / 4 * 16 + gid % 32 % 4 * 4 +  gid % 32 / 4 * 32 + gid / 32 * 256;

        }
        suma[gloc + gid / step2 * n / reg_num + blockIdx.y * n] = ca[i];
        sumb[gloc + gid / step2 * n / reg_num + blockIdx.y * n] = cb[i];
    }
}

__host__ void forwardNTT8(unsigned long long* device_b, unsigned long long* device_a, unsigned n, unsigned long long* psi_powers, int foo, int bar, int moduleQplen, unsigned long long* keya, unsigned long long* keyb,unsigned long long* suma, unsigned long long* sumb, bool isrot){

    if(n == 4096){
        const int l = 12;
        dim3 dim1(1 * n / 128 / reg_num,moduleQplen-1);
        dim3 dim2(1 * n / 128 / reg_num,moduleQplen);
        ntt_forward_reg_first3<1 << l,l><<<dim1, 128>>>(device_b, device_a, psi_powers, moduleQplen);
        ntt_forward_reg_second12<1 << l, 1 << (l - 8) ,l><<<dim2, 128>>>(device_b, psi_powers, moduleQplen,keya,keyb,suma,sumb,isrot);
    }
    else if(n == 8192){
        const int l = 13;
        dim3 dim1(1 * n / 128 / reg_num,moduleQplen-1);
        dim3 dim2(1 * n / 128 / reg_num,moduleQplen);
        ntt_forward_reg_first3<1 << l,l><<<dim1, 128>>>(device_b, device_a, psi_powers, moduleQplen);
        ntt_forward_reg_second12<1 << l, 1 << (l - 8) ,l><<<dim2, 128>>>(device_b, psi_powers, moduleQplen,keya,keyb,suma,sumb,isrot);

        // const int sizeScale = 4;
        // // dim3 dim1(64,moduleQplen-1);
        // dim3 dim2(32*sizeScale,moduleQplen);
        // // firstStep3i<1,2048*sizeScale,64><<<dim1, 64, 128 * sizeof(unsigned long long)>>>(device_a,device_b,psi_powers,moduleQplen);
        // secondStep3i<32*sizeScale,2048*sizeScale><<<dim2,32>>>(device_b,psi_powers,moduleQplen,keya,keyb,suma,sumb,isrot);

    }
    else if(n == 16384){
        const int l = 14;
        dim3 dim1(1 * n / 128 / reg_num,moduleQplen-1);
        dim3 dim2(1 * n / 128 / reg_num,moduleQplen);
        ntt_forward_reg_first3<1 << l,l><<<dim1, 128>>>(device_b, device_a, psi_powers, moduleQplen);
        ntt_forward_reg_second12<1 << l, 1 << (l - 8) ,l><<<dim2, 128>>>(device_b, psi_powers, moduleQplen,keya,keyb,suma,sumb,isrot); 
    } 
    else if(n == 16384 * 2){
        const int l = 15;
        dim3 dim1(1 * n / 128 / reg_num,moduleQplen-1);
        dim3 dim2(1 * n / 128 / reg_num,moduleQplen);
        ntt_forward_reg_first3<1 << l,l><<<dim1, 128>>>(device_b, device_a, psi_powers, moduleQplen);
        ntt_forward_reg_second12<1 << l, 1 << (l - 8) ,l><<<dim2, 128>>>(device_b, psi_powers, moduleQplen,keya,keyb,suma,sumb,isrot);
    }
    else if(n == 16384 * 4){
        const int l = 16;
        dim3 dim1(1 * n / 128 / reg_num,moduleQplen-1);
        dim3 dim2(1 * n / 128 / reg_num,moduleQplen);
        ntt_forward_reg_first3<1 << l,l><<<dim1, 128>>>(device_b, device_a, psi_powers, moduleQplen);
        ntt_forward_reg_second12<1 << l, 1 << (l - 8) ,l><<<dim2, 128>>>(device_b, psi_powers, moduleQplen,keya,keyb,suma,sumb,isrot);
    }
    else{
        printf("%d\n",n);
        throw "error";
    }
}



template <int n, int nlog>
__global__ void ntt_forward_reg_first3(unsigned long long arr[], unsigned long long tmp[],unsigned long long psiTable[])  {
    // return ;
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int ggid = blockDim.x * blockIdx.x + tid;
    // int mid = gid % (n / reg_num);
    const int warpSize = 32;
    int step1 = n / reg_num / warpSize;
    int step2 = n / reg_num;
    // gid %= step2;
    gid &= step2-1;
    unsigned long long q = q_const[ggid / step2];
    unsigned long long mu = mu_const[ggid / step2];
    int qbit = qbit_const[ggid / step2];  
    unsigned long long reg[reg_num];

    for(int i = 0; i < reg_num; i++){
        // reg[i] = arr[i * step2 + gid % warpSize * step1 + gid / warpSize];
        reg[i] = arr[i * step2 + (gid & 31) * step1 + (gid >> 5) + gid / step2 * n / reg_num + ggid / step2 * n ];

    }

    int step = reg_num;

    int gstepidx = nlog;
    #pragma unroll
    for(int l = 0; l < 3; l++) {
        step >>= 1;
        gstepidx--;
        #pragma unroll
        for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
            // int loc = bf_idx / (step) * 2 * step + bf_idx % step;
            int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
            
            // int gloc = loc * step2 + gid % warpSize * step1 + gid / warpSize;
            int gloc = loc * step2 + (gid & 31) * step1 + (gid >> 5);

            unsigned long long psi = psiTable[(1 << l) + (gloc >> (gstepidx + 1))];
            unsigned long long U = reg[loc];

            uint128_t temp_storage = reg[loc + step];  
            mul64(temp_storage.low, psi, temp_storage);
            singleBarrett(temp_storage, q, mu, qbit);
            unsigned long long V = temp_storage.low;
            reg[loc] = (U + V) ;
            if(reg[loc] > q){
                reg[loc] -= q;
            }
            reg[loc + step] = (q + U - V) ;
            if(reg[loc + step] > q){
                reg[loc + step] -= q;
            }
        }

    }
    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 int nloc = i * 32 + j ;
    //                 reg[i] = nloc;

    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }







    mymove1<unsigned long long, 1>(reg);
    mymove1<unsigned long long, 2>(reg);
    mymove1<unsigned long long, 4>(reg);
    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }

    step = reg_num;
    #pragma unroll
    for(int l = 0; l < 3; l++) {
        step >>= 1;
        gstepidx--;

        #pragma unroll
        for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
            int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));

            int gloc = loc * (step2/8) + (gid & 3) * step1 + (gid >> 5) + ((gid & 31) >> 2) * step2;

            unsigned long long psi = psiTable[(8 << l) + (gloc >> (gstepidx + 1)) + ggid / step2 * n ];
            unsigned long long U = reg[loc];

            uint128_t temp_storage = reg[loc + step];  
            mul64(temp_storage.low, psi, temp_storage);
            singleBarrett(temp_storage, q, mu, qbit);
            unsigned long long V = temp_storage.low;
            reg[loc] = (U + V) ;
            if(reg[loc] > q){
                reg[loc] -= q;
            }
            reg[loc + step] = (q + U - V) ;
            if(reg[loc + step] > q){
                reg[loc + step] -= q;
            }
        }

    }

    mymove<unsigned long long, 2>(reg);
    mymove<unsigned long long, 1>(reg);

    step = reg_num/2;
    #pragma unroll
    for(int l = 0; l < 2; l++) {
        step >>= 1;
        gstepidx--;

        #pragma unroll
        for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
            int loc = (bf_idx >> (2 - l - 1)) * 2 * step + (bf_idx & (step - 1));

            int gloc = (loc >> 2) * n / (4096 / 256) + (loc & (3)) * n / (4096 / 16) + (gid & 3) * (n / (4096 / 64)) + ((gid & 31) >> 2) * (n / (4096 / 512)) + (gid >> 5);

            unsigned long long psi = psiTable[(64 << l) + (gloc >> (gstepidx + 1))+ ggid / step2 * n];
            unsigned long long U = reg[loc];
            
            uint128_t temp_storage = reg[loc + step];  

            mul64(temp_storage.low, psi, temp_storage);
            singleBarrett(temp_storage, q, mu, qbit);
            unsigned long long V = temp_storage.low;
            reg[loc] = (U + V) ;
            if(reg[loc] > q){
                reg[loc] -= q;
            }
            reg[loc + step] = (q + U - V) ;
            if(reg[loc + step] > q){
                reg[loc + step] -= q;
            }
        }

    }


    #pragma unroll
    for(int i = 0; i < reg_num; i++){
        tmp[i * n / reg_num + gid + gid / step2 * n / reg_num + ggid / step2 * n ] = reg[i] ;

    }
}




template <int n, int r,int nlog>
__global__ void ntt_forward_reg_second4(unsigned long long* tmp, unsigned long long* arr, unsigned long long* psiTable){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    int ggid = blockDim.x * blockIdx.x + tid;

    int step1 = n / reg_num / reg_num;
    int step2 = n / reg_num;
    gid %= step2;
    
    unsigned long long reg[reg_num];
    int gstep = r;
    unsigned long long q = q_const[ggid / step2];
    unsigned long long mu = mu_const[ggid / step2];
    int qbit = qbit_const[ggid / step2];  
    for(int i = 0; i < reg_num; i++){
        // int nloc = gid % 32 + i * 32 + gid / 32 * 256;
        int nloc = (gid % 32) % (n / 2048) + (gid % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + gid / 32 * 256;
        int t = n / 256;
        int nlocx = nloc / t;
        int nloczz = nloc % t;

        int nlocxx = nlocx % 4 + nlocx % 32 / 16 * 4;
        int nlocyy = (nlocx % 16)/ 4 + (nlocx / 32) * 4 + 32 * nloczz ;

        int loc = nlocxx * n / reg_num + nlocyy ;
        // if(nloc == 32){
        //     printf("<<<%d,%d,%llu\n",gid,i,tmp[loc]);
        // }
        reg[i] = tmp[gid / step2 * n / reg_num + loc + ggid / step2 * n ];
    }




    int gstepidx = nlog - 8;
    int step = 8;
    #pragma unroll
    for(int l = 0; l < 3; l++) {
        step >>= 1;
        gstepidx--;
        #pragma unroll
        for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
            int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
            
            int gloc = gid % (r / reg_num) + loc * r / reg_num + gid / (r / reg_num) * r; 
            // int gstep = step * 
            unsigned long long psi = psiTable[(256 << l) + (gloc >> (gstepidx + 1))+ggid / step2 * n];
            unsigned long long U = reg[loc];

            // if(gid == 0){
            //     printf("$$$%4d,%4d,%4d,%4d,%4d\n",loc,loc+step,gloc,gloc + (1 << gstepidx),(256 << l));
            // }
            uint128_t temp_storage = reg[loc + step];  
            mul64(temp_storage.low, psi, temp_storage);
            singleBarrett(temp_storage, q, mu, qbit);
            unsigned long long V = temp_storage.low;
            reg[loc] = (U + V) ;
            if(reg[loc] > q){
                reg[loc] -= q;
            }
            reg[loc + step] = (q + U - V) ;
            if(reg[loc + step] > q){
                reg[loc + step] -= q;
            }
        }
    }

    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 int nloc = (j % 32) % (n / 2048) + (j % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + j / 32 * 256;
    //                 reg[i] = nloc;

    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }


    // mymove1<unsigned long long, 1>(reg);

    // mymove1<unsigned long long, 2>(reg);

    // mymove1<unsigned long long, 4>(reg);
    // if(nlog >= 13)mymove1<unsigned long long, 2>(reg);
    // if(nlog >= 14)mymove1<unsigned long long, 4>(reg);

    // mymove<unsigned long long, 2>(reg);

    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }





    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 int nloc = (j % 32) % (n / 2048) + (j % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + j / 32 * 256;
    //                 // int nloc = (j % 32) % (n / 2048) + (j % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + j / 32 * 256;

    //                 reg[i] = nloc;

    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }

    if(nlog == 16){
    mymove1<unsigned long long, 1>(reg);
    mymove1<unsigned long long, 2>(reg);
    mymove1<unsigned long long, 4>(reg);

    }

    if(nlog == 15){

        mymove1<unsigned long long, 1>(reg);
        mymove1<unsigned long long, 2>(reg);
        // mymove1<unsigned long long, 4>(reg);
        // mymove<unsigned long long, 1>(reg);
        // mymove<unsigned long long, 2>(reg);
    // mymove<unsigned long long, 1>(reg);
    // if(nlog >= 13)mymove<unsigned long long, 2>(reg);
    // if(nlog >= 14)mymove<unsigned long long, 4>(reg);
    }
    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }
    if(nlog == 15){
        step = 4;
        #pragma unroll
        for(int l = 0; l < 2; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3-l-2)) * 2 * step + (bf_idx & (step - 1));
                // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 

                // int gloc = loc * 4 + gid % 32 % 4 + gid % 32 / 4 * 32 + gid / 32 * 256;
                // int gloc = loc % 2 * 16 + loc / 4 * 64 + gid % 32 % 8 + gid % 16 / 8 * 32 + gid % 32 / 16 * 128 + gid / 32 * 256;
                int gloc = loc % 4 * 4 + loc / 4 * 64 + gid % 32 % 4 + gid % 16 / 4 * 16 + gid % 32 / 16 * 128 + gid / 32 * 256;
                // int gstep = step * 
                unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1))+ggid / step2 * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
                // if(gloc == 1 && ggid == gid){
                //     printf("%d,%lld,%d,%d\n",l,reg[loc],loc,gid);
                // }
            }
        }
    }
    if(nlog == 16){
        step = 8;
        #pragma unroll
        for(int l = 0; l < 3; l++) {
            step >>= 1;
            gstepidx--;
            #pragma unroll
            for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
                int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
                // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 

                int gloc = loc * 4 + gid % 32 % 4 + gid % 32 / 4 * 32 + gid / 32 * 256;
                // int gstep = step * 
                unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1))+ggid / step2 * n];
                unsigned long long U = reg[loc];

                uint128_t temp_storage = reg[loc + step];  
                mul64(temp_storage.low, psi, temp_storage);
                singleBarrett(temp_storage, q, mu, qbit);
                unsigned long long V = temp_storage.low;
                reg[loc] = (U + V) ;
                if(reg[loc] > q){
                    reg[loc] -= q;
                }
                reg[loc + step] = (q + U - V) ;
                if(reg[loc + step] > q){
                    reg[loc + step] -= q;
                }
            }
        }
    }


    if(nlog == 16){
    mymove<unsigned long long, 1>(reg);
    mymove<unsigned long long, 2>(reg);

    }

    if(nlog == 15){

        mymove<unsigned long long, 1>(reg);
        mymove<unsigned long long, 2>(reg);

    }

    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 int nloc = (j % 32) % (n / 2048) + (j % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + j / 32 * 256;
    //                 // int nloc = (j % 32) % (n / 2048) + (j % 32) / (n / 2048) * 8 * (n / 2048)  + i * n / 2048 + j / 32 * 256;

    //                 reg[i] = nloc;

    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }
    if(nlog <= 14){

        mymove<unsigned long long, 1>(reg);
        if(nlog >= 13)mymove<unsigned long long, 2>(reg);
        if(nlog >= 14)mymove<unsigned long long, 4>(reg);
    }

    // if(blockIdx.x == 0){
    //     for(int i = 0; i < reg_num; i++){
    //         for(int j = 0; j < 32; j++){
    //             if(j == threadIdx.x){
    //                 printf("%5d",reg[i]);
    //             }
    //         }
    //         if(gid == 0){
    //             printf("\n");
    //         }
    //     }
    // }
    step = 8 >> (3 - min(gstepidx, 3));
    #pragma unroll
    for(int l = (3 - min(gstepidx, 3)); l < 3; l++) {
        step >>= 1;
        gstepidx--;
        #pragma unroll
        for(int bf_idx = 0; bf_idx < reg_num / 2; bf_idx++){
            int loc = (bf_idx >> (3 - l - 1)) * 2 * step + (bf_idx & (step - 1));
            
            int gloc = gid % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 
            if(nlog == 15){
                // gloc = gid % 16 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 

                gloc = loc % 4 + loc / 4 * 64 + gid % 32 % 16 * 4 + gid % 32 / 16 * 128 + gid / 32 * 256; 

            }
            if(nlog == 16){
                // gloc = gid % 32 / 8 * 8 + gid % 8 % (r / reg_num) * r / reg_num + gid / (r / reg_num) * r + loc  ; 
                gloc = loc % 4 + loc / 4 * 16 + gid % 32 % 4 * 4 +  gid % 32 / 4 * 32 + gid / 32 * 256;

            }
 
            unsigned long long psi = psiTable[(n >> (gstepidx + 1)) + (gloc >> (gstepidx + 1))+ggid / step2 * n];
            unsigned long long U = reg[loc];

            uint128_t temp_storage = reg[loc + step];  
            mul64(temp_storage.low, psi, temp_storage);
            singleBarrett(temp_storage, q, mu, qbit);
            unsigned long long V = temp_storage.low;
            reg[loc] = (U + V) ;
            if(reg[loc] > q){
                reg[loc] -= q;
            }
            reg[loc + step] = (q + U - V) ;
            if(reg[loc + step] > q){
                reg[loc + step] -= q;
            }
        }
    }




    
    for(int i = 0; i < reg_num; i++){

        int loc = i;
        
        int gloc = gid % (r / reg_num) * r / reg_num+ loc / (r / 8 ) * (r / 8) * (r / 8) +  loc % (r / 8 ) + gid / (r / reg_num) * r  ; 
        if(nlog == 15){

                gloc = loc % 4 + loc / 4 * 64 + gid % 32 % 16 * 4 + gid % 32 / 16 * 128 + gid / 32 * 256; 
        }
        if(nlog == 16){
                gloc = loc % 4 + loc / 4 * 16 + gid % 32 % 4 * 4 +  gid % 32 / 4 * 32 + gid / 32 * 256;
        }
        arr[gloc + gid / step2 * n / reg_num + ggid / step2 * n ] = reg[i];
    }
}












void forwardNTT9(unsigned long long* device_a, unsigned n, unsigned long long* psi_powers,unsigned batchSize){
    static unsigned long long* tmp;
    if(!tmp){
        cudaMalloc(&tmp, 65536 * sizeof(unsigned long long) * 64);

    }  
    if(n == 4096){
        const int l = 12;
        ntt_forward_reg_first3<1 << l,l><<<batchSize * n / 32 / reg_num, 32>>>(device_a, tmp, psi_powers);
        ntt_forward_reg_second4<1 << l,1 << (l - 8),l><<<batchSize * n / 32 / reg_num, 32>>>(tmp, device_a,psi_powers);
    }
    else if(n == 8192){
        const int l = 13;
        ntt_forward_reg_first3<1 << l,l><<<batchSize * n / 32 / reg_num, 32>>>(device_a, tmp, psi_powers);
        ntt_forward_reg_second4<1 << l,1 << (l - 8),l><<<batchSize * n / 32 / reg_num, 32>>>(tmp, device_a,psi_powers);

    }
    else if(n == 16384){
        const int l = 14;
        ntt_forward_reg_first3<1 << l,l><<<batchSize * n / 32 / reg_num, 32>>>(device_a, tmp, psi_powers);
        ntt_forward_reg_second4<1 << l,1 << (l - 8),l><<<batchSize * n / 32 / reg_num, 32>>>(tmp, device_a,psi_powers);
    } 
    else if(n == 16384 * 2){
        const int l = 15;
        ntt_forward_reg_first3<1 << l,l><<<batchSize * n / 32 / reg_num, 32>>>(device_a, tmp, psi_powers);
        ntt_forward_reg_second4<1 << l,1 << (l - 8),l><<<batchSize * n / 32 / reg_num, 32>>>(tmp, device_a,psi_powers);
    }
    else if(n == 16384 * 4){
        const int l = 16;
        ntt_forward_reg_first3<1 << l,l><<<batchSize * n / 32 / reg_num, 32>>>(device_a, tmp, psi_powers);
        ntt_forward_reg_second4<1 << l,1 << (l - 8),l><<<batchSize * n / 32 / reg_num, 32>>>(tmp, device_a,psi_powers);
    }
    else{
        printf("%d\n",n);
        throw "error";
    }
}
