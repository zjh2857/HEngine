#include <cstdio>
#include <stdlib.h>
#include "helper.cuh"
#include "mempool.cuh"
#include "freshman.h"

using namespace std;

struct uint128{
    unsigned long long low;
    unsigned long long high;
};
extern __constant__ unsigned long long q_const[128];
extern __constant__ unsigned long long qbit_const[128];
extern __constant__ unsigned long long mu_const[128];
extern __constant__ unsigned long long inv_q_last_mod_q_const[128];
extern __constant__ unsigned long long inv_punctured_q_const[128];
extern __constant__ unsigned long long prod_t_gamma_mod_q_const[128];
__global__ void print1(unsigned long long* a){
    for(int i = 0; i < 8; i++){
        printf("%llu\t",a[i]);
    }printf("\n");
}
__global__ void print(cuDoubleComplex* h_A);
#define Check(call)														\
{																		\
	cudaError_t status = call;											\
	if (status != cudaSuccess)											\
	{																	\
		cout << "行号" << __FILE__ << __LINE__ << endl;							\
		cout << "错误:" << cudaGetErrorString(status) << endl;			\
	}																	\
}
__constant__ double half_d;
__device__ __host__ __forceinline__ void bigMul(unsigned long long a,unsigned long long b, uint128& res){
    u_int64_t a0 = a & 0xffffffff;
    u_int64_t a1 = a >> 32;
    u_int64_t b0 = b & 0xffffffff;
    u_int64_t b1 = b >> 32;
    u_int64_t low = a0 * b0 ;
    u_int64_t carry = 0;
    if(low > low + ((a0 * b1) << 32llu)){
        carry+=1;
    }
    low = low + ((a0 * b1) << 32llu);
    if(low > low + ((a1 * b0) << 32llu)){
        carry+=1;
    }
    low = low + ((a1 * b0) << 32llu);

    u_int64_t high = a1 * b1 + ((a0 * b1)>> 32llu) + ((a1 * b0)>> 32llu) + carry;
    // high += ((a0 * b0) >> 32) + 
    res.low = low;
    res.high = high;
    // return res;
}
__device__ __host__ __forceinline__ void bigIntegerMul(unsigned long long * a, unsigned long long b,int size){
    uint128 temp; 
    bigMul(a[0],b,temp);
    unsigned long long carry = temp.high;
    a[0] = temp.low;

    for(int i = 1; i < size; i++){
        if(!a[i] && !carry){
            break;
        }
        uint128 temp;
        bigMul(a[i],b,temp);
        // temp = te
        a[i] = temp.low;
        // a[i] = (temp & 0xffffffffffffffff);
        if(a[i] + carry < a[i]){
            a[i] += carry;
            carry = 1 + temp.high;
        }
        else{
            a[i] += carry;
            carry = temp.high;
        }
    }
}

__device__ __host__ __forceinline__ void bigIntegerMul32bit(unsigned long long * a, unsigned long long b,int size){
    unsigned long long temp = a[0] * b;
    unsigned long long carry = (temp >> 32);
    a[0] = temp & ((1<<32)-1);

    for(int i = 1; i < size; i++){
        if(!a[i] && !carry){
            break;
        }
        unsigned long long temp = a[0] * b;

        a[i] = temp & ((1<<32)-1);
        // a[i] = (temp & 0xffffffffffffffff);
        if(a[i] + carry < a[i]){
            a[i] += carry;
            carry = 1 + (temp >> 32);
        }
        else{
            a[i] += carry;
            carry = (temp >> 32);
        }
    }
}
__device__ __host__ __forceinline__ void bigIntegerAdd(unsigned long long * a, unsigned long long *b,int size){
    unsigned long long carry = 0;
    for(int i = 0; i < size; i++){
        if(a[i] + b[i] + carry < a[i]){
            // carry = 1;
            a[i] = a[i] + b[i] + carry;
            carry = 1;
        }
        else{
            a[i] = a[i] + b[i] + carry;
            carry = 0;
        }
    }
}
__device__ __host__ __forceinline__ int isneg(unsigned long long * a, unsigned long long *b ,unsigned long long p,int size){
    // unsigned long long* temp = (unsigned long long*)malloc(size * sizeof(unsigned long long)); 
    unsigned long long temp[64];
    for(int i = 0; i < size; i++){
        temp[i] = b[i];
    }

    bigIntegerMul(temp,p,size);
    unsigned long long borrow = 0;
    for(int i = 0; i < size; i++){
        // if(p == 1742){
        //     printf("&&%d,%llu\n",i,borrow);
        // }
        if(a[i] < borrow){
            borrow = 1;
            temp[i] = a[i] - borrow - temp[i];
            continue;
        }

        if(a[i] - borrow >= temp[i]){
            temp[i] = a[i] - borrow - temp[i];
            borrow = 0;
        }else{
            temp[i] = a[i] - borrow - temp[i];
            borrow = 1;
        }
        
    }
    if(borrow == 1){
        return -1;
    }
    return 1;
}


__device__ __host__ __forceinline__ double bigInteger2udouble(unsigned long long *a,unsigned long long *b,unsigned long long p,int size,int tid){
    // unsigned long long* temp = (unsigned long long*)malloc(size * sizeof(unsigned long long)); 
    unsigned long long temp[64];
    for(int i = 0; i < size; i++){
        temp[i] = b[i];
    }
    bigIntegerMul(temp,p,size);

    unsigned long long borrow = 0;
    for(int i = 0; i < size; i++){

        if(a[i] < borrow){
            borrow = 1;
            temp[i] = a[i] - borrow - temp[i];
            continue;
        }

        if(a[i] - borrow >= temp[i]){
            temp[i] = a[i] - borrow - temp[i];
            borrow = 0;
        }else{
            temp[i] = a[i] - borrow - temp[i];
            borrow = 1;
        }
        
    }

    // if(tid==0){
    //     for(int i = 0; i < size; i++){
    //         printf("%llu * (2 ** %d) + \t",temp[i],i * 64);
    //     }printf("\n");
    //     // printf("%lf,%lf\n",res_n,res_p);
    // }

    double res_p = 0;
    double res = 0;
    double res_n = 0;
    double twopow64 = 18446744073709551616.0;
    double base = 1.0;
    // if(tid==0){
    //     for(int i = 0; i < size; i++){
    //         printf("%llu!\t",temp[i]);
    //     }printf("\n");
    //     // printf("%lf,%lf\n",res_n,res_p);
    // }
    for(int i = 0; i < size && i < 8; i++){
        res_p += temp[i] * base;
        base*= twopow64;
    }

    for(int i = 0; i < size; i++){
        temp[i] = b[i];
    }

    bigIntegerMul(temp,p+1,size);
    borrow = 0;
    for(int i = 0; i < size; i++){

        if(a[i] < borrow){
            borrow = 1;
            temp[i] = a[i] - borrow - temp[i];
            continue;
        }

        if(a[i] - borrow >= temp[i]){
            temp[i] = a[i] - borrow - temp[i];
            borrow = 0;
        }else{
            temp[i] = a[i] - borrow - temp[i];
            borrow = 1;
        }
        
    }
    for(int i = 0; i < size; i++){
        temp[i] = ~temp[i];
    }
    unsigned long long carry = 1;
    for(int i = 0; i < size; i++){
        temp[i] += carry;
        if(temp[i] == 0 && carry){
            carry = 1;
        }else{
            carry = 0;
        }
    }
    base = 1.0;
    // if(tid==0){
    //     for(int i = 0; i < size; i++){
    //         printf("%llu * (2 ** %d) + \t",temp[i],i * 64);
    //     }printf("\n");
    //     // printf("%lf,%lf\n",res_n,res_p);
    // }
    for(int i = 0; i < size && i < 8; i++){
        res_n += temp[i] * base;
        base*= twopow64;
    }

    if(res_n < res_p){
        res = -res_n;
    }
    else{
        res = res_p;
    }

    return res;
}
__device__ __host__ __forceinline__ double bigIntegerMod(unsigned long long * a, unsigned long long *b,int size,int tid){
    unsigned long long l = 0;
    unsigned long long r = (1llu << 63);


    int cnt = 0;
    unsigned long long p = 0;
    // if(tid==0){
    //     for(int i = 0; i < 8;i++){
    //         printf("%llu\t",a[i]);
    //     }printf("\n");
    //     for(int i = 0; i < 8;i++){
    //         printf("%llu\t",b[i]);
    //     }printf("\n");
    // }
    while(l < r){
        if(cnt++ > 100){
            printf("114514\n");
            return 114514;
        }
        // printf("$$%d,%llu,%llu,%llu\n",cnt,l,r,(l+r)/2);
        unsigned long long guess = (l + r)/2;
        // if(tid == 114514){
        //     printf("%llu,%llu,%llu\n",l,r,(l+r)/2);
        // }
        int res = isneg(a,b,guess,size);
        if(res == 1){
            p = guess;
            l = guess + 1;

        }
        else if(res == -1){
            r = guess;
        }
    }
    // if(tid == 0){
    //     printf("%llu!!\n",p);
    // }
    return bigInteger2udouble(a,b,p,size,tid);

}


__global__ void cudadecompose(cuDoubleComplex *list,unsigned long long* moduleChain,int listLen,int moduleLen,unsigned long long * decomposeList,unsigned long long scale){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint128_t temp;

    // if(tid == 3744) {
    //     printf("%lf\n",list[tid].x / listLen * scale);
    // }
    if(list[tid].x < 0){
        temp = -list[tid].x / listLen * scale;
        // if(temp.low != 0){
        //     printf("JSDF%lld,%d\n",temp.low,tid);

        // }
        for(int i = 0; i < moduleLen; i++){
            decomposeList[i * listLen + tid] =moduleChain[i] - (temp % moduleChain[i]).low;
            decomposeList[i * listLen + tid] %= moduleChain[i];
            // if(decomposeList[i * listLen + tid] > moduleChain[i]){
            //     decomposeList[i * listLen + tid] -= moduleChain[i];
            // }
        }        
    }
    else{
        temp = list[tid].x / listLen * scale;
        // if(temp.low != 0){
        //     printf("JSDF%lld,%d\n",temp.low,tid);
        // }
        for(int i = 0; i < moduleLen; i++){
            decomposeList[i * listLen + tid] = (temp % moduleChain[i]).low;
            decomposeList[i * listLen + tid] %= moduleChain[i];

        }
    }
    if(decomposeList[0 * listLen + tid] != 0){
        // printf("DFH%lld,%d\n",decomposeList[0 * listLen + tid],tid);
    }
}

__global__ void cudadecompose(unsigned long long *list,unsigned long long* moduleChain,int listLen,int moduleLen,unsigned long long * decomposeList){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long temp = list[tid];
    for(int i = 0; i < moduleLen; i++){
        decomposeList[i * listLen + tid] = temp % moduleChain[i];
        // if(tid == 0){
        //     printf("%llu,%llu,%llu\n",list[tid],moduleChain[i],decomposeList[i * listLen + tid]);
        // }
    }
}
__global__ void cudacompose_MRS_reg(unsigned long long * decomposeList,cuDoubleComplex * composeList,unsigned long long * qi_qj_inv,int moduleLen,int listLen,int rescaleTimes){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // moduleLen = 32;
    register unsigned long long reg[32];
    register unsigned long long mod[32];
    for(int j = 0; j < moduleLen; j++){
        reg[j] = decomposeList[j * listLen + tid];
        mod[j] = qi_qj_inv[j*moduleLen+j];
    }    
    for(int i = rescaleTimes; i < moduleLen; i++){
        for(int j = i+1; j < moduleLen; j++){
            reg[j] = (mod[j] + reg[j] - reg[i]);
            reg[j] %= mod[j];
            reg[j] *= qi_qj_inv[i*moduleLen+j];
            reg[j] %= mod[j];
        }

    }
    for(int j = 0; j < moduleLen; j++){
        decomposeList[j * listLen + tid] = reg[j];
    }  
    // double base = 1.0;

    // double res1 = 0.0;
    // double res2 = 1.0;
    // for(int i = rescaleTimes; i < moduleLen ; i++){
    //     res1 += decomposeList[i * listLen + tid]*base;
    //     base *= qi_qj_inv[i*moduleLen+i];

    // }
    // base = 1.0;
    // for(int i = rescaleTimes; i < moduleLen; i++){
    //     res2 += (qi_qj_inv[i*moduleLen+i]-decomposeList[i * listLen + tid]-1)*base;
    //     base *= qi_qj_inv[i*moduleLen+i];
    // }        
    // double res;
    // if(res1 < res2){
    //     res = res1;
    // }else{
    //     res = -res2;
    // }
    // composeList[tid].x = res;
    // composeList[tid].y = 0;
}
__global__ void cudacompose_MRS(unsigned long long * decomposeList,cuDoubleComplex * composeList,unsigned long long * qi_qj_inv,int moduleLen,int listLen,int rescaleTimes){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // if(tid == 1481){
    //     for(int i = 0; i < 6; i++)printf("GH%d,%lld,%d\n",moduleLen,decomposeList[i * listLen + tid],tid);
    // }


    // moduleLen = 32;
    for(int i = rescaleTimes; i < moduleLen; i++){
        for(int j = i+1; j < moduleLen; j++){
            
            // decomposeList[j * listLen + tid] = (qi_qj_inv[j*moduleLen+j] + decomposeList[j * listLen + tid] - decomposeList[i * listLen + tid]);
            // decomposeList[j * listLen + tid] %= qi_qj_inv[j*moduleLen+j];
            // decomposeList[j * listLen + tid] *= qi_qj_inv[i*moduleLen+j];
            // decomposeList[j * listLen + tid] %= qi_qj_inv[j*moduleLen+j];

            unsigned long long q = q_const[j];
            unsigned long long mu = mu_const[j];
            int qbit = qbit_const[j];

            
            register uint128_t temp_storage = (qi_qj_inv[j*moduleLen+j] + decomposeList[j * listLen + tid] - decomposeList[i * listLen + tid]);  // this is for eliminating the possibility of overflow


            mul64(temp_storage.low, qi_qj_inv[i*moduleLen+j], temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);

            decomposeList[j * listLen + tid] = temp_storage.low;


        }
        // if(tid == 0){
        //     for(int k = 0; k < 6; k++)printf("$$%d,%d,%lld\n",i,moduleLen,decomposeList[k* listLen + 0]);
        // }
    }
    double base = 1.0;

    double res1 = 0.0;
    double res2 = 1.0;
    for(int i = rescaleTimes; i < moduleLen ; i++){
        res1 += decomposeList[i * listLen + tid]*base;
        base *= qi_qj_inv[i*moduleLen+i];

    }
    base = 1.0;
    for(int i = rescaleTimes; i < moduleLen; i++){
        res2 += (qi_qj_inv[i*moduleLen+i]-decomposeList[i * listLen + tid]-1)*base;
        base *= qi_qj_inv[i*moduleLen+i];
    }   


    // if(tid == 1481){
    //     printf("MBBBBB%lf,%lf\n",res1,res2);
    // }     
    double res;
    if(res1 < res2){
        res = res1;
    }else{
        res = -res2;
    }
    composeList[tid].x = res;
    // if(tid == 1481){
    //     printf("MBBBBB%lf\n",res);
    // }    
    composeList[tid].y = 0;
}


__global__ void cudacompose_MRS_batch(unsigned long long * decomposeList,cuDoubleComplex * composeList,unsigned long long * qi_qj_inv,int moduleLen,int listLen,int rescaleTimes){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
// moduleLen =32;
    int idx = tid / moduleLen + (tid % moduleLen) * listLen;
    int mid = tid % moduleLen;
    int cid = tid / moduleLen;
    
    for(int i = 0; i < moduleLen; i++){
        if (i > mid) {
            
            decomposeList[idx] = (qi_qj_inv[mid*moduleLen+mid] + decomposeList[idx] - decomposeList[cid + i * listLen]);
            decomposeList[idx] %= qi_qj_inv[mid*moduleLen+mid];
            decomposeList[idx] *= qi_qj_inv[i*moduleLen+mid];
            decomposeList[idx] %= qi_qj_inv[mid*moduleLen+mid];
        }
    }
    double base = 1.0;

    double res1 = 0.0;
    double res2 = 1.0;
    if (mid == 0){
        for(int i = 0; i < moduleLen ; i++){
            res1 += decomposeList[i * listLen + cid]*base;
            base *= qi_qj_inv[i*moduleLen+i];

        }
        base = 1.0;
        for(int i = 0; i < moduleLen; i++){
            res2 += (qi_qj_inv[i*moduleLen+i]-decomposeList[i * listLen + cid]-1)*base;
            base *= qi_qj_inv[i*moduleLen+i];
        }        
        double res;
        if(res1 < res2){
            res = res1;
        }else{
            res = -res2;
        }
        composeList[cid].x = res;
        composeList[cid].y = 0;
    }

}

__global__ void cudacompose_MRS_shuffle(unsigned long long * decomposeList,unsigned long long * composeList,unsigned long long * qi_qj_inv,int moduleLen,int listLen,int rescaleTimes){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
// moduleLen =32;
    int idx = tid / moduleLen + (tid % moduleLen) * listLen;
    int mid = tid % moduleLen;
    int cid = tid / moduleLen;

    register unsigned long long reg = decomposeList[idx];
    register unsigned long long get;
    register unsigned long long mod = qi_qj_inv[mid*moduleLen+mid];
    unsigned long long q = q_const[mid];
    unsigned long long mu = mu_const[mid];
    int qbit = qbit_const[mid];
    for(int i = rescaleTimes; i < moduleLen; i++){
        get = __shfl(reg,i,moduleLen);

        if(i < mid) {
            reg = mod + reg - get;
            if(reg > mod){
                reg -= mod;
            }
            if(reg > mod){
                reg -= mod;
            }
            // reg %= mod;
            register uint128_t temp_storage = reg;  // this is for eliminating the possibility of overflow


            mul64(temp_storage.low, qi_qj_inv[i*moduleLen+mid], temp_storage);

            singleBarrett(temp_storage, q, mu, qbit);
            // reg *= qi_qj_inv[i*moduleLen+mid];
            reg = temp_storage.low;
        }
    }
    // if(tid == 0){
    //     printf("ADDDDD%llu\n",reg);
    // }
    composeList[idx] = reg;
}
__global__ void MRS2float(unsigned long long * decomposeList,cuDoubleComplex* composeList,unsigned long long *qi_qj_inv,int moduleLen,int listLen,int rescaleTimes){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    double base = 1.0;
    double res1 = 0.0;
    double res2 = 1.0;
    for(int i = rescaleTimes; i < moduleLen && i < 16; i++){
        res1 += decomposeList[i * listLen + tid]*base;
        base *= qi_qj_inv[i*moduleLen+i];

    }
    base = 1.0;
    for(int i = rescaleTimes; i < moduleLen && i < 16; i++){
        res2 += (qi_qj_inv[i*moduleLen+i]-decomposeList[i * listLen + tid]-1)*base;
        base *= qi_qj_inv[i*moduleLen+i];
    }        
    double res;
    if(res1 < res2){
        res = res1;
    }else{
        res = -res2;
    }
    // if(tid == 47){
    //     for(int i = 0; i < moduleLen; i++){
    //         printf("%llu\t",decomposeList[i * listLen + tid - 1]);
    //     }printf("\n\n\n");   
    //     for(int i = 0; i < moduleLen; i++){
    //         printf("%llu\t",decomposeList[i * listLen + tid]);
    //     }printf("\n\n\n");            
    // }
    composeList[tid].x = res;
    composeList[tid].y = 0;
}

__global__ void cudacompose(unsigned long long *decomposeList,
                            unsigned long long* moduleChain,
                            int listLen,
                            int moduleLen,
                            unsigned long long* Ni,
                            unsigned long long *bigN,cuDoubleComplex * composeList,unsigned long long* temp1,unsigned long long* temp2,unsigned long long scale){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = 0; i < moduleLen; i++){
        temp2[tid * moduleLen + i] = 0;
    }
    for(int i = 0; i < moduleLen; i++){
        for(int j = 0; j < moduleLen; j++){
            temp1[tid * moduleLen + j] = Ni[i * moduleLen + j];
        }
        
        bigIntegerMul(&(temp1[tid * moduleLen]),decomposeList[i * listLen + tid]%moduleChain[i],moduleLen);
        bigIntegerAdd(&temp2[tid * moduleLen],&temp1[tid * moduleLen],moduleLen);
    }
    cuDoubleComplex res;
    res.x = bigIntegerMod(&temp2[tid * moduleLen],bigN,moduleLen,tid);
    res.y = 0;

    composeList[tid] = res;
}


__global__ void cudacompose_batch(unsigned long long *decomposeList,
                            unsigned long long* moduleChain,
                            int listLen,
                            int moduleLen,
                            unsigned long long* Ni,
                            unsigned long long *bigN,cuDoubleComplex * composeList,unsigned long long* temp1,unsigned long long* temp2,unsigned long long scale){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // for(int i = 0; i < moduleLen; i++){
    //     temp2[tid * moduleLen + i] = 0;
    // }
    // for(int i = 0; i < moduleLen; i++){
        // for(int j = 0; j < moduleLen; j++){
        //     temp1[tid * moduleLen + j] = Ni[i * moduleLen + j];
        // }
        
        bigIntegerMul32bit(&(temp1[tid * moduleLen]),decomposeList[tid]%moduleChain[tid%moduleLen],moduleLen);
        // bigIntegerAdd(&temp2[tid * moduleLen],&temp1[tid * moduleLen],moduleLen);
    // }
    // cuDoubleComplex res;
    // res.x = bigIntegerMod(&temp2[tid * moduleLen],bigN,moduleLen,tid);
    // res.y = 0;

    // composeList[tid] = res;
}
__global__ void cudacompose(unsigned long long *decomposeList,
                            unsigned long long* moduleChain,
                            int listLen,
                            int moduleLen,
                            unsigned long long* Ni,
                            unsigned long long *bigN,cuDoubleComplex * composeList,unsigned long long* temp1,unsigned long long* temp2,unsigned long long scale,int rescaleTimes){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = 0; i < moduleLen; i++){
        temp2[tid * moduleLen + i] = 0;
    }
    for(int i = rescaleTimes; i < moduleLen; i++){
        for(int j = 0; j < moduleLen; j++){
            temp1[tid * moduleLen + j] = Ni[i * moduleLen + j];
        }
        // if(tid == 0){
        //     for(int j = 0; j < moduleLen; j++){
        //         printf("##%llu\n",temp1[tid * moduleLen + j]);
        //     }
        // }
        bigIntegerMul(&(temp1[tid * moduleLen]),decomposeList[i * listLen + tid]%moduleChain[i],moduleLen);
        bigIntegerAdd(&temp2[tid * moduleLen],&temp1[tid * moduleLen],moduleLen);
    }
    cuDoubleComplex res;
    // if(tid == 0){
    //     for(int i = 0; i < 8; i++){
    //         printf("!!%llu\n",decomposeList[i * listLen + tid]);
    //     }
    // }
    res.x = bigIntegerMod(&temp2[tid * moduleLen],bigN,moduleLen,tid);
    res.y = 0;

    composeList[tid] = res;
}


class RNS{
    public:
    int N;
    unsigned long long* moduleChain;
    unsigned long long* moduleChain_B;
    unsigned long long* Ni;
    unsigned long long *bigN;
    unsigned long long *buff1;
    unsigned long long *buff2;
    unsigned long long* moduleChain_h;
    unsigned long long scale;
    unsigned long long step = 2;
    unsigned long long* qi_qj_inv;
    RNS(int N,unsigned long long scale){
        this->N = N;
        this->scale = scale;
        buff1 = nullptr;
        buff2 = nullptr;

        moduleChain_h = (unsigned long long*)malloc(N * sizeof(unsigned long long));
        // qi_qj_inv = (unsigned long long*)malloc(N * N * sizeof(unsigned long long));
        unsigned long long* temp = (unsigned long long*)malloc(N * sizeof(unsigned long long));
        unsigned long long* B_basis = (unsigned long long*)malloc(N * sizeof(unsigned long long));
        unsigned long long* qi_qj_inv_t = (unsigned long long*)malloc(N * N * sizeof(unsigned long long));
        getrnsPrime(temp,B_basis,qi_qj_inv_t,N);
        double half_h = 1.0;
        for(int i = 0; i < N; i++){
            moduleChain_h[i] = temp[i];
            half_h *= temp[i];
        }
        half_h /= 2.0;

        // cudaMemcpyToSymbol(&half_d, &half_h, sizeof(double));
        // genPrime(moduleChain_h,scale,N);
        unsigned long long** Ni_h = (unsigned long long**)calloc(N,sizeof(unsigned long long**));
        for(int i = 0; i < N;i++){
            Ni_h[i] = (unsigned long long*)calloc(N,sizeof(unsigned long long));
        }
        // unsigned long long bigN[N];
        unsigned long long *bigN_h = (unsigned long long*)calloc(N,sizeof(unsigned long long));
        unsigned long long *ti_h = (unsigned long long*)calloc(N,sizeof(unsigned long long));
        bigN_h[0] = 1;
        for(int i = 0;i < N; i++){
            Ni_h[i][0] = 1;
            ti_h[i] = 1;
        }

        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                if(i==j)continue;
                bigIntegerMul(Ni_h[i],moduleChain_h[j],N);
                ti_h[i] = ti_h[i] * modpow128(moduleChain_h[j],moduleChain_h[i]-2,moduleChain_h[i]) % moduleChain_h[i];
            }
        }
        
        for(int i = 0; i < N; i++){
            bigIntegerMul(bigN_h,moduleChain_h[i],N);
        }
        for(int i = 0; i < N; i++){
            bigIntegerMul(Ni_h[i],ti_h[i],N);
        }


        Check(mempool(&qi_qj_inv,N * N * sizeof(unsigned long long)));

        Check(mempool(&moduleChain,N * sizeof(unsigned long long)));
        Check(mempool(&moduleChain_B,N * sizeof(unsigned long long)));
        Check(mempool(&Ni,N * N * sizeof(unsigned long long)));
        Check(mempool(&bigN,N * sizeof(unsigned long long)));

        cudaMemcpy(qi_qj_inv,qi_qj_inv_t,N * N * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(moduleChain,moduleChain_h,N * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(moduleChain_B,B_basis,N * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(bigN,bigN_h,N * sizeof(unsigned long long),cudaMemcpyHostToDevice);

        for(int i = 0; i < N; i++){
            cudaMemcpy(Ni+(N*i),Ni_h[i],N * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        }
    }
    unsigned long long* decompose(cuDoubleComplex *list,int listLen){
        unsigned long long * decomposeList;
        // printf("@@@@@@@@@@@@@@@@@@@@@%d\n",N);
        Check(mempool(&decomposeList, listLen * N * sizeof(unsigned long long)));
        cudadecompose<<<listLen/1024,1024>>>(list,moduleChain,listLen,N,decomposeList,scale);
        return decomposeList;
    }

    unsigned long long* decompose(unsigned long long *list,int listLen){
        unsigned long long * decomposeList;
        Check(mempool(&decomposeList, listLen * N * sizeof(unsigned long long)));
        cudadecompose<<<listLen/1024,1024>>>(list,moduleChain,listLen,N,decomposeList);
        return decomposeList;
    }
    unsigned long long* decomposeLongBasis(unsigned long long *list,int listLen){
        unsigned long long * decomposeList;
        Check(mempool(&decomposeList, 2 * listLen * N * sizeof(unsigned long long)));
        cudadecompose<<<listLen/1024,1024>>>(list,moduleChain,listLen,N,decomposeList);
        cudadecompose<<<listLen/1024,1024>>>(list,moduleChain_B,listLen,N,decomposeList + (listLen * N));
        return decomposeList;
    }
    cuDoubleComplex* compose(unsigned long long * decomposeList, int listLen){
        cuDoubleComplex * composeList;
        Check(mempool(&composeList, listLen * N * sizeof(cuDoubleComplex)));

        if(!buff1)Check(mempool(&buff1, listLen * N * sizeof(unsigned long long)));
        if(!buff2)Check(mempool(&buff2, listLen * N * sizeof(unsigned long long)));

        cudacompose<<<listLen/1024,1024>>>(decomposeList,moduleChain,listLen,N,Ni,bigN,composeList,buff1,buff2,scale);
        return composeList;        
    }

    cuDoubleComplex* compose_batch(unsigned long long * decomposeList, int listLen){
        cuDoubleComplex * composeList;
        Check(mempool(&composeList, listLen * N * sizeof(cuDoubleComplex)));

        if(!buff1)Check(mempool(&buff1, listLen * N * sizeof(unsigned long long)));
        if(!buff2)Check(mempool(&buff2, listLen * N * sizeof(unsigned long long)));

        cudacompose_batch<<<N*listLen/1024,1024>>>(decomposeList,moduleChain,listLen,N,Ni,bigN,composeList,buff1,buff2,scale);
        return composeList;        
    }
    cuDoubleComplex* compose(unsigned long long * decomposeList, int listLen,int rescaleTimes){
        cuDoubleComplex * composeList;
        Check(mempool(&composeList, listLen * N * sizeof(cuDoubleComplex)));

        if(!buff1)Check(mempool(&buff1, listLen * N * sizeof(unsigned long long)));
        if(!buff2)Check(mempool(&buff2, listLen * N * sizeof(unsigned long long)));


        unsigned long long** Ni_h = (unsigned long long**)calloc(N,sizeof(unsigned long long**));
        for(int i = 0; i < N;i++){
            Ni_h[i] = (unsigned long long*)calloc(N,sizeof(unsigned long long));
        }
        // unsigned long long bigN[N];
        unsigned long long *bigN_h = (unsigned long long*)calloc(N,sizeof(unsigned long long));
        unsigned long long *ti_h = (unsigned long long*)calloc(N,sizeof(unsigned long long));
        bigN_h[0] = 1;
        for(int i = 0;i < N; i++){
            Ni_h[i][0] = 1;
            ti_h[i] = 1;
        }
        for(int i = rescaleTimes; i < N; i++){
            for(int j = rescaleTimes; j < N; j++){
                if(i==j)continue;
                bigIntegerMul(Ni_h[i],moduleChain_h[j],N);
                ti_h[i] = ti_h[i] * modpow128(moduleChain_h[j],moduleChain_h[i]-2,moduleChain_h[i]) % moduleChain_h[i];
            }
        }

        for(int i = rescaleTimes; i < N; i++){
            bigIntegerMul(bigN_h,moduleChain_h[i],N);
        }
        for(int i = rescaleTimes; i < N; i++){
            bigIntegerMul(Ni_h[i],ti_h[i],N);
        }
        free(ti_h);
        unsigned long long *moduleChain_r,*Ni_r,*bigN_r;
        Check(mempool(&moduleChain_r,N * sizeof(unsigned long long)));
        Check(mempool(&Ni_r,N * N * sizeof(unsigned long long)));
        Check(mempool(&bigN_r,N * sizeof(unsigned long long)));

        cudaMemcpy(moduleChain_r,moduleChain_h,N * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(bigN_r,bigN_h,N * sizeof(unsigned long long),cudaMemcpyHostToDevice);

        for(int i = 1; i < N; i++){
            cudaMemcpy(Ni_r+(N*i),Ni_h[i],N * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        }
        cudacompose<<<listLen/1024,1024>>>(decomposeList,moduleChain,listLen,N,Ni_r,bigN_r,composeList,buff1,buff2,scale,rescaleTimes);
        return composeList;        
    }


    cuDoubleComplex* compose_MRS(unsigned long long * decomposeList, int listLen,int rescaleTimes){
        cuDoubleComplex * composeList;
        Check(mempool(&composeList, listLen * N * sizeof(cuDoubleComplex)));
        cudacompose_MRS<<<listLen/1024,1024>>>(decomposeList,composeList,qi_qj_inv,N,listLen,rescaleTimes);
        return composeList;        
    }
    cuDoubleComplex* compose_MRS_shuffle(unsigned long long * decomposeList, int listLen,int rescaleTimes){
        cuDoubleComplex * composeList;
        Check(mempool(&composeList, listLen * N * sizeof(cuDoubleComplex)));
        unsigned long long *mrsList;
        Check(mempool(&mrsList, listLen * N * sizeof(unsigned long long)));

        cudacompose_MRS_shuffle<<<N * listLen/512,512>>>(decomposeList,mrsList,qi_qj_inv,N,listLen,rescaleTimes);
        MRS2float<<<listLen/1024,1024>>>(mrsList,composeList,qi_qj_inv,N,listLen,rescaleTimes);
        return composeList;        
    }
    cuDoubleComplex* compose_MRS_batch(unsigned long long * decomposeList, int listLen,int rescaleTimes){
        cuDoubleComplex * composeList;
        Check(mempool(&composeList, listLen * N * sizeof(cuDoubleComplex)));
        cudacompose_MRS_batch<<< N * listLen/1024,1024>>>(decomposeList,composeList,qi_qj_inv,N,listLen,rescaleTimes);
        return composeList;        
    }
    // unsigned long long* compose_u(unsigned long long * decomposeList, int listLen){
    //     unsigned long long* composeList;
    //     mempool(&composeList, listLen * N * sizeof(unsigned long long));

    //     if(!buff1)mempool(&buff1, listLen * N * sizeof(unsigned long long));
    //     if(!buff2)mempool(&buff2, listLen * N * sizeof(unsigned long long));

    //     cudacompose<<<listLen/1024,1024>>>(decomposeList,moduleChain,listLen,N,Ni,bigN,composeList,buff1,buff2);
    //     return composeList;        
    // }
    // void getParams(unsigned long long *q, unsigned long long* psi,unsigned long long* psiinv, unsigned long long* q_bit,int polylen){
    //     //hard
    //     unsigned long long psi_t[8] = {1034474, 1172569, 1557013, 349058, 785782, 1977521, 627147, 4561077};
    //     unsigned long long psiinv_t[8] = {441827, 1160428, 233173, 928390, 2113364, 499635, 1712567, 3973130};
    //     unsigned long long q_bit_t[8] = {21, 21, 21, 22, 22, 22, 22, 23};
    //     if(polylen == 4096){

    //         for(int i = 0; i < 8; i++){
    //             q[i] = moduleChain_h[i];
    //             psi[i] = psi_t[i];
    //             psiinv[i] = psiinv_t[i];
    //             q_bit[i] = q_bit_t[i];
    //         }
    //     }else{
    //         throw "wrong polylen";
    //     }
    // }
    private:
    void genPrime(unsigned long long* moduleChain_h,unsigned long long scale,int N){
        if(scale % 2 == 1){
            scale+=1;
        }
        int cnt = 0;
        while(cnt < N){
            if(MillerRabin(scale+1)){
                moduleChain_h[cnt] = scale + 1;
                scale += step;
                cnt++;
            }
            else{
                scale += step;
            }
        }
    }

    bool MillerRabin(unsigned long long n){

        if(n == 2){
            return true;
        }
        if(n % 2 == 0){
            return false;
        }
        for(unsigned long long a = 2; a < 64 && a < n; a++){
            unsigned long long d = n - 1;

            while(d % 2 == 0){
                if(modpow128(a,d,n) == 1){
                    // continue;
                }
                else if(modpow128(a,d,n) == n-1){
                    break;
                }
                else{
                    return false;
                }
                d /=2;
            }
        }
        return true;
    }
};

// __global__ void init(unsigned long long* a){
//     for(int i = 0; i < 4096; i++){
//         a[i] = i ;
//     }
// }

// int main(){
//     RNS rns(8,10000);
//     unsigned long long *ptr;
//     mempool(&ptr,4096 * sizeof(unsigned long long));
//     init<<<1,1>>>(ptr);
//     unsigned long long *res = rns.decompose(ptr,4096);
//     printf("decompose finish\n");
//     unsigned long long *ori = rns.compose(res,4096);
//     print<<<1,1>>>(ori);
//     cudaDeviceSynchronize();
// }
void bigMulTest(){
    srand((unsigned)time(NULL)); 
    for(int i = 0; i < 1000000; i++){
        unsigned long long r = rand();
        r = (r * 11451419 + 3777);
        unsigned long long s = rand();
        s = (s * 11451419 + 3777);
        uint128 res;
        bigMul(r,s,res);
        unsigned __int128 rr = r;
        unsigned __int128 ss = s;
        if(res.high != (unsigned long long)((rr*ss)>>64)){
            printf("%llu,%llu,%llu,%llu\n",r,s,res.low , (unsigned long long)((rr*ss)>>64));
        }
    }
}
