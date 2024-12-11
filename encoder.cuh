#pragma once
#include "parameter.cuh"
#include "uint128.cuh"
#include "ntt_60bit.cuh"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include <iostream>
#include <cmath>
#include "rns.cuh"
#include "freshman.h"
#include "mempool.cuh"
// #include "random.cuh"
#include "polycalc.cuh"
__global__ void genRandom(unsigned long long *randomVec,unsigned long long scale);
__global__ void print(unsigned long long* a,int size);
// typedef unsigned long long long long;
#define Check(call)														\
{																		\
	cudaError_t status = call;											\
	if (status != cudaSuccess)											\
	{																	\
		cout << "行号" << __FILE__ << __LINE__ << endl;							\
		cout << "错误:" << cudaGetErrorString(status) << endl;			\
	}																	\
}
__global__ void print_ddddd(unsigned long long* a){
    printf("*********\n");

    for(int i = 0; i < 8; i++){
        if(a[i]!=0)printf("%d,%llu\t",i,a[i ]);
    }
    printf("=======\n");
}
__global__ void print_out(cuDoubleComplex* a,int size){
    printf("*********\n");

    for(int i = 0; i < 8192*2; i++){
        if(a[i].x>=0.000001)printf("%d,%lf\t",i,a[i].x);
        // a[16383].x = 0;
    }
    printf("\n=======\n");
}

__global__ void print_out_t(cuDoubleComplex* a,int size){
    printf("*********\n");

    for(int i = 0; i < 8192*2; i++){
        if(a[i].x>=0.000001)printf("%d,%lf\t",i,a[i].x);
        if(i!=0)a[i].x = 0;

    }
    printf("\n=======\n");
}

__global__ void print_in(unsigned long long* a,int size){
    printf("*********\n");

    for(int i = 0; i < 8192*2; i++){
        if(a[i]!=0)printf("%d,%lu\t",i,a[i]);
    }
    printf("\n=======\n");
}

__global__ void print_e(unsigned long long* a,int size){
    printf("*********\n");

    for(int i = 0; i < size; i++){
        printf("%llu\t",a[i + size]);
    }
    printf("\n=======\n");
}
__global__ void print_f(unsigned long long* a,int size){
    printf("*********\n");

    for(int i = 0; i < 8192 * 2; i++){
        printf("%llu\t",a[i+8192*2*2]);
    }
    printf("\n=======\n");
}
__global__ void print_g(cuDoubleComplex* a){
    printf("*********\n");

    for(int i = 0; i < 8; i++){
        if(a[i].x != 0){
            printf("%d,%lf;",i,a[i].x);
        }
        // printf("%4lf,",a[i].x);
    }
    printf("\n=======\n");
}
__global__ void print_h(cuDoubleComplex* a){
    printf("*********\n");

    for(int i = 0; i < 1024; i++){
        printf("%d,%lf\t",i,a[i].x);
    }
    printf("\n=======\n");
}
using namespace std;
const double pi = 3.14159265358979323846264338327950288;;
extern __constant__  unsigned long long q_const[128];
extern __constant__  unsigned long long mu_const[128];
extern __constant__  unsigned long long qbit_const[128];
// __global__ void delta(cuDoubleComplex* in,unsigned long long* out,double scale,unsigned long long q,int N){

//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     long long t = in[tid].x / N  * scale;
//     t = (t+q)%q;
//     out[tid] = t; 
// }
// __global__ void zerocnt(unsigned long long* a,int N){
//     int cnt = 0;
//     for(int i = 0; i < N; i++){
//         if(a[i] != 0){
//             printf("%d,%llu\t",i,a[i]);
//             cnt++;
//         }
//     }printf("\n");
//     printf("%d\n",cnt);
// }
// __global__ void eqcnt(unsigned long long* a,int N){
//     int cnt = 0;
//     for(int i = 1; i < N; i++){
//         if(a[i] != a[i-1]){
//             cnt++;
//             printf("%d,%llu\n",i,a[i]);
//         }
//     }printf("\n");
//     printf("%d\n",cnt);
// }
// __global__ void deltainv(unsigned long long* in,cuDoubleComplex* out,double scale,unsigned long long q){
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     long long t = in[tid];
//     if(t > q/2){
//         t -= q;
//     }
//     out[tid].x = (double)(t * 1.0 / scale);
//     out[tid].y = 0;
// }
__global__ void print(unsigned long long* a);
__global__ void print_d(unsigned long long* a,int d);

//     printf("%llu\n",a[0]);
// }
__global__ void print(cuDoubleComplex* h_A){
    for(int i = 0;i < 8; i++){
        printf("%lf+%lfi\n",h_A[i].x,h_A[i].y);
    }
    printf("\n");
}
__global__ void initfft(cuDoubleComplex* h_A,int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid/N;
    int j = tid%N;
    int k = modpow64(3,j,2*N);
    if(j >= N / 2){
        k = (2*N-modpow64(3,N-j-1,2*N));
    }
    double real = cos(-pi/N*i*(k));
    double imag = sin(-pi/N*i*(k));
    h_A[i*N+j].x = real;
    h_A[i*N+j].y = imag;
}


__global__ void fft(cuDoubleComplex* h_A,cuDoubleComplex* h_B,int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid;
    h_B[i].x = 0;
    h_B[i].y = 0;
    long double hx = 0;
    long double hy = 0;
    for(int j = 0; j < N; j++){
        
        int k = modpow64(3,j,2*N);
        if(j >= N / 2){
            k = (2*N-modpow64(3,N-j-1,2*N));
        }
        cuDoubleComplex t;
        long double tx = cos(-pi/N*i*(k));
        long double ty = sin(-pi/N*i*(k));
        // if(tid == 100 && j == 100){
        //     printf("%lf\n",t.x);
        // }
        hx += tx * h_A[j].x - ty * h_A[j].y;
        hy += tx * h_A[j].y + ty * h_A[j].x;
    }
    h_B[i].x = hx;
    h_B[i].y = hy;
}


__global__ void initfftinv(cuDoubleComplex* h_A,int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid/N;
    int j = tid%N;
    int k = modpow64(3,j,2*N);
    if(j >= N / 2){
        k = (2*N-modpow64(3,N-j-1,2*N));
    }    
    double real = cos(pi/N*i*(k));
    double imag = sin(pi/N*i*(k));
    h_A[i*N+j].x = real;
    h_A[i*N+j].y = imag;
}

__global__ void ifft(cuDoubleComplex* h_A,cuDoubleComplex* h_B,int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid;
    h_B[i].x = 0;
    h_B[i].y = 0;
    long double hx = 0;
    long double hy = 0;
    for(int j = 0; j < N; j++){
        int k = modpow64(3,i,2*N);
        if(i >= N / 2){
            k = (2*N-modpow64(3,N-i-1,2*N));
        } 
        cuDoubleComplex t;

        long double tx = cos(pi/N*j*(k));
        long double ty = sin(pi/N*j*(k));

        hx += tx * h_A[j].x - ty * h_A[j].y;
        hy += tx * h_A[j].y + ty * h_A[j].x;
    }
    h_B[i].x = hx;
    h_B[i].y = hy;

}
__global__ void product(cuDoubleComplex* h_A){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    h_A[tid].x = h_A[tid].x * h_A[tid].x;
}
class Matrix{
    public:
    int M;
    int N;
    unsigned long long *data;
    Matrix(int M,int N,unsigned long long *data){
        this->M = M;
        this->N = N;
        this->data = data;
    }
};
class BigMatrix{
    public:
    int M;
    int N;
    unsigned long long** data;
    BigMatrix(int M,int N,unsigned long long **data){
        this->M = M;
        this->N = N;
        this->data = data;
    }
};

class ConvKer{
    public:
    int len;
    unsigned long long **data;
    ConvKer(int len,unsigned long long **data){
        this->len = len;
        this->data = data;
    }
};

class Encoder{
    public:
        int n;
        int N;
        double scale;
        unsigned long long** psiTable;
        unsigned long long** psiinvTable; 
        unsigned long long* psi;
        unsigned long long* psiinv;
        unsigned long long* q;
        unsigned long long* mu;
        unsigned long long* q_bit;
        unsigned long long* Qmod;
        unsigned long long* q_hatinv;
        unsigned long long* Pmod;
        unsigned long long* p_hat_inv;
        unsigned long long* Pinv;
        int BATCH = 1;
        int decomposeSize;
        RNS rns;
        // cuDoubleComplex *d_A, *inv_A;


        // cufftHandle cufftForwrdHandle, cufftInverseHandle;
        Encoder(int n,double scale,int decomposeSize):rns(decomposeSize,scale){
            this->n = n;
            this->decomposeSize = decomposeSize;
            N = n * 2;
            // this->N = N;
            this->scale = scale;
            q = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            psi = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            psiinv = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            q_bit = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            Qmod = (unsigned long long *)malloc(decomposeSize * decomposeSize * sizeof(unsigned long long));
            q_hatinv = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            p_hat_inv = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            Pmod = (unsigned long long*)malloc(decomposeSize * decomposeSize * sizeof(unsigned long long *));
            Pinv = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            
            getParams(q, psi, psiinv, q_bit, Qmod,q_hatinv,Pmod,p_hat_inv,Pinv,decomposeSize);
            
            psiTable = (unsigned long long**)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            psiinvTable = (unsigned long long**)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            mu = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            for(int i = 0; i < 2 * decomposeSize; i++){
                Check(mempool(&psiTable[i], N * sizeof(unsigned long long)));
                Check(mempool(&psiinvTable[i], N * sizeof(unsigned long long)));
                // printf("%p,%p\n",psiTable[i],psiinvTable[i]);
            }
            
            for(int i = 0; i < 2 * decomposeSize; i++){
                fillTablePsi128<<<N/1024,1024>>>(psi[i], q[i], psiinv[i], psiTable[i], psiinvTable[i], log2(N));
                uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
                // printf("%llu\n",(mu1/q[i]).low);
                mu[i] = (mu1 / q[i]).low;
            }


            // cudaMalloc (
            //     (void**)&d_A,   
            //     N*N * sizeof(cuDoubleComplex)    
            // );
            // cudaMalloc (
            //     (void**)&inv_A,    
            //     N*N * sizeof(cuDoubleComplex)    
            // );
            // initfft<<<N*N/1024,1024>>>(d_A,N);
            // initfftinv<<<N*N/1024,1024>>>(inv_A,N);
            cudaMemcpyToSymbol(q_const,q,2 * decomposeSize * sizeof(unsigned long long));
            cudaMemcpyToSymbol(mu_const,mu,2 * decomposeSize * sizeof(unsigned long long));
            cudaMemcpyToSymbol(qbit_const,q_bit,2 * decomposeSize * sizeof(unsigned long long));
        };
    unsigned long long* encode(cuDoubleComplex* plainVec){
            cuDoubleComplex *fft_in,*host_in;
            cuDoubleComplex *fft_out;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&host_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&ntt_in, N * decomposeSize * sizeof(unsigned long long)));
            Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            for(int i = 0; i < n; i++){
                host_in[i].x = plainVec[i].x;
                host_in[i].y = plainVec[i].y;
            }
            for(int i = 0; i < n; i++){
                host_in[N-i-1].x = plainVec[i].x;
                host_in[N-i-1].y = -plainVec[i].y;
            }
            
            Check(cudaMemcpy(fft_in, host_in, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;


            fft<<<N/1024,1024>>>(fft_in,fft_out,N);
            // for(int i = 0; i < 1; i++){
            //     cublasZgemv (
            //         handle,    
            //         CUBLAS_OP_T,    
            //         N,    
            //         N,    
            //         &a,    
            //         d_A,    
            //         N,    
            //         fft_in,    
            //         1,    
            //         &b,    
            //         fft_out,    
            //         1   
            //     );
            // }
            ntt_in = rns.decompose(fft_out,N);
            for(int i = 0; i < decomposeSize; i++){
                forwardNTT(ntt_in+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            return ntt_in;
        }
        unsigned long long* encode(double* plainVec){
            cuDoubleComplex *fft_in,*host_in;
            cuDoubleComplex *fft_out;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&host_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&ntt_in, N * decomposeSize * sizeof(unsigned long long)));
            Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            for(int i = 0; i < n; i++){
                host_in[i].x = plainVec[i];
                host_in[i].y = 0;
            }
            for(int i = 0; i < n; i++){
                host_in[N-i-1].x = plainVec[i];
                host_in[N-i-1].y = 0;
            }
            
            Check(cudaMemcpy(fft_in, host_in, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;
            double start = cpuSecond();
            fft<<<N/1024,1024>>>(fft_in,fft_out,N);

            // for(int i = 0; i < 1; i++){
            //     cublasZgemv (
            //         handle,    
            //         CUBLAS_OP_T,    
            //         N,    
            //         N,    
            //         &a,    
            //         d_A,    
            //         N,    
            //         fft_in,    
            //         1,    
            //         &b,    
            //         fft_out,    
            //         1   
            //     );
            // }
    // printf("SGEM Times:%lf seconds\n",(cpuSecond() - start));
            // print<<<1,1>>>(fft_out);
            // product<<<N/1024,1024>>>(fft_out);
            ntt_in = rns.decompose(fft_out,N);
            // print
            // print_ddddd<<<1,1>>>(ntt_in);
            // for(int i = 0; i < N * 8; i += N){
            //     print_d<<<1,1>>>(ntt_in,i);
            // }
            // print_in<<<1,1>>>(ntt_in,16);
            for(int i = 0; i < decomposeSize; i++){
                forwardNTT(ntt_in+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            // print<<<1,1>>>(ntt_in);
            // print_d<<<1,1>>>(ntt_in,0);
            // print_d<<<1,1>>>(ntt_in,2048);
            // print_d<<<1,1>>>(ntt_in,4096);

            return ntt_in;
        }

        unsigned long long* encode_buff(double* plainVec,cuDoubleComplex* fft_in,cuDoubleComplex* fft_out){
            cuDoubleComplex *host_in;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&host_in, N * sizeof(cuDoubleComplex)));
            // Check(mempool((void**)&ntt_in, N * decomposeSize * sizeof(unsigned long long)));
            // Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            // Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            for(int i = 0; i < n; i++){
                host_in[i].x = plainVec[i];
                host_in[i].y = 0;
            }
            for(int i = 0; i < n; i++){
                host_in[N-i-1].x = plainVec[i];
                host_in[N-i-1].y = 0;
            }
            
            Check(cudaMemcpy(fft_in, host_in, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;
            double start = cpuSecond();
            fft<<<N/1024,1024>>>(fft_in,fft_out,N);


            ntt_in = rns.decompose(fft_out,N);

            for(int i = 0; i < decomposeSize; i++){
                forwardNTT(ntt_in+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }

            return ntt_in;
        }
        cuDoubleComplex* decode(unsigned long long* encodeVec,int depth){
            cuDoubleComplex *fft_in,*host_in;
            cuDoubleComplex *fft_out,*fft_out_t;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;

            // Check(cudaMallocHost((void**)&host_in, n * sizeof(cuDoubleComplex)));

            host_in = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));

            Check(mempool((void**)&ntt_in, N * sizeof(unsigned long long)));
            Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out_t, N * sizeof(cuDoubleComplex)));
            // print_e<<<1,1>>>(encodeVec,N/1024);
            // for(int i = 0; i < N * decomposeSize; i += N){
            //     print_e<<<1,1>>>(encodeVec + i,8);
            // }
            for(int i = 0; i < decomposeSize; i++){
                inverseNTT(encodeVec + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }
            
            // print_e<<<1,1>>>(encodeVec,N/1024);
            // for(int i = 0; i < N * decomposeSize; i += N){
            //     print_e<<<1,1>>>(encodeVec + i,8);
            // }
            fft_out = rns.compose(encodeVec,N,1);
            // fft_out = rns.compose_MRS(encodeVec,N,depth);
            // cudaDeviceSynchronize();
            // printf("Times: %lf\n",cpuSecond() - start);
            // print_g<<<1,1>>>(fft_out);      
            // print_h<<<1,1>>>(fft_out);        
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;
            ifft<<<N/1024,1024>>>(fft_out,fft_in,N);
            // print_g<<<1,1>>>(fft_in);      

            // for(int i = 0; i < 1; i++){
            //     cublasZgemv (
            //         handleinv,    
            //         CUBLAS_OP_N,    
            //         N,    
            //         N,    
            //         &a,    
            //         inv_A,   
            //         N,    
            //         fft_out,   
            //         1,    
            //         &b,    
            //         fft_in,   
            //         1   
            //     );
            // }
            Check(cudaMemcpy(host_in, fft_in, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            return host_in;
            // double *res;
            // res = (double *)malloc(n * sizeof(double));
            
            // for(int i = 0; i < n ; i++){
            //     res[i] = host_in[i].x / scale;
            // }
            // return res;
        }
        
        Matrix EncodeMatrix(double* matrixPlain,int row,int col){
            unsigned long long *res;
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);
            if(M * N > n){
                throw "useEncodeBigMatrix";
            }

            double *tmp = (double*)malloc(n * sizeof(double));
            // printf("##\n");
            for(int i = 0; i < n;  i++){
                tmp[i] = 0.0;
            }
            for(int i = 0;i < M;i++){
                for(int j = 0; j < N;j++){
                    if(i >= row || j >= col){
                        tmp[i*N+j] = 0;
                        continue;
                    }
                    tmp[i*N+j] = matrixPlain[j + i * col];
                }
            }
            for(int i = 0; i < n; i++){
                tmp[i] = tmp[i%1024];
            }
            res = encode(tmp);
            return Matrix(M,N,res);
        }
        // BigMatrix EncodeBigMatrix(double* matrixPlain,int row, int col,int padcol){
        //     unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
        //     int M = 1 << ((int)log2(row-0.5)+1);
        //     int N = 1 << ((int)log2(col-0.5)+1);
        //     int K = 1 << ((int)log2(padcol-0.5)+1);
        //     if(M * N <= n){
        //         throw "useEncodeMatrix";
        //     }
        //     // printf("%d\n",K);
        //     double *tmp = (double*)malloc(n * sizeof(double));
        //     int cnt = 0;
        //     for(int idx = 0; idx < row; idx++){
        //         // printf("%d\n",idx);
        //         for(int i = 0; i < n;  i++){
        //             tmp[i] = 0.0;
        //         }
        //         for(int i = 0; ;i++){
        //             for(int j = 0; j < padcol;j++){
        //                 tmp[i*K+j] = matrixPlain[cnt++];
        //                 if(cnt%col==0)break;
        //             }
        //             if(cnt%col==0)break;
        //         } 
        //         // printf("%d,%d\n",cnt,row*col);

        //         res[idx] = encode(tmp);
        //     }
        //     // printf("%d,%d\n",cnt,row*col);
        //     if(cnt!=row*col)throw "error";
        //     // for(int i = 0; i < K; i++){
        //     //     res[i] = encode(tmplist[i]);
        //     // }
        //     return BigMatrix(row,col,res);    
        // }
        BigMatrix EncodeBigMatrix(double* matrixPlain,int row, int col,int padcol){
            unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);
            int K = 1 << ((int)log2(padcol-0.5)+1);
            int collen = col/padcol*K;
            if(M * N <= n){
                throw "useEncodeMatrix";
            }
            // printf("%d\n",K);
            double *tmp = (double*)malloc(n * sizeof(double));
            double *tmp2 = (double*)malloc(n * sizeof(double));
            for(int i=0;i < row;i++){

                for(int j=0;j < n;j++)tmp[j]=0;
                // printf("@@\n");
                for(int j=0;j<collen;j++){
                    double value;
                    int idx = (j + i)% row;
                    if(j%K>=padcol){
                        value=0;
                    }else{
                        value=matrixPlain[idx*col+j/K*padcol+j%K];    
                    }
                    tmp[j] = value;
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                // if(i==1){
                //     for(int j = 0; j < 784;j++){
                //         if(j%32==0)printf("\n\n");
                //         printf("%.1lf ",tmp2[j]);
                        
                //     }
                // }
                res[i] = encode(tmp2);
            }
            return BigMatrix(row,col,res);    
        }
    BigMatrix EncodeBigMatrix_bsgs(double* matrixPlain,int row, int col,int padcol){
            unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);
            int K = 1 << ((int)log2(padcol-0.5)+1);
            int collen = col/padcol*K;

            const int bsgs = 8;
            double *tmp = (double*)malloc(n * sizeof(double));
            double *tmp2 = (double*)malloc(n * sizeof(double));
            for(int i=0;i < row;i++){
                for(int j=0;j < n;j++)tmp[j]=0;
                for(int j=0;j < n;j++)tmp2[j]=0;

                for(int j=0;j<collen;j++){
                    double value;
                    int idx = (j + i)% row;
                    if(j%K>=padcol){
                        value=0;
                    }else{
                        value=matrixPlain[idx*col+j/K*padcol+j%K];    
                    }
                    tmp[j] = value;
                }


                for(int j = 0; j < 1024;j++){
                    tmp2[(i % bsgs+j)%1024] = tmp[j];
                }
                for(int i = 0; i < n; i++){
                    tmp2[i] = tmp2[i % 1024];
                }
                // if(i == 4){
                //     for(int j = 0; j < n; j ++)printf("%d,%lf\t",j,tmp2[j]);
                //     printf("\n");
                // }

                // exit(0);
                res[i] = encode(tmp2);
            }
            return BigMatrix(row,col,res);    
        }
    BigMatrix EncodeBigMatrix_bsgs_test(double* matrixPlain,int row, int col){
            unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);

            const int bsgs = 4;
            double *tmp = (double*)malloc(n * sizeof(double));
            double *tmp2 = (double*)malloc(n * sizeof(double));
            for(int i=0;i < row;i++){
                for(int j=0;j < n;j++)tmp[j]=0;
                for(int j=0;j<col;j++){
                    double value;
                    int idx = (j + i)% row;
                    value=matrixPlain[idx*col + j % col];    
                    tmp[j] = value;
                }

                for(int j = 0; j < col;j++){
                    tmp2[(i % bsgs+j)%n] = tmp[j];
                }
                for(int i = 0; i < n; i++){
                    tmp2[i] = tmp2[i % col];
                }

                res[i] = encode(tmp2);
            }
            return BigMatrix(row,col,res);    
        }
    BigMatrix EncodeBigMatrix_bsgs_test_test(double* matrixPlain,int row, int col){
            unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));


            double *tmp = (double*)malloc(n * sizeof(double));
            double *tmp2 = (double*)malloc(n * sizeof(double));
            for(int i=0;i < row;i++){
                for(int j=0;j < n;j++)tmp[j]=0;
                for(int j=0;j < n;j++)tmp2[j]=0;

                for(int j=0;j<col;j++){
                    double value;
                    int idx = (j + i)% row;
                    if(idx*col + j % col < 640)value=matrixPlain[idx*col + j % col];
                    else value = 0;    
                    tmp[j] = value;
                }

                for(int j = 0; j < col;j++){
                    tmp2[(i%4 +j)%1024] = tmp[j];
                }
                for(int i = 0; i < n; i++){
                    tmp2[i] = tmp2[i % 1024];
                }
                // if(i == 2){
                //     for(int j = 0; j < n; j ++)printf("%d,%lf\t",j,tmp2[j]);
                //     printf("\n");
                // }
                res[i] = encode(tmp2);
            }
            return BigMatrix(row,col,res);    
        }
        BigMatrix Coeff2SlotMatrixV(){
            cuDoubleComplex* V = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            // cuDoubleComplex U_bar[n][n];
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,j,2*N);
                    double real = cos(-pi/N*(i+n)*(k));
                    double imag = sin(-pi/N*(i+n)*(k));
                    V[i * n + j].x = real/n*scale;
                    V[i * n + j].y = imag/n*scale;
                }
            }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = V[(i+j) % n * n + j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }
        BigMatrix Coeff2SlotMatrixU(){
            cuDoubleComplex* U = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            // cuDoubleComplex U_bar[n][n];
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,j,2*N);
                    double real = cos(-pi/N*i*(k));
                    double imag = sin(-pi/N*i*(k));
                    U[i * n + j].x = real/n*scale;
                    U[i * n + j].y = imag/n*scale;
                }
            }
            // for(int i = 0; i < n; i++){
            //     U[i*n+(i+5)%n].x = 1;
            //     U[i*n+i].y = 0;
            // }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = U[((i+j) % n) * n + j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }
        // BigMatrix Coeff2SlotMatrix(){
        //     cuDoubleComplex U[n][n];
        //     // cuDoubleComplex U_bar[n][n];
        //     cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
        //     cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
        //     unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));

        //     for(int i = 0; i < n; i++){
        //         for(int j = 0; j < n; j++){
        //             int k = modpow64(3,j,2*N);
        //             double real = cos(-pi/N*i*(k));
        //             double imag = sin(-pi/N*i*(k));
        //             U[i][j].x = real;
        //             U[i][j].y = imag;
        //         }
        //     }
        //     for(int i = 0; i < n; i++){
        //         for(int j = 0; j < n; j++){
        //             tmp[j] = U[i+j][j];
        //         }
        //         for(int j = 0; j < n;j++){
        //             tmp2[(i+j)%n] = tmp[j];
        //         }
        //         res[i] = encode(tmp2);
        //     }
        //     return res;
        // }
        
        BigMatrix Slot2CoeffMatrixU(){
            cuDoubleComplex *U = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,i,2*N);
                    double real = cos(pi/N*j*(k));
                    double imag = sin(pi/N*j*(k));
                    U[i*n+j].x = real;
                    U[i*n+j].y = imag;
                }
            }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = U[(i+j) % n * n + j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }

        BigMatrix Slot2CoeffMatrixV(){
            cuDoubleComplex *V = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,i,2*N);
                    double real = cos(pi/N*(j+n)*(k));
                    double imag = sin(pi/N*(j+n)*(k));
                    V[i*n+j].x = real;
                    V[i*n+j].y = imag;
                }
            }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = V[(i+j)%n*n+j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }
        // BigMatrix EncodeBigMatrix(double* matrixPlain,int row,int col){
        //     unsigned long long **res = (unsigned long long**)malloc(K * sizeof(unsigned long long*));
        //     int M = 1 << ((int)log2(row-0.5)+1);
        //     int N = 1 << ((int)log2(col-0.5)+1);
        //     if(M * N <= n){
        //         throw "useEncodeMatrix";
        //     }
        //     int K = M * N / n;
        //     double tmplist = (double**)malloc(K * sizeof(double*));
        //     for(int idx = 0; idx < K; idx++){
        //         double *tmp = (double*)malloc(n * sizeof(double));
        //         // printf("##\n");
        //         for(int i = 0; i < n;  i++){
        //             tmp[i] = 0.0;
        //         }
        //         for(int i = 0;i < n/N;i++){
        //             for(int j = 0; j < N;j++){
        //                 if(i + idx * (n/N) >= row || j >= col){
        //                     tmp[i*N+j] = 0;
        //                     continue;
        //                 }
        //                 tmp[i*N+j] = matrixPlain[j + (i + idx * (n/N))* col ];
        //             }
        //         }
        //         tmplist[idx] = tmp;
        //     }
        //     for(int i = 0; i < K; i++){
        //         res[i] = encode(tmplist[i]);
        //     }
        //     return bi
        // }
    
        ConvKer encode(double* matrixPlain,int len){
            unsigned long long **data = (unsigned long long**)malloc(len * sizeof(unsigned long long*));
            double *tmp = (double*)malloc(n * sizeof(double));
            for(int idx = 0; idx < len; idx++){
                for(int i = 0; i < n;i++){
                    tmp[i] = matrixPlain[idx];
                }
                data[idx] = encode(tmp);
            }
            ConvKer res(len,data);
            return res;
        }
        void test(){
            unsigned long long *temp1;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&temp1, N * sizeof(unsigned long long)));
            // Check(cudaMallocHost((void**)&temp2, N * sizeof(unsigned long long)));
            // for(int i = 0; i < 8; i++){
            //     printf("%llu\n",q[i]);
            // }
            genRandom<<<N/1024,1024>>>(temp1,114514);
            // genRandom<<<N/1024,1024>>>(temp2,2);
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);
            // rns.decompose(temp1,N);
            // // rns.decompose(temp2,N);
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);
            forwardNTT(temp1 ,N,ntt,q[0],mu[0],q_bit[0],psiTable[0]);
            // for(int i = 0; i < decomposeSize; i++){
            //     forwardNTT(temp1 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            print<<<1,1>>>(temp1);
            // for(int i = 0; i < decomposeSize; i++){
            //     // forwardNTT(temp2 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);
            // for(int i = 0; i < decomposeSize; i++){
            //     // polymul<<<N/1024,1024>>>(temp1 + N * i,temp2 + N * i,temp1 + N * i,q[i],mu[i],q_bit[i]);
            // }
            // print<<<1,1>>>(temp1);
            // for(int i = 0; i < decomposeSize; i++){
            //     inverseNTT(temp1 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // }
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);

        }
};


class EncoderT{
    public:
        int n;
        int N;
        double scale;
        unsigned long long** psiTable;
        unsigned long long** psiinvTable; 
        unsigned long long* psi;
        unsigned long long* psiinv;
        unsigned long long* q;
        unsigned long long* mu;
        unsigned long long* q_bit;
        unsigned long long* Qmod;
        unsigned long long* q_hatinv;
        unsigned long long* Pmod;
        unsigned long long* p_hat_inv;
        unsigned long long* Pinv;
        int BATCH = 1;
        int decomposeSize;
        RNS rns;

        // cufftHandle cufftForwrdHandle, cufftInverseHandle;
        EncoderT(int n,double scale,int decomposeSize):rns(decomposeSize,scale){
            this->n = n;
            this->decomposeSize = decomposeSize;
            N = n * 2;
            // this->N = N;
            this->scale = scale;
            q = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            psi = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            psiinv = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            q_bit = (unsigned long long *)malloc(2 * decomposeSize * sizeof(unsigned long long));
            Qmod = (unsigned long long *)malloc(decomposeSize * decomposeSize * sizeof(unsigned long long));
            q_hatinv = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            p_hat_inv = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            Pmod = (unsigned long long*)malloc(decomposeSize * decomposeSize * sizeof(unsigned long long *));
            Pinv = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            
            getParams(q, psi, psiinv, q_bit, Qmod,q_hatinv,Pmod,p_hat_inv,Pinv,decomposeSize);
            
            psiTable = (unsigned long long**)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            psiinvTable = (unsigned long long**)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            mu = (unsigned long long*)malloc(2 * decomposeSize * sizeof(unsigned long long *));
            for(int i = 0; i < 2 * decomposeSize; i++){
                Check(mempool(&psiTable[i], N * sizeof(unsigned long long)));
                Check(mempool(&psiinvTable[i], N * sizeof(unsigned long long)));
                // printf("%p,%p\n",psiTable[i],psiinvTable[i]);
            }
            
            for(int i = 0; i < 2 * decomposeSize; i++){
                fillTablePsi128<<<N/1024,1024>>>(psi[i], q[i], psiinv[i], psiTable[i], psiinvTable[i], log2(N));
                uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
                // printf("%llu\n",(mu1/q[i]).low);
                mu[i] = (mu1 / q[i]).low;
            }



            cudaMemcpyToSymbol(q_const,q,2 * decomposeSize * sizeof(unsigned long long));
            cudaMemcpyToSymbol(mu_const,mu,2 * decomposeSize * sizeof(unsigned long long));
            cudaMemcpyToSymbol(qbit_const,q_bit,2 * decomposeSize * sizeof(unsigned long long));
        };
    unsigned long long* encode(cuDoubleComplex* plainVec){
            cuDoubleComplex *fft_in,*host_in;
            cuDoubleComplex *fft_out;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&host_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&ntt_in, N * decomposeSize * sizeof(unsigned long long)));
            Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            for(int i = 0; i < n; i++){
                host_in[i].x = plainVec[i].x;
                host_in[i].y = plainVec[i].y;
            }
            for(int i = 0; i < n; i++){
                host_in[N-i-1].x = plainVec[i].x;
                host_in[N-i-1].y = -plainVec[i].y;
            }
            
            Check(cudaMemcpy(fft_in, host_in, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;


            ntt_in = rns.decompose(fft_in,N);
            for(int i = 0; i < decomposeSize; i++){
                forwardNTT(ntt_in+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            return ntt_in;
        }
        unsigned long long* encode(double* plainVec){
            cuDoubleComplex *fft_in,*host_in;
            cuDoubleComplex *fft_out;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&host_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&ntt_in, N * decomposeSize * sizeof(unsigned long long)));
            Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            for(int i = 0; i < n; i++){
                host_in[i].x = plainVec[i];
                host_in[i].y = 0;
            }
            for(int i = 0; i < n; i++){
                host_in[N-i-1].x = plainVec[i];
                host_in[N-i-1].y = 0;
            }
            
            Check(cudaMemcpy(fft_in, host_in, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;

    // printf("SGEM Times:%lf seconds\n",(cpuSecond() - start));
            // print<<<1,1>>>(fft_out);
            // product<<<N/1024,1024>>>(fft_out);
            ntt_in = rns.decompose(fft_in,N);
            // print
            // print<<<1,1>>>(ntt_in,8);
            // for(int i = 0; i < N * 8; i += N){
            //     print_d<<<1,1>>>(ntt_in,i);
            // }
            for(int i = 0; i < decomposeSize; i++){
                forwardNTT(ntt_in+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            // print<<<1,1>>>(ntt_in);
            // print_d<<<1,1>>>(ntt_in,0);
            // print_d<<<1,1>>>(ntt_in,2048);
            // print_d<<<1,1>>>(ntt_in,4096);

            return ntt_in;
        }
        cuDoubleComplex* decode(unsigned long long* encodeVec,int depth){
            cuDoubleComplex *fft_in,*host_in;
            cuDoubleComplex *fft_out,*fft_out_t;
            unsigned long long *ntt_in;
            cudaStream_t ntt = 0;

            // Check(cudaMallocHost((void**)&host_in, n * sizeof(cuDoubleComplex)));

            host_in = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));

            Check(mempool((void**)&ntt_in, N * sizeof(unsigned long long)));
            Check(mempool((void**)&fft_in, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out, N * sizeof(cuDoubleComplex)));
            Check(mempool((void**)&fft_out_t, N * sizeof(cuDoubleComplex)));

            for(int i = 0; i < decomposeSize; i++){
                inverseNTT(encodeVec + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }
            
            // print<<<1,1>>>(encodeVec);
            // for(int i = 0; i < N * decomposeSize; i += N){
            //     print_d<<<1,1>>>(encodeVec,i);
            // }
            // fft_out_t = rns.compose(encodeVec,N,1);
            // double start = cpuSecond();
            fft_out = rns.compose(encodeVec,N,depth);
            // cudaDeviceSynchronize();
            // printf("Times: %lf\n",cpuSecond() - start);
            // print<<<1,1>>>(fft_out_t);            
            cuDoubleComplex a, b;
            a.x = 1;a.y = 0;b.x = 0;b.y = 0;


            Check(cudaMemcpy(host_in, fft_out, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            return host_in;
            // double *res;
            // res = (double *)malloc(n * sizeof(double));
            
            // for(int i = 0; i < n ; i++){
            //     res[i] = host_in[i].x / scale;
            // }
            // return res;
        }
        
        Matrix EncodeMatrix(double* matrixPlain,int row,int col){
            unsigned long long *res;
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);
            if(M * N > n){
                throw "useEncodeBigMatrix";
            }

            double *tmp = (double*)malloc(n * sizeof(double));
            // printf("##\n");
            for(int i = 0; i < n;  i++){
                tmp[i] = 0.0;
            }
            for(int i = 0;i < M;i++){
                for(int j = 0; j < N;j++){
                    if(i >= row || j >= col){
                        tmp[i*N+j] = 0;
                        continue;
                    }
                    tmp[i*N+j] = matrixPlain[j + i * col];
                }
            }
            for(int i = 0; i < n; i++){
                tmp[i] = tmp[i%1024];
            }
            res = encode(tmp);
            return Matrix(M,N,res);
        }
        // BigMatrix EncodeBigMatrix(double* matrixPlain,int row, int col,int padcol){
        //     unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
        //     int M = 1 << ((int)log2(row-0.5)+1);
        //     int N = 1 << ((int)log2(col-0.5)+1);
        //     int K = 1 << ((int)log2(padcol-0.5)+1);
        //     if(M * N <= n){
        //         throw "useEncodeMatrix";
        //     }
        //     // printf("%d\n",K);
        //     double *tmp = (double*)malloc(n * sizeof(double));
        //     int cnt = 0;
        //     for(int idx = 0; idx < row; idx++){
        //         // printf("%d\n",idx);
        //         for(int i = 0; i < n;  i++){
        //             tmp[i] = 0.0;
        //         }
        //         for(int i = 0; ;i++){
        //             for(int j = 0; j < padcol;j++){
        //                 tmp[i*K+j] = matrixPlain[cnt++];
        //                 if(cnt%col==0)break;
        //             }
        //             if(cnt%col==0)break;
        //         } 
        //         // printf("%d,%d\n",cnt,row*col);

        //         res[idx] = encode(tmp);
        //     }
        //     // printf("%d,%d\n",cnt,row*col);
        //     if(cnt!=row*col)throw "error";
        //     // for(int i = 0; i < K; i++){
        //     //     res[i] = encode(tmplist[i]);
        //     // }
        //     return BigMatrix(row,col,res);    
        // }
        BigMatrix EncodeBigMatrix(double* matrixPlain,int row, int col,int padcol){
            unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);
            int K = 1 << ((int)log2(padcol-0.5)+1);
            int collen = col/padcol*K;
            if(M * N <= n){
                throw "useEncodeMatrix";
            }
            // printf("%d\n",K);
            double *tmp = (double*)malloc(n * sizeof(double));
            double *tmp2 = (double*)malloc(n * sizeof(double));
            for(int i=0;i < row;i++){

                for(int j=0;j < n;j++)tmp[j]=0;
                // printf("@@\n");
                for(int j=0;j<collen;j++){
                    double value;
                    int idx = (j + i)% row;
                    if(j%K>=padcol){
                        value=0;
                    }else{
                        value=matrixPlain[idx*col+j/K*padcol+j%K];    
                    }
                    tmp[j] = value;
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                // if(i==1){
                //     for(int j = 0; j < 784;j++){
                //         if(j%32==0)printf("\n\n");
                //         printf("%.1lf ",tmp2[j]);
                        
                //     }
                // }
                res[i] = encode(tmp2);
            }
            return BigMatrix(row,col,res);    
        }
    BigMatrix EncodeBigMatrix_bsgs(double* matrixPlain,int row, int col,int padcol){
            unsigned long long **res = (unsigned long long**)malloc(row * sizeof(unsigned long long*));
            int M = 1 << ((int)log2(row-0.5)+1);
            int N = 1 << ((int)log2(col-0.5)+1);
            int K = 1 << ((int)log2(padcol-0.5)+1);
            int collen = col/padcol*K;

            const int bsgs = 8;
            double *tmp = (double*)malloc(n * sizeof(double));
            double *tmp2 = (double*)malloc(n * sizeof(double));
            for(int i=0;i < row;i++){
                for(int j=0;j < n;j++)tmp[j]=0;
                for(int j=0;j<collen;j++){
                    double value;
                    int idx = (j + i)% row;
                    if(j%K>=padcol){
                        value=0;
                    }else{
                        value=matrixPlain[idx*col+j/K*padcol+j%K];    
                    }
                    tmp[j] = value;
                }

                for(int j = 0; j < n;j++){
                    tmp2[(i % bsgs+j)%n] = tmp[j];
                }
                for(int i = 0; i < n; i++){
                    tmp2[i] = tmp2[i % 1024];
                }
                // exit(0);
                res[i] = encode(tmp2);
            }
            return BigMatrix(row,col,res);    
        }

        BigMatrix Coeff2SlotMatrixV(){
            cuDoubleComplex* V = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            // cuDoubleComplex U_bar[n][n];
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,j,2*N);
                    double real = cos(-pi/N*(i+n)*(k));
                    double imag = sin(-pi/N*(i+n)*(k));
                    V[i * n + j].x = real/n*scale;
                    V[i * n + j].y = imag/n*scale;
                }
            }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = V[(i+j) % n * n + j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }
        BigMatrix Coeff2SlotMatrixU(){
            cuDoubleComplex* U = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            // cuDoubleComplex U_bar[n][n];
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,j,2*N);
                    double real = cos(-pi/N*i*(k));
                    double imag = sin(-pi/N*i*(k));
                    U[i * n + j].x = real/n*scale;
                    U[i * n + j].y = imag/n*scale;
                }
            }
            // for(int i = 0; i < n; i++){
            //     U[i*n+(i+5)%n].x = 1;
            //     U[i*n+i].y = 0;
            // }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = U[((i+j) % n) * n + j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }
        // BigMatrix Coeff2SlotMatrix(){
        //     cuDoubleComplex U[n][n];
        //     // cuDoubleComplex U_bar[n][n];
        //     cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
        //     cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
        //     unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));

        //     for(int i = 0; i < n; i++){
        //         for(int j = 0; j < n; j++){
        //             int k = modpow64(3,j,2*N);
        //             double real = cos(-pi/N*i*(k));
        //             double imag = sin(-pi/N*i*(k));
        //             U[i][j].x = real;
        //             U[i][j].y = imag;
        //         }
        //     }
        //     for(int i = 0; i < n; i++){
        //         for(int j = 0; j < n; j++){
        //             tmp[j] = U[i+j][j];
        //         }
        //         for(int j = 0; j < n;j++){
        //             tmp2[(i+j)%n] = tmp[j];
        //         }
        //         res[i] = encode(tmp2);
        //     }
        //     return res;
        // }
        
        BigMatrix Slot2CoeffMatrixU(){
            cuDoubleComplex *U = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,i,2*N);
                    double real = cos(pi/N*j*(k));
                    double imag = sin(pi/N*j*(k));
                    U[i*n+j].x = real;
                    U[i*n+j].y = imag;
                }
            }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = U[(i+j) % n * n + j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }

        BigMatrix Slot2CoeffMatrixV(){
            cuDoubleComplex *V = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));;
            cuDoubleComplex *tmp = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            cuDoubleComplex *tmp2 = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
            unsigned long long **res = (unsigned long long**)malloc(n * sizeof(unsigned long long*));
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    int k = modpow64(3,i,2*N);
                    double real = cos(pi/N*(j+n)*(k));
                    double imag = sin(pi/N*(j+n)*(k));
                    V[i*n+j].x = real;
                    V[i*n+j].y = imag;
                }
            }
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    tmp[j] = V[(i+j)%n*n+j];
                }
                for(int j = 0; j < n;j++){
                    tmp2[(i+j)%n] = tmp[j];
                }
                res[i] = encode(tmp2);
            }
            return BigMatrix(n,n,res);
        }
        // BigMatrix EncodeBigMatrix(double* matrixPlain,int row,int col){
        //     unsigned long long **res = (unsigned long long**)malloc(K * sizeof(unsigned long long*));
        //     int M = 1 << ((int)log2(row-0.5)+1);
        //     int N = 1 << ((int)log2(col-0.5)+1);
        //     if(M * N <= n){
        //         throw "useEncodeMatrix";
        //     }
        //     int K = M * N / n;
        //     double tmplist = (double**)malloc(K * sizeof(double*));
        //     for(int idx = 0; idx < K; idx++){
        //         double *tmp = (double*)malloc(n * sizeof(double));
        //         // printf("##\n");
        //         for(int i = 0; i < n;  i++){
        //             tmp[i] = 0.0;
        //         }
        //         for(int i = 0;i < n/N;i++){
        //             for(int j = 0; j < N;j++){
        //                 if(i + idx * (n/N) >= row || j >= col){
        //                     tmp[i*N+j] = 0;
        //                     continue;
        //                 }
        //                 tmp[i*N+j] = matrixPlain[j + (i + idx * (n/N))* col ];
        //             }
        //         }
        //         tmplist[idx] = tmp;
        //     }
        //     for(int i = 0; i < K; i++){
        //         res[i] = encode(tmplist[i]);
        //     }
        //     return bi
        // }
    
        ConvKer encode(double* matrixPlain,int len){
            unsigned long long **data = (unsigned long long**)malloc(len * sizeof(unsigned long long*));
            double *tmp = (double*)malloc(n * sizeof(double));
            for(int idx = 0; idx < len; idx++){
                for(int i = 0; i < n;i++){
                    tmp[i] = matrixPlain[idx];
                }
                data[idx] = encode(tmp);
            }
            ConvKer res(len,data);
            return res;
        }
        void test(){
            unsigned long long *temp1;
            cudaStream_t ntt = 0;
            Check(cudaMallocHost((void**)&temp1, N * sizeof(unsigned long long)));
            // Check(cudaMallocHost((void**)&temp2, N * sizeof(unsigned long long)));
            // for(int i = 0; i < 8; i++){
            //     printf("%llu\n",q[i]);
            // }
            genRandom<<<N/1024,1024>>>(temp1,114514);
            // genRandom<<<N/1024,1024>>>(temp2,2);
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);
            // rns.decompose(temp1,N);
            // // rns.decompose(temp2,N);
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);
            forwardNTT(temp1 ,N,ntt,q[0],mu[0],q_bit[0],psiTable[0]);
            // for(int i = 0; i < decomposeSize; i++){
            //     forwardNTT(temp1 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            print<<<1,1>>>(temp1);
            // for(int i = 0; i < decomposeSize; i++){
            //     // forwardNTT(temp2 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);
            // for(int i = 0; i < decomposeSize; i++){
            //     // polymul<<<N/1024,1024>>>(temp1 + N * i,temp2 + N * i,temp1 + N * i,q[i],mu[i],q_bit[i]);
            // }
            // print<<<1,1>>>(temp1);
            // for(int i = 0; i < decomposeSize; i++){
            //     inverseNTT(temp1 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // }
            // print<<<1,1>>>(temp1);
            // print<<<1,1>>>(temp2);

        }
};