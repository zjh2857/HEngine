#pragma once

#include "polycalc.cuh"
#include "encryptor.cuh"
#include "helper.cuh"
#include <algorithm>
#include "rottable.cuh"
__global__ void print(unsigned long long* a);
__global__ void print(unsigned long long* a,int size);
__global__ void print_dd(unsigned long long* a,int size){
    printf("*********\n");

    for(int i = 0; i < size; i++){
        // printf("%llu\n",a[i * 8192 + 0]);
    }
    printf("\n");
    // for(int i = 0; i < size; i++){
    //     printf("%llu\n",a[i * 8192 + 1]);
    // }
    // printf("\n");

    // for(int i = 0; i < size; i++){
    //     printf("%llu\n",a[i * 8192 + 2]);
    // }
    printf("=======\n");
}
__global__ void print_z(unsigned long long* a,unsigned long long* b,int N);
__global__ void print_dddd(unsigned long long* a){
    printf("ddddd*********\n");

    for(int i = 0; i < 64; i++){
        if(a[i]!=0)printf("%d,%llu\t",i,a[i]);
    }
    printf("=======\n");
}
__global__ void set(unsigned long long* a){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    a[tid] = tid;
}
const int bsgs = 8; 
const int stream_num = 32;
int* rot1;
int* rot1r;
int* rot2r;

class CudaParam{
    public:
    unsigned long long* q;
    unsigned long long* Qmod;
    unsigned long long* q_hatinv;
    unsigned long long* Pmod;
    unsigned long long* p_hatinv;
    unsigned long long* Pinv;
    unsigned long long* spinv;
    unsigned long long size;
    CudaParam(int size){
        this->size = size;
        Check(mempool(&q, 2 * size * sizeof(unsigned long long)));
        Check(mempool(&Qmod, size * size * sizeof(unsigned long long)));
        Check(mempool(&q_hatinv, size * sizeof(unsigned long long)));
        Check(mempool(&Pmod, size * size * sizeof(unsigned long long)));
        Check(mempool(&p_hatinv, size * sizeof(unsigned long long)));
        Check(mempool(&Pinv, size * sizeof(unsigned long long)));
        Check(mempool(&spinv, size * sizeof(unsigned long long)));
    }
    void set(unsigned long long* q,unsigned long long* Qmod,unsigned long long* q_hatinv,unsigned long long* Pmod,unsigned long long* p_hatinv, unsigned long long* Pinv){
        unsigned long long *spinv = (unsigned long long*)malloc(size * sizeof(unsigned long long));
        for(int i = 0; i < size; i++){
            unsigned long long p = modpow128(q[size],q[i]-2,q[i]);
            spinv[i] = p;
        }
        
        cudaMemcpy(this->q,q,2 * size * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(this->Qmod,Qmod,size * size * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(this->q_hatinv,q_hatinv, size * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(this->Pmod,Pmod,size * size * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(this->p_hatinv,p_hatinv, size * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(this->Pinv,Pinv, size * sizeof(unsigned long long),cudaMemcpyHostToDevice);
        cudaMemcpy(this->spinv,spinv, size * sizeof(unsigned long long),cudaMemcpyHostToDevice);

    }
};

class Evaluator{
    public:
    int N;
    int n;
    unsigned long long** psiTable;
    unsigned long long** psiinvTable;   
    unsigned long long** psiRotTable;
    unsigned long long* psi;
    unsigned long long* psiinv;
    unsigned long long* q;
    unsigned long long* mu;
    unsigned long long* ninv;
    unsigned long long* q_bit;
    unsigned long long* Qmod;
    unsigned long long size;
    unsigned long long* q_hatinv;
    unsigned long long* Pmod;
    unsigned long long* p_hatinv;
    unsigned long long* Pinv;
    unsigned long long* psiList;
    unsigned long long* psiinvList;
    unsigned long long ** rottable;
    unsigned long long ** rottabler;
    unsigned long long *spinv_rescale_device;
    double scale;
    CudaParam cudaParam;
    cudaStream_t streams[stream_num];

    // RNS rns;
    Encoder encoder;
    dim3 dim;

    cudaStream_t ntt = 0;

    Evaluator(int n,double scale,int size):dim(2*n/1024, size),encoder(n,scale,size),cudaParam(size){
        N = n * 2;
        this->n = n;
        this->size = size;
        this->scale = scale;
        q = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        psi = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        psiinv = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        q_bit = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        psiTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));
        psiinvTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));
        psiRotTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));
        q_hatinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
        Qmod = (unsigned long long*)malloc(size * size * sizeof(unsigned long long *));
        
        Pmod = (unsigned long long*)malloc(size * size * sizeof(unsigned long long *));
        p_hatinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
        Pinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
        rottable = (unsigned long long**)malloc(N * sizeof(unsigned long long*));
        rottabler = (unsigned long long**)malloc(N * sizeof(unsigned long long*));

        getParams(q, psi, psiinv, q_bit, Qmod,q_hatinv,Pmod,p_hatinv,Pinv,size);

        cudaParam.set(q,Qmod, q_hatinv,Pmod, p_hatinv, Pinv);
        mu = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long));
        Check(mempool(&psiList, 2 * size * N * sizeof(unsigned long long)));
        Check(mempool(&psiinvList,2 * size * N * sizeof(unsigned long long)));
        for(int i = 0; i < 2 * size; i++){
            Check(mempool(&psiTable[i], N * sizeof(unsigned long long)));
            Check(mempool(&psiinvTable[i], N * sizeof(unsigned long long)));
            Check(mempool(&psiRotTable[i], N * sizeof(unsigned long long)));
        }
        for(int i = 0; i <2 * size; i++){
            fillTablePsi128<<<N/1024,1024>>>(psi[i], q[i], psiinv[i], psiTable[i], psiinvTable[i], log2(N));
            fillTablePsi128<<<N/1024,1024>>>(psi[i], q[i], psiinv[i], psiList + i * N, psiinvList + i * N, log2(N));

            uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
            mu[i] = (mu1 / q[i]).low;
        }
        for(int i = 0; i <2 * size; i++){
            // unsigned long long psiRot = modpow128(psi[i],5,q[i]);
            fillTablePsi128Rot<<<N/1024,1024>>>(q[i],psiTable[i],psiRotTable[i]);
        }
        
        // cudaMalloc(&rot1,N*sizeof(int));
        // cudaMalloc(&rot1r,N*sizeof(int));
        // cudaMalloc(&rot2r,N*sizeof(int));
        // cudaMemcpy(rot1,rot1h,8192*sizeof(int),cudaMemcpyHostToDevice);
        // cudaMemcpy(rot1r,rot1rh,8192*sizeof(int),cudaMemcpyHostToDevice);
        // cudaMemcpy(rot2r,rot2rh,8192*sizeof(int),cudaMemcpyHostToDevice);
        // for(int i = 0; i < N / 8192; i++){
        //     cudaMemcpy(rot1+i*8192,rot1h,8192*sizeof(int),cudaMemcpyHostToDevice);
        //     cudaMemcpy(rot1r+i*8192,rot1rh,8192*sizeof(int),cudaMemcpyHostToDevice);
        //     cudaMemcpy(rot2r+i*8192,rot2rh,8192*sizeof(int),cudaMemcpyHostToDevice);
        // }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %sn", cudaGetErrorString(err));
        }
        unsigned long long *seq,*seq_tmp;
        mempool(&seq,N*sizeof(unsigned long long));

        set<<<N/1024,1024>>>(seq);
        inverseNTT(seq,N,ntt,q[0],mu[0],q_bit[0],psiinvTable[0]);
        // for(int i = 0; i < 128; i++){
        //     int elt = modpow128(3,i,(2*N));
        //     cudarotation<<<N/1024,1024>>>(seq,seq_tmp,q[0],elt,N);
        //     forwardNTT(seq_tmp,N,ntt,q[0],mu[0],q_bit[0],psiTable[0]);
        //     rottabler[i] = seq_tmp;
        // }
        for(int i = 1; i < n; i*=2){
            mempool(&seq_tmp,N*sizeof(unsigned long long));
            int elt = modpow128(3,i,(2*N));
            cudarotation<<<N/1024,1024>>>(seq,seq_tmp,q[0],elt,N);
            forwardNTT(seq_tmp,N,ntt,q[0],mu[0],q_bit[0],psiTable[0]);
            rottabler[i] = seq_tmp;
        }


        // for(int i = 0; i < 128; i++){
        //     int elt = modpow128(3,2*N-i,(2*N));
        //     cudarotation<<<N/1024,1024>>>(seq,seq_tmp,q[0],elt,N);
        //     forwardNTT(seq_tmp,N,ntt,q[0],mu[0],q_bit[0],psiTable[0]);
        //     rottable[i] = seq_tmp;
        // }
        for(int i = 1; i < n; i*=2){
            mempool(&seq_tmp,N*sizeof(unsigned long long));

            int elt = modpow128(3,2*N-i,(2*N));
            // if(i == 1)print_dddd<<<1,1>>>(seq);

            cudarotation<<<N/1024,1024>>>(seq,seq_tmp,q[0],elt,N);
            // if(i == 1)print_dddd<<<1,1>>>(seq_tmp);

            // printf("AAAAAAAAA%llu\n",q[3]);
            forwardNTT(seq_tmp,N,ntt,q[0],mu[0],q_bit[0],psiTable[0]);
            // if(i == 1)print_dddd<<<1,1>>>(seq_tmp);
            rottable[i] = seq_tmp;
        }

        unsigned long long *spinv_rescale = (unsigned long long*)malloc(size * size * sizeof(unsigned long long));
        for(int i = 0; i < size; i++){
            for(int j = i+1; j < size; j++){
                unsigned long long p = modpow128(q[i],q[j]-2,q[j]);
                // printf("%llu,%llu,%llu\n",q[i],q[j],p);
                spinv_rescale[i*size+j] = p;
            }
        }
        Check(mempool(&spinv_rescale_device, size * size * sizeof(unsigned long long)));
        cudaMemcpy(spinv_rescale_device,spinv_rescale, size * size * sizeof(unsigned long long),cudaMemcpyHostToDevice);


        for(int i = 0; i < stream_num; i++){
            cudaStreamCreate(&streams[i]);
        }
    }
    cipherText* convbatch(int row,int col,int w,int h, cipherText* cipherlist,double *ker){

        int r = row-w;
        int c = col-h;
        cipherText* res = (cipherText*)malloc((row-w)*(col-h)*sizeof(cipherText));
        
        cudaStream_t stream[stream_num];
        for(int i = 0; i < stream_num; i++){
            cudaStreamCreate(&stream[i]);
        }
        for(int i = 0; i < row-w; i++) {
            for(int j = 0; j < col-h; j++){
                Check(mempool((void**)&res[i*c+j].a , N * size * sizeof(unsigned long long)));
                Check(mempool((void**)&res[i*c+j].b , N * size * sizeof(unsigned long long)));
                res[i*c+j].depth = cipherlist[0].depth;
                for(int x = 0; x < w; x++){
                    for(int y = 0; y < h; y++){
                        if(cipherlist[(i+x)*col+j+y].a == 0)continue;
                        polymuladdscalar<<<N/1024,1024,0,stream[(i*c+j)%stream_num]>>>(cipherlist[(i+x)*col+j+y].a,res[i*c+j].a,cipherlist[(i+x)*col+j+y].b,res[i*c+j].b,ker[x*h+y]*scale,size);
                    }
                }
            }
        }

        for(int i = 0; i < stream_num; i++){
            cudaStreamSynchronize(stream[i]); 
            cudaStreamDestroy(stream[i]);
        }

        return res;
    }
    cipherText* dotbatch(int row,int col, cipherText* cipherlist,double *ker){
        cipherText* res = (cipherText*)malloc(row*sizeof(cipherText));
        cudaStream_t stream[stream_num];
        for(int i = 0; i < stream_num; i++){
            cudaStreamCreate(&stream[i]);
        }
        for(int i = 0; i < row; i++){
            
            Check(mempool((void**)&res[i].a , N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&res[i].b , N * size * sizeof(unsigned long long)));
            res[i].depth = cipherlist[0].depth;
            for(int j = 0; j < col; j++){
                polymuladdscalar<<<N/1024,1024,0,stream[i%stream_num]>>>(cipherlist[j].a,res[i].a,cipherlist[j].b,res[i].b,ker[i*col+j]*scale,size);
            }
        }
        for(int i = 0; i < stream_num; i++){
            cudaStreamSynchronize(stream[i]); 
            cudaStreamDestroy(stream[i]);
        }
        return res;  
    }    
    cipherText* dotbatch_old(int row,int col, cipherText* cipherlist,double *ker){
        cipherText* res = (cipherText*)malloc(row*sizeof(cipherText));

        for(int i = 0; i < row; i++){
            
            Check(mempool((void**)&res[i].a , N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&res[i].b , N * size * sizeof(unsigned long long)));
            res[i].depth = cipherlist[0].depth;
            for(int j = 0; j < col; j++){
                polymuladdscalar<<<N/1024,1024>>>(cipherlist[j].a,res[i].a,cipherlist[j].b,res[i].b,ker[i*col+j]*scale,size);
            }
        }

        return res;  
    }   
    void addbatch(int row,cipherText* cipherlist,double scalesize,double *ker){
        for(int i = 0; i < row; i++){
            polyaddcalar<<<N/1024,1024>>>(cipherlist[i].a,ker[i]*scalesize,size);
        }
    }   
    void addPlain(cipherText& cipter, unsigned long long* plain){
        unsigned long long *a = cipter.a;
        polyaddsingle<<<dim,1024>>>(a,plain,a);
    }
    void addcipter(cipherText& cipher1, cipherText cipher2){
        // unsigned long long *a = cipter.a;
        polyadddouble<<<dim,1024>>>(cipher1.a,cipher2.a,cipher1.a,cipher1.b,cipher2.b,cipher1.b);

        cipher1.depth = max(cipher1.depth,cipher2.depth);
    }
    // void addcipter(cipherText cipter1, cipherText cipter2){

    //     polyadd<<<N/1024,1024>>>(cipter1.a,cipter2.a,cipter1.a,N,q);
        
    //     polyadd<<<N/1024,1024>>>(cipter1.b,cipter2.b,cipter1.b,N,q);
    // }
    void mulPlain(cipherText& cipher, unsigned long long* plain){
        polymuldouble<<<dim,1024>>>(cipher.a,plain,cipher.a,cipher.b,plain,cipher.b);

        rescale(cipher);
    }
    void mulPlain_lazy(cipherText& cipher, unsigned long long* plain){
        polymuldouble<<<dim,1024>>>(cipher.a,plain,cipher.a,cipher.b,plain,cipher.b);

        // rescale(cipher);
    }
    void mulPlain(unsigned long long* plain1, unsigned long long* plain2){
        // print<<<1,1>>>(plain1+2 * N + (8448 - 8190));
        for(int i = 0; i < size; i++){
            polymul<<<N/1024,1024>>>(plain1 + N * i,plain2 + N * i,plain1 + N * i,q[i],mu[i],q_bit[i]);
        }
        // for()
        // print<<<1,1>>>(plain1+2 * N + (8448 - 8190));
        
        // <<<N/1024,1024>>>(cipter.b,plain,cipter.b,q,mu,q_bit);
    }

    void mulPlain_new1(cipherText& ciphertmp,cipherText& cipher, unsigned long long* plain){

        
        polymuldouble<<<dim,1024>>>(cipher.a,plain,ciphertmp.a,cipher.b,plain,ciphertmp.b);

        ciphertmp.depth = cipher.depth;
        rescale(ciphertmp);
        // print<<<1,1>>>(ciphertmp.b);  
    }

    void mulPlain_new1_lazy(cipherText& ciphertmp,cipherText& cipher, unsigned long long* plain){

        
        polymuldouble<<<dim,1024>>>(cipher.a,plain,ciphertmp.a,cipher.b,plain,ciphertmp.b);

        ciphertmp.depth = cipher.depth;
        // print<<<1,1>>>(ciphertmp.b);  
    }
    void cipherClear(cipherText& cipher){
        for(int i = 0; i < size; i++){
            makezero<<<N/1024,1024>>>(cipher.a + N * i);
            makezero<<<N/1024,1024>>>(cipher.b + N * i);
        }
        cipher.depth = 0;
    }
    void rescale1(cipherText& cipher){
        if(cipher.depth==7)return;

        inverseNTT_batch(cipher.a, N, psiinvList, size, size);
        inverseNTT_batch(cipher.b, N, psiinvList, size, size);

        cudaRescale_fusion<<<dim,1024>>>(cipher.a,cipher.a,cipher.depth,size,N,spinv_rescale_device);
        cudaRescale_fusion<<<dim,1024>>>(cipher.b,cipher.b,cipher.depth,size,N,spinv_rescale_device);

        forwardNTT_batch(cipher.a, N, psiList, size, size);
        forwardNTT_batch(cipher.b, N, psiList, size, size);

        cipher.depth = cipher.depth+1;
    }

    void rescale(cipherText& cipher){
        if(cipher.depth==7)return;
        int dep = cipher.depth;
        // inverseNTT_batch(cipher.a + dep * N, N, psiinvList + N * dep, 1, 1);
        // inverseNTT_batch(cipher.b + dep * N, N, psiinvList + N * dep, 1, 1);

        inverseNTT(cipher.a + N * dep,N,ntt,q[dep],mu[dep],q_bit[dep],psiinvTable[dep]);
        inverseNTT(cipher.b + N * dep,N,ntt,q[dep],mu[dep],q_bit[dep],psiinvTable[dep]);

        unsigned long long *tmp_a,*tmp_b;
        Check(mempool((void**)&tmp_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b, N * size * sizeof(unsigned long long)));

        // cudaConvMov_rescale<<<N/1024,1024>>>(N,tmp_a,cipher.a,dep,size);
        // cudaConvMov_rescale<<<N/1024,1024>>>(N,tmp_b,cipher.b,dep,size);
        cudaConvMov_rescale<<<N/1024,1024>>>(N,tmp_a,cipher.a,tmp_b,cipher.b,dep,size);

        forwardNTT_batch(tmp_a, N, psiList, size, size);
        forwardNTT_batch(tmp_b, N, psiList, size, size);

        // cudaRescale_fusion_opt<<<dim,1024>>>(cipher.a,tmp_a,cipher.depth,size,N,spinv_rescale_device);
        // cudaRescale_fusion_opt<<<dim,1024>>>(cipher.b,tmp_b,cipher.depth,size,N,spinv_rescale_device);

        cudaRescale_fusion_opt<<<dim,1024>>>(cipher.a,tmp_a,cipher.b,tmp_b,cipher.depth,size,N,spinv_rescale_device);
        
        cipher.depth = cipher.depth+1;
    }
    void rescale_buff(cipherText& cipher,unsigned long long *tmp_a,unsigned long long *tmp_b){
        if(cipher.depth==7)return;
        int dep = cipher.depth;
        inverseNTT(cipher.a + N * dep,N,ntt,q[dep],mu[dep],q_bit[dep],psiinvTable[dep]);
        inverseNTT(cipher.b + N * dep,N,ntt,q[dep],mu[dep],q_bit[dep],psiinvTable[dep]);
        // unsigned long long *tmp_a,*tmp_b;
        // Check(mempool((void**)&tmp_a, N * size * sizeof(unsigned long long)));
        // Check(mempool((void**)&tmp_b, N * size * sizeof(unsigned long long)));

        cudaConvMov_rescale<<<N/1024,1024>>>(N,tmp_a,cipher.a,dep,size);
        cudaConvMov_rescale<<<N/1024,1024>>>(N,tmp_b,cipher.b,dep,size);

        forwardNTT_batch(tmp_a, N, psiList, size, size);
        forwardNTT_batch(tmp_b, N, psiList, size, size);

        cudaRescale_fusion_opt<<<dim,1024>>>(cipher.a,tmp_a,cipher.depth,size,N,spinv_rescale_device);
        cudaRescale_fusion_opt<<<dim,1024>>>(cipher.b,tmp_b,cipher.depth,size,N,spinv_rescale_device);


        cipher.depth = cipher.depth+1;
    }
    void rescale(cipherText& cipher,cudaStream_t stream){
        // printf("rescale %d\n",cipher.depth);
        if(cipher.depth==7)return;
        // for(int i = 0; i < size; i++){
            // inverseNTT(cipher.a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        // }
        inverseNTT_batch(cipher.a, N, psiinvList, size, size);
        inverseNTT_batch(cipher.b, N, psiinvList, size, size);

        // for(int i = 0; i < 8 * N; i += N){
        //     print_d<<<1,1>>>(cipher.a,i);
        // }
        // auto res = rns.compose(cipher.a,N);
        // print<<<1,1>>>(res);

        for(int i = cipher.depth + 1; i < size; i++){
            unsigned long long qinv = modpow128(q[cipher.depth],q[i]-2,q[i]);
            
            cudaRescale<<<N/1024,1024,0,stream>>>(cipher.a + i * N,cipher.a + cipher.depth * N,q[i],mu[i],q_bit[i],qinv);

            cudaRescale<<<N/1024,1024,0,stream>>>(cipher.b + i * N,cipher.b + cipher.depth * N,q[i],mu[i],q_bit[i],qinv);
        }
        // for(int i = 0; i < 8 * N; i += N){
        //     print_d<<<1,1>>>(cipher.a,i);
        // }
        // res = rns.compose(cipher.a,N);
        // print<<<1,1>>>(res);
        // if(cipher.depth==0){
        //     printf("modraise\n");
        //     ModRaise(cipher);
        // }
        // for(int i = 0; i < size; i++){
        //     forwardNTT(cipher.a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     forwardNTT(cipher.b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        forwardNTT_batch(cipher.a, N, psiList, size, size);
        forwardNTT_batch(cipher.b, N, psiList, size, size);

        cipher.depth = cipher.depth+1;
        // printf("%d\n",cipher.depth);
    }
    void rescale(unsigned long long* cipher){
        for(int i = 0; i < size; i++){
            inverseNTT(cipher + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        // auto res = rns.compose(cipher.a,N);
        // print<<<1,1>>>(res);
        // for(int i = 0; i < 8 * N; i += N){
        //     print_d<<<1,1>>>(cipher,i);
        // }
        for(int i = 1; i < size; i++){
            unsigned long long qinv = modpow128(q[0],q[i]-2,q[i]);
            
            cudaRescale<<<N/1024,1024>>>(cipher + i * N,cipher,q[i],mu[i],q_bit[i],qinv);

            // cudaRescale<<<N/1024,1024>>>(cipher + i * N,cipher + 1 * N,q[i],mu[i],q_bit[i],qinv);
        }
        // for(int i = 0; i < 8 * N; i += N){
        //     print_d<<<1,1>>>(cipher,i);
        // }

        for(int i = 0; i < size; i++){
            forwardNTT(cipher+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        // cipher.depth++;
    }
    triplePoly mulcipter(cipherText& cipter1, cipherText& cipter2){
        unsigned long long *a,*b,*c;


        Check(mempool((void**)&a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&c, N * size * sizeof(unsigned long long)));
        polymultriple<<<dim,1024>>>(cipter1.a,cipter1.b,cipter2.a,cipter2.b,a,b,c);

        triplePoly res;
        res.set(a,b,c);
        res.depth = max(cipter1.depth,cipter2.depth);
        return res;
    }
    triplePoly mulcipter(cipherText& cipter1, cipherText& cipter2,cudaStream_t stream){
        unsigned long long *a,*b,*c;


        Check(mempool((void**)&a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&c, N * size * sizeof(unsigned long long)));
        polymultriple<<<dim,1024,0,stream>>>(cipter1.a,cipter1.b,cipter2.a,cipter2.b,a,b,c);

        triplePoly res;
        res.set(a,b,c);
        res.depth = max(cipter1.depth,cipter2.depth);
        return res;
    }
    triplePoly mulcipter(cipherText& cipter1, cipherText& cipter2,unsigned long long*a,unsigned long long* b,unsigned long long* c){
        // unsigned long long *a,*b,*c;

        polymultriple<<<dim,1024>>>(cipter1.a,cipter1.b,cipter2.a,cipter2.b,a,b,c);

        triplePoly res;
        res.set(a,b,c);
        res.depth = max(cipter1.depth,cipter2.depth);
        return res;
    }
    // cipherText mulcipter(cipherText& cipter1, cipherText& cipter2,privateKey key){
    //     unsigned long long *a,*b,*c,*s_2,*res;

    //     Check(mempool((void**)&a, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&c, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&s_2, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));
    //     // print<<<1,1>>>(cipter2.b);
    //     // for(int i = 0; i < size; i++){
    //     //     polymul<<<N/1024,1024>>>(cipter1.b,cipter2.b,c,q[i],mu[i],q_bit[i]);
    //     //     polymul<<<N/1024,1024>>>(cipter1.a,cipter2.b,b,q[i],mu[i],q_bit[i]);
    //     //     polymuladd<<<N/1024,1024>>>(cipter1.b,cipter2.a,b,b,q[i],mu[i],q_bit[i]);
    //     //     polymul<<<N/1024,1024>>>(cipter1.a,cipter2.a,a,q[i],mu[i],q_bit[i]);
    //     //     polymul<<<N/1024,1024>>>(key.b,key.b,s_2,q[i],mu[i],q_bit[i]);
    //     // }
    //     for(int i = 0; i < size; i++){
    //         polymuladd<<<N/1024,1024>>>(s_2 + N * i,c + N * i,a + N * i,res + N * i,q[i],mu[i],q_bit[i]);
    //     }        
    //     triplePoly res;
    //     res.set(a,b,c);
    //     return res;
    // }



    cipherText relien(triplePoly cipher,relienKey key){
        unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
        unsigned long long* d2 = Modup(cipher.c);
        // unsigned long long* d2 = cipher.c;

        // for(int i = 0; i < 2 * size; i++){
        //     polymul<<<N/1024,1024>>>(d2 + N * i,key.a + N * i,a + N * i,q[i],mu[i],q_bit[i]);
        //     polymul<<<N/1024,1024>>>(d2 + N * i,key.b + N * i,b + N * i,q[i],mu[i],q_bit[i]);
        // }
        dim3 dimdouble(N/1024,size);
        polymuldouble<<<dimdouble,1024>>>(d2,key.a,a,d2,key.b,b);


        unsigned long long* a_down = Moddown(a);
        unsigned long long* b_down = Moddown(b);
        // for(int i = 0; i < size; i++){
        //     polyadd<<<N/1024,1024>>>(cipher.a + N * i,a_down + N * i,a_down + N * i, N,q[i]);
        //     polyadd<<<N/1024,1024>>>(cipher.b + N * i,b_down + N * i,b_down + N * i, N,q[i]);
        // }
        polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);
        // polyaddsingle<<<dim,1024>>>(cipher.a,a_down,a_down);
        // polyaddsingle<<<dim,1024>>>(b_down,cipher.b,b_down);

        // print<<<1,1>>>(a_down,8);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }

    cipherText relien_dcomp_old(triplePoly cipher,relienKey* key){
        unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
        unsigned long long** d2s = Modup_dcomp_batch2(cipher.c);   

        dim3 dimdouble(N/1024,size+1);
        for(int i = 0; i < 1;i++)polymuldouble<<<dimdouble,1024>>>(d2s[i],key[i].a,a,d2s[i],key[i].b,b);


        // unsigned long long *a_down;
        // Check(mempool((void**)&a_down, N * size * sizeof(unsigned long long)));
        // unsigned long long *b_down;
        // Check(mempool((void**)&b_down, N * size * sizeof(unsigned long long)));                
        // fusionModDown<<<dim,1024>>>(N,a_down,a,b_down,b,cipher.a,cipher.b,size,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        unsigned long long* a_down = Moddown_dcomp(a);
        unsigned long long* b_down = Moddown_dcomp(b);

        forwardNTT3(a_down,N,psiTable[0],1);
        inverseNTT(a_down,N,ntt,q[0],mu[0],q_bit[0],psiinvTable[0]);

        polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }

    cipherText relien_dcomp(triplePoly cipher,relienKey* key){
        unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
         Modup_dcomp_batch2_fusion(cipher.c,key[0].a,key[0].b,a,b,false);   
        // dim3 dimdouble(N/1024,size+1);
        // polymuldouble_sum_batch<<<dimdouble,1024>>>(d2s[0],key[0].a,a,d2s[0],key[0].b,b,N,size);

        // unsigned long long *a_down;
        // Check(mempool((void**)&a_down, N * size * sizeof(unsigned long long)));
        // unsigned long long *b_down;
        // Check(mempool((void**)&b_down, N * size * sizeof(unsigned long long)));                
        // fusionModDown<<<dim,1024>>>(N,a_down,a,b_down,b,cipher.a,cipher.b,size,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        unsigned long long* a_down = Moddown_dcomp(a);
        unsigned long long* b_down = Moddown_dcomp(b);


        polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }
    cipherText relien_dcomp(triplePoly cipher,relienKey* key,unsigned long long* buffer1,unsigned long long* buffer2,unsigned long long* buffer3,unsigned long long* a,unsigned long long* b){
        // unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
        // makezero<<<2 * N * size * size /1024,1024>>>(a);
        // makezero<<<2 * N * size * size /1024,1024>>>(b);

        unsigned long long** d2s = Modup_dcomp_batch2(cipher.c,buffer1,buffer2);   

        dim3 dimdouble(N/1024,size+1);
        polymuldouble_sum_batch<<<dimdouble,1024>>>(d2s[0],key[0].a,a,d2s[0],key[0].b,b,N,size);


        // unsigned long long *a_down;
        // Check(mempool((void**)&a_down, N * size * sizeof(unsigned long long)));
        // unsigned long long *b_down;
        // Check(mempool((void**)&b_down, N * size * sizeof(unsigned long long)));                
        // fusionModDown<<<dim,1024>>>(N,a_down,a,b_down,b,cipher.a,cipher.b,size,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        unsigned long long* a_down = Moddown_dcomp(a,buffer3);
        unsigned long long* b_down = Moddown_dcomp(b,buffer3);


        polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }

    cipherText relien_dcomp_fusion_buff(triplePoly cipher,relienKey* key,unsigned long long* buffer1,unsigned long long* buffer2,unsigned long long* buffer3,unsigned long long* a,unsigned long long* b){
        // unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
        // makezero<<<2 * N * size * size /1024,1024>>>(a);
        // makezero<<<2 * N * size * size /1024,1024>>>(b);

        Modup_dcomp_batch2_fusion_buff(buffer1,cipher.c,key[0].a,key[0].b,a,b,false);   

        // unsigned long long** d2s = Modup_dcomp_batch2(cipher.c,buffer1,buffer2);   

        // dim3 dimdouble(N/1024,size+1);
        // polymuldouble_sum_batch<<<dimdouble,1024>>>(d2s[0],key[0].a,a,d2s[0],key[0].b,b,N,size);


        // unsigned long long *a_down;
        // Check(mempool((void**)&a_down, N * size * sizeof(unsigned long long)));
        // unsigned long long *b_down;
        // Check(mempool((void**)&b_down, N * size * sizeof(unsigned long long)));                
        // fusionModDown<<<dim,1024>>>(N,a_down,a,b_down,b,cipher.a,cipher.b,size,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        unsigned long long* a_down = Moddown_dcomp(a,buffer3);
        unsigned long long* b_down = Moddown_dcomp(b,buffer3);


        polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }
    cipherText relien(triplePoly cipher,relienKey key,cudaStream_t stream){
        unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
        unsigned long long* d2 = Modup(cipher.c,stream);
        // for(int i = 0; i < 2 * size; i++){
        //     polymul<<<N/1024,1024>>>(d2 + N * i,key.a + N * i,a + N * i,q[i],mu[i],q_bit[i]);
        //     polymul<<<N/1024,1024>>>(d2 + N * i,key.b + N * i,b + N * i,q[i],mu[i],q_bit[i]);
        // }

        dim3 dimdouble(N/1024,2 * size);
        polymuldouble<<<dimdouble,1024,0,stream>>>(d2,key.a,a,d2,key.b,b);

        unsigned long long* a_down = Moddown(a,stream);
        unsigned long long* b_down = Moddown(b,stream);
        // for(int i = 0; i < size; i++){
        //     polyadd<<<N/1024,1024>>>(cipher.a + N * i,a_down + N * i,a_down + N * i, N,q[i]);
        //     polyadd<<<N/1024,1024>>>(cipher.b + N * i,b_down + N * i,b_down + N * i, N,q[i]);
        // }
        polyadddouble<<<dim,1024,0,stream>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);

        // print<<<1,1>>>(a_down,8);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }
    void rotation(unsigned long long *plain){

        unsigned long long *temp;
        Check(mempool((void**)&temp, N * size * sizeof(unsigned long long)));
        cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        for(int i = 0; i < size; i++){
            inverseNTT(plain + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        for(int i = 0; i < size; i++){
            cudarotation<<<N/1024,1024>>>(plain + N * i,temp + N * i,q[i],1 * 2 + 1,N);
        }
        cudaMemcpy(plain,temp,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        for(int i = 0; i < size; i++){
            forwardNTT(plain+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        // for(int i = 0; i < size; i++){
        //     polyadd<<<N/1024,1024>>>(plain + N * i,temp + N * i,plain + N * i, N,q[i]);
        // }
        // print<<<1,1>>>(plain);
    }
    void rotation(cipherText cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(step == 0){
            return ;
        }
        return rotation_comp(cipher,step,baby,gaint);
        
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,n-bstep,(2*N));
        auto key = baby[bstep];

        // print<<<1,1>>>(res_a);
        inverseNTT_batch(res_a, N, psiinvList, size, size);
        inverseNTT_batch(res_b, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        forwardNTT_batch(res_b_tmp, N, psiList, size, size);


        int gstep = step / baby_step;

        elt = modpow128(3,n-gstep*baby_step,(2*N));
        key = gaint[gstep];

        if(gstep != 0){        
            polymuladdsingle<<<dim,1024>>>(res_b_tmp,key.a,res_a_tmp,res_a_tmp);
            polymulsingle<<<dim,1024>>>(res_b_tmp,key.b,res_b_tmp);
        }else{
            unsigned long long **d2s = Modup_dcomp_batch2(res_a_tmp);
            Moddown(d2s[0]);
            for(int i = 0; i < size;i++)polymuladdsingle<<<dim,1024>>>(res_b_tmp,key.a,res_a_tmp,cipher.a);
            polymulsingle<<<dim,1024>>>(res_b_tmp,key.b,cipher.b);
            return ;            
        }
        


        // for(int i = dep; i < size; i++){
            // inverseNTT(res_a_tmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // inverseNTT(res_b_tmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        // }
        inverseNTT_batch(res_a_tmp, N, psiinvList, size, size);
        inverseNTT_batch(res_b_tmp, N, psiinvList, size, size);
        // for(int i = dep; i < size; i++){
            // cudarotation<<<N/1024,1024>>>(res_a_tmp + N * i,cipher.a + N * i,q[i],elt,N);
            // cudarotation<<<N/1024,1024>>>(res_b_tmp + N * i,cipher.b + N * i,q[i],elt,N);
        // }
        cudarotation_new<<<N/1024,1024>>>(res_a_tmp,cipher.a,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b_tmp,cipher.b,elt,N,size);
        // for(int i = dep; i < size; i++){
            // forwardNTT(cipher.a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // forwardNTT(cipher.b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        // forwardNTT3(cipher.a, N, cudaStream_t& stream1, unsigned long long q, unsigned long long mu, int bit_length, unsigned long long* psi_powers){
        forwardNTT_batch(cipher.a, N, psiList, size, size);

        forwardNTT_batch(cipher.b, N, psiList, size, size);
        // for(int i = dep; i < size; i++){
            // polymuladd<<<N/1024,1024>>>(cipher.b + N * i,key.a + N * i,cipher.a + N * i,cipher.a + N * i,q[i],mu[i],q_bit[i]);
            // polymul<<<N/1024,1024>>>(cipher.b + N * i,key.b + N * i,cipher.b + N * i,q[i],mu[i],q_bit[i]);
        // }

        polymuladdsingle<<<dim,1024>>>(cipher.b,key.a,cipher.a,cipher.a);
        polymulsingle<<<dim,1024>>>(cipher.b,key.b,cipher.b);
        // cudaFree(res_a_tmp);
        // cudaFree(res_b_tmp);
        // cudaFree(res_a);
        // cudaFree(res_b);
        // res.set(res_a,res_b);res.depth=cipher.depth;
        // return res;
    }
    void rotation_comp(cipherText& cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(step == 0){
            return ;
        }
        // cipher.a = baby[2*size].a;
        // cipher.b = baby[2*size].b;

        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, 2 *N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,n - bstep,(2*N));

        // set<<<8,1024>>>(res_a);
        // print<<<1,1>>>(res_a);

        // inverseNTT_batch(res_a, N, psiinvList, size, size);

        // print_dd<<<1,1>>>(res_a,8);

        // forwardNTT_batch(res_a, N, psiList, size, size);   

        inverseNTT_batch(res_a, N, psiinvList, size, size);
        inverseNTT_batch(res_b, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        forwardNTT_batch(res_b_tmp, N, psiList, size, size);


        unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);


        rotation_innerProd<<<dimdouble,1024>>>(d2s[0],key[bstep*size+0].a,tmp_a,d2s[0],key[bstep*size+0].b,tmp_b,N,size);


        unsigned long long* a_down = Moddown_dcomp(tmp_a);
        unsigned long long* b_down = Moddown_dcomp(tmp_b);

    
        // polymuladdsingle<<<dim,1024>>>(cipher.b,key.a,cipher.a,cipher.a);
 

        polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cipher.a);
        
        cudaMemcpy(cipher.b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        return ;            
    }

  void rotation_comp_table(cipherText& cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(step == 0){
            return ;
        }
        // cipher.a = baby[2*size].a;
        // cipher.b = baby[2*size].b;

        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, 2 *N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,n - bstep,(2*N));

        // set<<<8,1024>>>(res_a);
        // print<<<1,1>>>(res_a);

        // inverseNTT_batch(res_a, N, psiinvList, size, size);

        // print_dd<<<1,1>>>(res_a,8);

        // forwardNTT_batch(res_a, N, psiList, size, size);   

        // inverseNTT_batch(res_a, N, psiinvList, size, size);
        // inverseNTT_batch(res_b, N, psiinvList, size, size);

        // cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        // cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        // forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        // forwardNTT_batch(res_b_tmp, N, psiList, size, size);
        // cudaDeviceSynchronize();
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Error111: %sn", cudaGetErrorString(err));
        // }
        // print_dddd<<<1,1>>>(rottable[1]);
        cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottable[step],N,size);
        cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rottable[step],N,size);



        // print_dddd<<<1,1>>>(res_b_tmp);

        // inverseNTT_batch(res_b_tmp, N, psiinvList, size, size);
        // print_dddd<<<1,1>>>(res_b_tmp);
        // forwardNTT_batch(res_b_tmp, N, psiList, size, size);

        // cudarotation_new_table<<<N/1024,1024>>>(res_a,cipher.a,rottable[step],N,size);
        // cudarotation_new_table<<<N/1024,1024>>>(res_b,cipher.b,rottable[step],N,size);
        
        // print_dddd<<<1,1>>>(cipher.a);
        
        // return ;
        // // res_a_tmp = res_a;
        // // res_b_tmp = res_b;
        int depth = cipher.depth;
        auto key = baby;        
        Modup_dcomp_batch2_fusion(res_b_tmp,key[bstep*size+0].a,key[bstep*size+0].b,tmp_a,tmp_b,true); 
        // unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 
        // dim3 dimdouble(N/1024,size+1);


        // rotation_innerProd<<<dimdouble,1024>>>(d2s[0],key[bstep*size+0].a,tmp_a,d2s[0],key[bstep*size+0].b,tmp_b,N,size);

        // unsigned long long *a_down;
        // Check(mempool((void**)&a_down, N * size * sizeof(unsigned long long)));
        // unsigned long long *b_down;
        // Check(mempool((void**)&b_down, N * size * sizeof(unsigned long long)));                
        // fusionModDownROT<<<dim,1024>>>(N,cipher.a,tmp_a,cipher.b,tmp_b,res_a_tmp,b_down,size,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);


        unsigned long long* a_down = Moddown_dcomp(tmp_a);
        unsigned long long* b_down = Moddown_dcomp(tmp_b);
    
    
        // polymuladdsingle<<<dim,1024>>>(cipher.b,key.a,cipher.a,cipher.a);


        polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cipher.a);
        
        cudaMemcpy(cipher.b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        return ;            
    }




  void rotation_comp_table(cipherText& cipher,int step,galoisKey* galois){
        if(step == 0){
            return ;
        }


        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, 2 *N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        int bstep = log2(step);
        
        if((1 << bstep) != step){
            printf("NONONO11\n");
            throw "a";
        }





        // print_dddd<<<1,1>>>(res_a);
        // print_dddd<<<1,1>>>(res_a_tmp);

        cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottabler[step],N,size);
        cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rottabler[step],N,size);
        // cudarotation_new_table<<<N/1024,1024>>>(res_a,res_b,res_a_tmp,res_b_tmp,rottabler[step],N,size);

        // print_z<<<1,1>>>(res_a, res_a_tmp, N);

        int depth = cipher.depth;
        auto key = galois;  
              
        Modup_dcomp_batch2_fusion(res_b_tmp,key[bstep*size+0].a,key[bstep*size+0].b,tmp_a,tmp_b,true); 

        unsigned long long* a_down = Moddown_dcomp(tmp_a);
        unsigned long long* b_down = Moddown_dcomp(tmp_b);
    
    
        // polymuladdsingle<<<dim,1024>>>(cipher.b,key.a,cipher.a,cipher.a);


        polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cipher.a);
        
        cudaMemcpy(cipher.b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        // cudaMemcpy(cipher.b,res_b_tmp,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(cipher.a,res_a_tmp,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        return ;            
    }

    void rotation_comp_table(cipherText& cipher, cipherText& cipher2,int step,galoisKey* galois){
        if(step == 0){
            return ;
        }


        static unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        if(!res_a_tmp)Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        if(!res_b_tmp)Check(mempool((void**)&res_b_tmp, 2 *N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        if(!res_a)Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        if(!res_b)Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        if(!tmp_a)Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        if(!tmp_b)Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        int bstep = log2(step);
        
        if((1 << bstep) != step){
            printf("NONONO12\n");
            throw "a";
        }

        cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottabler[step],N,size);
        cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rottabler[step],N,size);

        int depth = cipher.depth;
        auto key = galois;  
              
        Modup_dcomp_batch2_fusion(res_b_tmp,key[bstep*size+0].a,key[bstep*size+0].b,tmp_a,tmp_b,true); 

        unsigned long long* a_down = Moddown_dcomp(tmp_a);
        unsigned long long* b_down = Moddown_dcomp(tmp_b);

        polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cipher2.a);
        
        cudaMemcpy(cipher2.b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        return ;            
    }

    void rotation_comp_tableinv(cipherText& cipher, cipherText& cipher2,int step,galoisKey* galois){
        if(step == 0){
            return ;
        }


        static unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        if(!res_a_tmp)Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        if(!res_b_tmp)Check(mempool((void**)&res_b_tmp, 2 *N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        if(!res_a)Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        if(!res_b)Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        if(!tmp_a)Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        if(!tmp_b)Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        int bstep = log2(step);
        
        if((1 << bstep) != step){
            printf("NONONO12\n");
            throw "a";
        }

        cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottable[step],N,size);
        cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rottable[step],N,size);

        int depth = cipher.depth;
        auto key = galois;  
              
        Modup_dcomp_batch2_fusion(res_b_tmp,key[bstep*size+0].a,key[bstep*size+0].b,tmp_a,tmp_b,true); 

        unsigned long long* a_down = Moddown_dcomp(tmp_a);
        unsigned long long* b_down = Moddown_dcomp(tmp_b);

        polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cipher2.a);
        
        cudaMemcpy(cipher2.b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        return ;            
    }
    // cipherText rotation_dcomp_fusion(cipherText cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
    //     unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
    //     cipherText res;res.depth = cipher.depth;
    //     int dep = cipher.depth;
    //     Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
    //     // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

    //     cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

    //     cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rot1r,N,size);
    //     cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rot1r,N,size);

    //     unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp);   

    //     dim3 dimdouble(N/1024,size+1);
    //     polymuldouble_sum_batch<<<dimdouble,1024>>>(d2s[0],baby[0].a,tmp_a,d2s[0],baby[0].b,tmp_b,N,size);
    //     unsigned long long* a_down = Moddown_dcomp(tmp_a);
    //     unsigned long long* b_down = Moddown_dcomp(tmp_b);
    //     polyaddsingle<<<dim,1024>>>(a_down,res_b,cipher.a);

    //     // polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);


    //     cudaMemcpy(cipher.b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);


    //     // unsigned long long *a_down;
    //     // Check(mempool((void**)&a_down, N * size * sizeof(unsigned long long)));
    //     // unsigned long long *b_down;
    //     // Check(mempool((void**)&b_down, N * size * sizeof(unsigned long long)));                
    //     // fusionModDown<<<dim,1024>>>(N,a_down,a,b_down,b,cipher.a,cipher.b,size,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
    //     // unsigned long long* a_down = a//Moddown_dcomp(a);
    //     // unsigned long long* b_down = b;//Moddown_dcomp(b);


    //     // polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);

    //     // for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(d2s[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);
    //     // unsigned long long* a_down = Moddown_dcomp(tmp);

    //     // cudaMemcpy(cipher.a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     // polymulsingle<<<dim,1024>>>(res_b_tmp,key[bstep*size].b,cipher.b);



    //     // cipherText res;
    //     // res.set(a_down,b_down);
    //     // res.depth = cipher.depth;
    //     return cipher;
    // }
    void rotation_comp_r(cipherText cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(step == 0){
            return ;
        }
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,bstep,(2*N));

        // print<<<1,1>>>(res_a);
        inverseNTT_batch(res_a, N, psiinvList, size, size);
        inverseNTT_batch(res_b, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        forwardNTT_batch(res_b_tmp, N, psiList, size, size);


        int gstep = step / baby_step;

        elt = modpow128(3,n-gstep*baby_step,(2*N));


        unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        // for(int i = 0; i < size;i++)polymuldouble<<<dimdouble,1024>>>(d2s[i],key[i].a,a,d2s[i],key[i].b,b);

        for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(d2s[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);
        unsigned long long* a_down = Moddown_dcomp(tmp);

        cudaMemcpy(cipher.a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        polymulsingle<<<dim,1024>>>(res_b_tmp,key[bstep*size].b,cipher.b);
        return ;            
    }
    void rotation_comp_r_T(cipherText cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(step == 0){
            return ;
        }
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step ;
        int elt = modpow128(3,bstep,(2*N));

        // print<<<1,1>>>(res_a);
        inverseNTT_batch(res_a, N, psiinvList, size, size);
        inverseNTT_batch(res_b, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        forwardNTT_batch(res_b_tmp, N, psiList, size, size);


        int gstep = step / baby_step;

        elt = modpow128(3,n-gstep*baby_step,(2*N));


        unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        // for(int i = 0; i < size;i++)polymuldouble<<<dimdouble,1024>>>(d2s[i],key[i].a,a,d2s[i],key[i].b,b);

        for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(d2s[i],key[size + i].a,res_a_tmp,tmp,q[size]);
        unsigned long long* a_down = Moddown_dcomp(tmp);

        cudaMemcpy(cipher.a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        polymulsingle<<<dim,1024>>>(res_b_tmp,key[size].b,cipher.b);
        return ;            
    }
    cipherText* rotation_comp_hoist_new(cipherText cipher,int step,int num,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,n-bstep,(2*N));

        // inverseNTT_batch(res_a, N, psiinvList, size, size);
        // inverseNTT_batch(res_b, N, psiinvList, size, size);

        // cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        // cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        // forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        // forwardNTT_batch(res_b_tmp, N, psiList, size, size);

        cipherText* cips = (cipherText*)malloc(num*sizeof(cipherText));

        cips[0].a = cipher.a;
        cips[0].b = cipher.b;

        if(num == 1)return cips;
        unsigned long long** d2s = Modup_dcomp_batch2(res_b); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        for(int idx = 1; idx < num; idx++){
            Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
            unsigned long long **res_b_tmp = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
            for(int i = 0; i < size; i++)Check(mempool((void**)&res_b_tmp[i], N * (1 + size) * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++)cudarotation_new_table<<<N/1024,1024>>>(d2s[i],res_b_tmp[i],rottable[step],N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottable[step],N,size);
            // for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(res_b_tmp[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);
            rotation_innerProd<<<dimdouble,1024>>>(res_b_tmp[0],key[idx*step*size+0].a,tmp_a,res_b_tmp[0],key[idx*step*size+0].b,tmp_b,N,size);

            unsigned long long* a_down = Moddown_dcomp(tmp_a);
            unsigned long long* b_down = Moddown_dcomp(tmp_b);

            Check(mempool((void**)&cips[idx].a, N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&cips[idx].b, N * size * sizeof(unsigned long long)));
            // cudaMemcpy(cips[idx].a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            // polymulsingle<<<dim,1024>>>(tmp,key[bstep*size].b,cips[idx].b);

            polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cips[idx].a);
            cudaMemcpy(cips[idx].b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);


            d2s = res_b_tmp;
            res_a = res_a_tmp;
            cips[idx].depth = cipher.depth;
        }

        return cips;            
    }
    cipherText* rotation_comp_hoist(cipherText cipher,int step,int num,galoisKeyRoot* baby,galoisKeyRoot* gaint){

       unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,n-bstep,(2*N));

        cipherText* cips = (cipherText*)malloc(num*sizeof(cipherText));

        cips[0].a = cipher.a;
        cips[0].b = cipher.b;

        unsigned long long** d2s = Modup_dcomp_batch2(res_b); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        for(int idx = 1; idx < num; idx++){
            Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
            unsigned long long **res_b_tmp = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
            for(int i = 0; i < size; i++)Check(mempool((void**)&res_b_tmp[i], N * (1 + size) * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++)cudarotation_new_table<<<N/1024,1024>>>(d2s[i],res_b_tmp[i],rot1,N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rot1,N,size);
            for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(res_b_tmp[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);

            unsigned long long* a_down = Moddown_dcomp(tmp);
            Check(mempool((void**)&cips[idx].a, N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&cips[idx].b, N * size * sizeof(unsigned long long)));
            cudaMemcpy(cips[idx].a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            polymulsingle<<<dim,1024>>>(tmp,key[bstep*size].b,cips[idx].b);
            d2s = res_b_tmp;
            res_a = res_a_tmp;
            cips[idx].depth = cipher.depth;
        }

        return cips;     
    }
    cipherText* rotation_comp_hoist_conv_baby(cipherText* cipher,int step,int num,galoisKeyRoot* baby,galoisKeyRoot* gaint){

        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp;
        cipherText res;res.depth = cipher[0].depth;
        int dep = cipher[0].depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher[0].a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher[0].b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        int bstep = step % baby_step;
        int elt = modpow128(3,n-bstep,(2*N));



        // cips[0].a = cipher.a;
        // cips[0].b = cipher.b;
        cipherText* cips = cipher;
        unsigned long long** d2s = Modup_dcomp_batch2(res_b); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        for(int idx = 1; idx < num; idx++){
            Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
            unsigned long long **res_b_tmp = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
            for(int i = 0; i < size; i++)Check(mempool((void**)&res_b_tmp[i], N * (1 + size) * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++)cudarotation_new_table<<<N/1024,1024>>>(d2s[i],res_b_tmp[i],rot1r,N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rot1r,N,size);
            for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(res_b_tmp[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);
            unsigned long long* a_down = Moddown_dcomp(tmp);

            cudaMemcpy(cips[idx].a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            polymulsingle<<<dim,1024>>>(tmp,key[bstep*size].b,cips[idx].b);
            d2s = res_b_tmp;
            res_a = res_a_tmp;
            cips[idx].depth = cips[0].depth;
        }

        return cips;            
    }
    cipherText* rotation_comp_hoist_conv_giant(cipherText* cipher,int step,int m,int num,galoisKeyRoot* baby,galoisKeyRoot* gaint){

        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp;
        cipherText res;res.depth = cipher[0].depth;
        int dep = cipher[0].depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher[0].a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher[0].b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        int bstep = step % baby_step;
        int elt = modpow128(3,n-bstep,(2*N));



        // cips[0].a = cipher.a;
        // cips[0].b = cipher.b;
        cipherText* cips = cipher;
        unsigned long long** d2s = Modup_dcomp_batch2(res_b); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        for(int idx = 1; idx < num; idx++){
            Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
            unsigned long long **res_b_tmp = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
            for(int i = 0; i < size; i++)Check(mempool((void**)&res_b_tmp[i], N * (1 + size) * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++)cudarotation_new_table<<<N/1024,1024>>>(d2s[i],res_b_tmp[i],rot2r,N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rot2r,N,size);
            for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(res_b_tmp[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);
            unsigned long long* a_down = Moddown_dcomp(tmp);

            cudaMemcpy(cips[idx * m].a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            polymulsingle<<<dim,1024>>>(tmp,key[bstep*size].b,cips[idx * m].b);
            d2s = res_b_tmp;
            res_a = res_a_tmp;
            cips[idx * m].depth = cips[0].depth;
        }

        return cips;            
    }

    cipherText* rotation_comp_hoist_conv_giant_new(cipherText* cipher,int step,int m,int num,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(num == 1){
            return cipher;
        }
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher[0].depth;
        int dep = cipher[0].depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher[0].a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher[0].b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);





        // cips[0].a = cipher.a;
        // cips[0].b = cipher.b;
        cipherText* cips = cipher;
        unsigned long long** d2s = Modup_dcomp_batch2(res_b); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        for(int idx = 1; idx < num; idx++){

            dim3 dimdouble(N/1024,size+1);

            Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
            unsigned long long **res_b_tmp = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
            for(int i = 0; i < size; i++)Check(mempool((void**)&res_b_tmp[i], N * (1 + size) * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++)cudarotation_new_table<<<N/1024,1024>>>(d2s[i],res_b_tmp[i],rottabler[step],N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottabler[step],N,size);
            rotation_innerProd<<<dimdouble,1024>>>(res_b_tmp[0],key[idx*step*size+0].a,tmp_a,res_b_tmp[0],key[idx*step*size+0].b,tmp_b,N,size);

            unsigned long long* a_down = Moddown_dcomp(tmp_a);
            unsigned long long* b_down = Moddown_dcomp(tmp_b);
            polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cips[idx * m].a);
            cudaMemcpy(cips[idx * m].b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

            d2s = res_b_tmp;
            res_a = res_a_tmp;
            cips[idx * m].depth = cips[0].depth;
        }
        return cips;            
    }


    cipherText* rotation_comp_hoist_conv_baby_new(cipherText* cipher,int step,int num,int stepbase,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(num == 1){
            return cipher;
        }
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;res.depth = cipher[0].depth;
        int dep = cipher[0].depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher[0].a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher[0].b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);




        // cips[0].a = cipher.a;
        // cips[0].b = cipher.b;
        cipherText* cips = cipher;
        unsigned long long** d2s = Modup_dcomp_batch2(res_b); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        for(int idx = 1; idx < num; idx++){

            dim3 dimdouble(N/1024,size+1);

            Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
            unsigned long long **res_b_tmp = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
            for(int i = 0; i < size; i++)Check(mempool((void**)&res_b_tmp[i], N * (1 + size) * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++)cudarotation_new_table<<<N/1024,1024>>>(d2s[i],res_b_tmp[i],rottabler[step],N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottabler[step],N,size);
            rotation_innerProd<<<dimdouble,1024>>>(res_b_tmp[0],key[idx *size+0].a,tmp_a,res_b_tmp[0],key[idx*size+0].b,tmp_b,N,size);

            unsigned long long* a_down = Moddown_dcomp(tmp_a);
            unsigned long long* b_down = Moddown_dcomp(tmp_b);
            polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,cips[idx].a);
            cudaMemcpy(cips[idx].b,b_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

            d2s = res_b_tmp;
            res_a = res_a_tmp;
            cips[idx].depth = cips[0].depth;
        }
        return cips;            
    }
    cipherText rotation_tt(cipherText cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        if(step == 0){
            return cipher;
        }
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int bstep = step % baby_step;
        int elt = modpow128(3,n-bstep,(2*N));

        // print<<<1,1>>>(res_a);
        inverseNTT_batch(res_a, N, psiinvList, size, size);
        inverseNTT_batch(res_b, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

        forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        forwardNTT_batch(res_b_tmp, N, psiList, size, size);


        int gstep = step / baby_step;

        elt = modpow128(3,n-gstep*baby_step,(2*N));


        unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 
        auto key = baby;        
        dim3 dimdouble(N/1024,size+1);
        // for(int i = 0; i < size;i++)polymuldouble<<<dimdouble,1024>>>(d2s[i],key[i].a,a,d2s[i],key[i].b,b);

        for(int i = 0; i < size;i++)polymuladdsingle<<<dimdouble,1024>>>(d2s[i],key[bstep*size+i].a,res_a_tmp,tmp,q[size]);
        unsigned long long* a_down = Moddown_dcomp(tmp);

        // cudaMemcpy(cipher.a,a_down,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        polymulsingle<<<dim,1024>>>(res_b_tmp,key[bstep*size].b,cipher.b);
        res.set(a_down,res_b_tmp);res.depth=cipher.depth;
        
        return res;            
    }

    cipherText rotation_t(cipherText cipher,int step,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        // if(step == 0){
        //     return ;
        // }
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
        cipherText res;res.depth = cipher.depth;
        int dep = cipher.depth;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        if(step == 0){
            res.set(res_a,res_b);res.depth=cipher.depth;
            return res; 
        }
        int bstep = step % baby_step;
        int elt = modpow128(3,bstep,(2*N));
        auto key = baby[bstep];

        inverseNTT_batch(res_a, N, psiinvList, size, size);
        inverseNTT_batch(res_b, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);


        forwardNTT_batch(res_a_tmp, N, psiList, size, size);
        forwardNTT_batch(res_b_tmp, N, psiList, size, size);


        int gstep = step / baby_step;

        elt = modpow128(3,gstep*baby_step,(2*N));
        key = gaint[gstep];

        // dim3 dim(N/1024, size);

        if(gstep != 0){        
            polymuladdsingle<<<dim,1024>>>(res_b_tmp,key.a,res_a_tmp,res_a_tmp);
            polymulsingle<<<dim,1024>>>(res_b_tmp,key.b,res_b_tmp);
        }else{
            polymuladdsingle<<<dim,1024>>>(res_b_tmp,key.a,res_a_tmp,res_a);
            polymulsingle<<<dim,1024>>>(res_b_tmp,key.b,res_b);
            res.set(res_a,res_b);res.depth=cipher.depth;
            return res;         
        }
        



        inverseNTT_batch(res_a_tmp, N, psiinvList, size, size);
        inverseNTT_batch(res_b_tmp, N, psiinvList, size, size);

        cudarotation_new<<<N/1024,1024>>>(res_a_tmp,res_a,elt,N,size);
        cudarotation_new<<<N/1024,1024>>>(res_b_tmp,res_b,elt,N,size);

        forwardNTT_batch(res_a, N, psiList, size, size);
        forwardNTT_batch(res_b, N, psiList, size, size);
        // for(int i = dep; i < size; i++){
            // polymuladd<<<N/1024,1024>>>(cipher.b + N * i,key.a + N * i,cipher.a + N * i,cipher.a + N * i,q[i],mu[i],q_bit[i]);
            // polymul<<<N/1024,1024>>>(cipher.b + N * i,key.b + N * i,cipher.b + N * i,q[i],mu[i],q_bit[i]);
        // }
        polymuladdsingle<<<dim,1024>>>(res_a,key.a,cipher.a,res_a);
        polymulsingle<<<dim,1024>>>(res_b,key.b,res_b);
        // cudaFree(res_a_tmp);
        // cudaFree(res_b_tmp);
        // cudaFree(res_a);
        // cudaFree(res_b);
        res.set(res_a,res_b);res.depth=cipher.depth;
        return res;
    }
    void sum(unsigned long long *plain){
        throw "not implement";
        unsigned long long *temp1,*temp2;
        Check(mempool((void**)&temp1, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&temp2, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        for(int i = 0; i < size; i++){
            inverseNTT(plain + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        for(int slot_idx = 0; slot_idx < N/2;slot_idx++){
            for(int i = 0; i < size; i++){
                cudarotation<<<N/1024,1024>>>(plain + N * i,plain + N * i,q[i],3,N);
            }
            for(int i = 0; i < size; i++){
                polyadd<<<N/1024,1024>>>(plain + N * i,temp2 + N * i,temp2 + N * i, N,q[i]);
            }
        }

        cudaMemcpy(plain,temp2,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        for(int i = 0; i < size; i++){
            forwardNTT(plain+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
    }
    cipherText sum(cipherText cipher, galoisKey* keylist){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
        cipherText res;res.depth = cipher.depth;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int elt = 3;
        for(int idx = 0; idx < log2(N) - 1;idx++){
            auto key = keylist[idx];
            for(int i = 0; i < size; i++){
                inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
                inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }
            for(int i = 0; i < size; i++){
                cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
                cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
            }

            for(int i = 0; i < size; i++){
                forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
                forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            for(int i = 0; i < size; i++){
                forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
                forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            for(int i = 0; i < size; i++){
                polymuladd<<<N/1024,1024>>>(res_b_tmp + N * i,key.a + N * i,res_a_tmp + N * i,res_a_tmp + N * i,q[i],mu[i],q_bit[i]);
                polymul<<<N/1024,1024>>>(res_b_tmp + N * i,key.b + N * i,res_b_tmp + N * i,q[i],mu[i],q_bit[i]);
            }
            for(int i = 0; i < size; i++){
                polyadd<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,res_a + N * i, N,q[i]);
                polyadd<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,res_b + N * i, N,q[i]);
            }
            elt = elt * elt % (2 * N);
        }
        res.set(res_a,res_b);res.depth=cipher.depth;
        return res;
    }

    cipherText sum_h(cipherText cipher, int step, galoisKey* keylist){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
        cipherText res;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int elt = modpow128(3,step,2*N);
        // printf("%lf\n",log2(N/step));
        
        for(int idx = log2(step); idx < log2(N) - 1;idx++){
            
            // for(int i = 0; i < size; i++){
            //     inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            //     inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // }
            inverseNTT_batch(res_a,N,psiinvList,size,size);
            inverseNTT_batch(res_b,N,psiinvList,size,size);

            // for(int i = 0; i < size; i++){
            //     cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
            //     cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
            // }
            cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
            cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

            // for(int i = 0; i < size; i++){
            //     forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            //     forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }

            forwardNTT_batch(res_a_tmp, N, psiList, size, size);
            forwardNTT_batch(res_b_tmp, N, psiList, size, size);

            // for(int i = 0; i < size; i++){
            //     forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            //     forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            forwardNTT_batch(res_a, N, psiList, size, size);
            forwardNTT_batch(res_b, N, psiList, size, size);

            unsigned long long *tmp;
            Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

            unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 
            for(int i = 0; i < size;i++)polymuladdsingle<<<dim,1024>>>(d2s[i],keylist[idx*size+i].a,res_a_tmp,tmp,q[size]);
            unsigned long long* a_down = Moddown_dcomp(tmp);
            polymulsingle<<<dim,1024>>>(res_b_tmp,keylist[idx*size].b,cipher.b);


            for(int i = 0; i < size; i++){
                polyadd<<<N/1024,1024>>>(res_a + N * i,a_down + N * i,res_a + N * i, N,q[i]);
                polyadd<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,res_b + N * i, N,q[i]);
            }
            elt = elt * elt % (2 * N);
        }
        res.set(res_a,res_b);res.depth=cipher.depth;
        return res;
    }
    cipherText sum_th(cipherText cipher, int step,int stepend, galoisKey* keylist){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b, 2 * N * size * sizeof(unsigned long long)));
        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int elt = modpow128(3,step,2*N);
        // printf("%lf\n",log2(N/step));
        int ttstep = step;
        for(int idx = log2(step); idx < log2(stepend);idx++){
            // printf("%d\n",idx);
            auto key = keylist[idx];
            // for(int i = 0; i < size; i++){
            //     inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            //     inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // }
            // inverseNTT_batch(res_a,N,psiinvList,size,size);
            // inverseNTT_batch(res_b,N,psiinvList,size,size);

            // for(int i = 0; i < size; i++){
            //     cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
            //     cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
            // }
            // printf("%p\n",rottable[ttstep]);
            // cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
            // cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottabler[ttstep],N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rottabler[ttstep],N,size);
            ttstep*=2;

            // for(int i = 0; i < size; i++){
            //     forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            //     forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }

            // forwardNTT_batch(res_a_tmp, N, psiList, size, size);
            // forwardNTT_batch(res_b_tmp, N, psiList, size, size);

            // for(int i = 0; i < size; i++){
            //     forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            //     forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            // forwardNTT_batch(res_a, N, psiList, size, size);
            // forwardNTT_batch(res_b, N, psiList, size, size);

            unsigned long long *tmp;
            Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

            // unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 

            // dim3 dimdouble(N/1024,size+1);

            // rotation_innerProd<<<dimdouble,1024>>>(d2s[0],keylist[idx*size+0].a,tmp_a,d2s[0],keylist[idx*size+0].b,tmp_b,N,size);
            // unsigned long long* a_down = Moddown_dcomp(tmp_a);
            // unsigned long long* b_down = Moddown_dcomp(tmp_b);



            int depth = cipher.depth;
    
            Modup_dcomp_batch2_fusion(res_b_tmp,keylist[idx*size+0].a,keylist[idx*size+0].b,tmp_a,tmp_b,true); 



            unsigned long long* a_down = Moddown_dcomp(tmp_a);
            unsigned long long* b_down = Moddown_dcomp(tmp_b);


            polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,a_down);
            

            polyaddsingle<<<dim,1024>>>(res_a ,a_down ,res_a);
            polyaddsingle<<<dim,1024>>>(res_b ,b_down,res_b);

            elt = elt * elt % (2 * N);
        }
        res.set(res_a,res_b);res.depth=cipher.depth;
        return res;
    }

//

       
//
    cipherText sum_th_r(cipherText cipher, int step,int stepend, galoisKey* keylist){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp,*tmp_a,*tmp_b;
        cipherText res;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&tmp_b, 2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int elt = modpow128(3,2*N - step,2*N);
        // printf("%lf\n",log2(N/step));
        int ttstep = step;
        for(int idx = log2(step); idx < log2(stepend);idx++){
            
            auto key = keylist[idx*size];
            // for(int i = 0; i < size; i++){
            //     inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            //     inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            // }
            // inverseNTT_batch(res_a,N,psiinvList,size,size);
            // inverseNTT_batch(res_b,N,psiinvList,size,size);

            // for(int i = 0; i < size; i++){
            //     cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
            //     cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
            // }
            // cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
            // cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

            cudarotation_new_table<<<N/1024,1024>>>(res_a,res_a_tmp,rottable[ttstep],N,size);
            cudarotation_new_table<<<N/1024,1024>>>(res_b,res_b_tmp,rottable[ttstep],N,size);
            ttstep*=2;

            // for(int i = 0; i < size; i++){
            //     forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            //     forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }

            // forwardNTT_batch(res_a_tmp, N, psiList, size, size);
            // forwardNTT_batch(res_b_tmp, N, psiList, size, size);

            // for(int i = 0; i < size; i++){
            //     forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            //     forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            // }
            // forwardNTT_batch(res_a, N, psiList, size, size);
            // forwardNTT_batch(res_b, N, psiList, size, size);

            // __global__ void polymuladd_new(unsigned long long a[], unsigned long long b[], unsigned long long c[],unsigned long long d[],int batchSize)

            unsigned long long *tmp;
            Check(mempool((void**)&tmp,2 * N * size * sizeof(unsigned long long)));

            unsigned long long** d2s = Modup_dcomp_batch2(res_b_tmp); 

            dim3 dimdouble(N/1024,size+1);

            rotation_innerProd<<<dimdouble,1024>>>(d2s[0],keylist[idx*size+0].a,tmp_a,d2s[0],keylist[idx*size+0].b,tmp_b,N,size);
            unsigned long long* a_down = Moddown_dcomp(tmp_a);
            unsigned long long* b_down = Moddown_dcomp(tmp_b);

            polyaddsingle<<<dim,1024>>>(a_down,res_a_tmp,a_down);
            

            polyaddsingle<<<dim,1024>>>(res_a ,a_down ,res_a);
            polyaddsingle<<<dim,1024>>>(res_b ,b_down,res_b);

            elt = elt * elt % (2 * N);
        }
        res.set(res_a,res_b);res.depth=cipher.depth;
        return res;
    }
    // cipherText sum_th_r(cipherText cipher, int step,int stepend, galoisKey* keylist){
    //     unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
    //     cipherText res;
    //     Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
    //     // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

    //     cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     int elt = modpow128(3,2*N - step,2*N);
    //     // printf("%lf\n",log2(N/step));
        
    //     for(int idx = log2(step); idx < log2(stepend);idx++){
            
    //         auto key = keylist[idx];
    //         // for(int i = 0; i < size; i++){
    //         //     inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         //     inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         // }
    //         inverseNTT_batch(res_a,N,psiinvList,size,size);
    //         inverseNTT_batch(res_b,N,psiinvList,size,size);

    //         // for(int i = 0; i < size; i++){
    //         //     cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
    //         //     cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
    //         // }
    //         cudarotation_new<<<N/1024,1024>>>(res_a,res_a_tmp,elt,N,size);
    //         cudarotation_new<<<N/1024,1024>>>(res_b,res_b_tmp,elt,N,size);

    //         // for(int i = 0; i < size; i++){
    //         //     forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         //     forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         // }

    //         forwardNTT_batch(res_a_tmp, N, psiList, size, size);
    //         forwardNTT_batch(res_b_tmp, N, psiList, size, size);

    //         // for(int i = 0; i < size; i++){
    //         //     forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         //     forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         // }
    //         forwardNTT_batch(res_a, N, psiList, size, size);
    //         forwardNTT_batch(res_b, N, psiList, size, size);
    //         for(int i = 0; i < size; i++){
    //             polymuladd<<<N/1024,1024>>>(res_b_tmp + N * i,key.a + N * i,res_a_tmp + N * i,res_a_tmp + N * i,q[i],mu[i],q_bit[i]);
    //             polymul<<<N/1024,1024>>>(res_b_tmp + N * i,key.b + N * i,res_b_tmp + N * i,q[i],mu[i],q_bit[i]);
    //         }
    //         for(int i = 0; i < size; i++){
    //             polyadd<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,res_a + N * i, N,q[i]);
    //             polyadd<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,res_b + N * i, N,q[i]);
    //         }
    //         elt = elt * elt % (2 * N);
    //     }
    //     res.set(res_a,res_b);res.depth=cipher.depth;
    //     return res;
    // }
    cipherText sum_t(cipherText cipher, int step, galoisKey* keylist){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
        cipherText res;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int elt = 3;
        // printf("%lf\n",log2(N/step));
        
        for(int idx = 0; idx < log2(step);idx++){
            
            auto key = keylist[idx];
            for(int i = 0; i < size; i++){
                inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
                inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }
            for(int i = 0; i < size; i++){
                cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
                cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
            }

            for(int i = 0; i < size; i++){
                forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
                forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            for(int i = 0; i < size; i++){
                forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
                forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }

            Modup_dcomp_batch2(res_a_tmp);
            for(int i = 0; i < size; i++){
                polymuladd<<<N/1024,1024>>>(res_b_tmp + N * i,key.a + N * i,res_a_tmp + N * i,res_a_tmp + N * i,q[i],mu[i],q_bit[i]);
                polymul<<<N/1024,1024>>>(res_b_tmp + N * i,key.b + N * i,res_b_tmp + N * i,q[i],mu[i],q_bit[i]);
            }
            for(int i = 0; i < size; i++){
                polyadd<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,res_a + N * i, N,q[i]);
                polyadd<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,res_b + N * i, N,q[i]);
            }
            elt = elt * elt % (2 * N);
        }
        res.set(res_a,res_b);
        return res;
    }
    cipherText sum_r(cipherText cipher, int step, galoisKey* keylist){
        unsigned long long *res_a,*res_b,*res_a_tmp,*res_b_tmp;
        cipherText res;
        Check(mempool((void**)&res_a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b_tmp, N * size * sizeof(unsigned long long)));
        // cudaMemcpy(temp,plain,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));

        cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_a_tmp,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(res_b_tmp,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        int elt = modpow128(3,n-1,(2*N));
        // printf("%lf\n",log2(N/step));
        
        for(int idx = 0; idx < log2(step);idx++){
            
            auto key = keylist[idx];
            for(int i = 0; i < size; i++){
                inverseNTT(res_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
                inverseNTT(res_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }
            for(int i = 0; i < size; i++){
                cudarotation<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,q[i],elt,N);
                cudarotation<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,q[i],elt,N);
            }

            for(int i = 0; i < size; i++){
                forwardNTT(res_a_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
                forwardNTT(res_b_tmp+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            for(int i = 0; i < size; i++){
                forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
                forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            Modup_dcomp_batch2(res_a_tmp);

            for(int i = 0; i < size; i++){
                polymuladd<<<N/1024,1024>>>(res_b_tmp + N * i,key.a + N * i,res_a_tmp + N * i,res_a_tmp + N * i,q[i],mu[i],q_bit[i]);
                polymul<<<N/1024,1024>>>(res_b_tmp + N * i,key.b + N * i,res_b_tmp + N * i,q[i],mu[i],q_bit[i]);
            }
            for(int i = 0; i < size; i++){
                polyadd<<<N/1024,1024>>>(res_a + N * i,res_a_tmp + N * i,res_a + N * i, N,q[i]);
                polyadd<<<N/1024,1024>>>(res_b + N * i,res_b_tmp + N * i,res_b + N * i, N,q[i]);
            }
            elt = elt * elt % (2 * N);
        }
        res.set(res_a,res_b);
        res.depth = cipher.depth;
        return res;
    }

    cipherText innerProduct(cipherText& cipher1,cipherText& cipher2 ,galoisKey* galois,relienKey relien_key){
        auto mul_res = mulcipter(cipher1,cipher2);
        auto relien_res = relien(mul_res,relien_key);

        cipherText res = sum(relien_res,galois);
        return res;
    }

    cipherText innerProduct(cipherText& cipher1,unsigned long long *plain ,galoisKey* galois){
        mulPlain(cipher1,plain);
        // return cipher1;
        // auto relien_res = relien(mul_res,relien_key);

        cipherText res = sum(cipher1,galois);
        return res;
    }
    // cipherText dot(Matrix matrix, cipherText vec){
    //     unsigned long long *res_a,*res_b;
    //     Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));        
    //     for(int idx = 0; idx < matrix.M; idx++){
    //         for(int i = 0; i < size; i++){
    //             polymuladd<<<N/1024,1024>>>(matrix.data[idx] + N * i,vec.a + N * i,res_a + N * i,res_a + N * i,q[i],mu[i],q_bit[i]);
    //             polymuladd<<<N/1024,1024>>>(matrix.data[idx] + N * i,vec.b + N * i,res_b + N * i,res_b + N * i,q[i],mu[i],q_bit[i]);
    //         }
    //     }
    //     cipherText res;
    //     res.set(res_a,res_b);
    //     return res;
    // }
    // cipherText replication(cipherText vec){
        
    // }
    cipherText dot(Matrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right){
        // unsigned long long *res_a,*res_b;
        // Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        // Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        auto pad = sum_th_r(vec,matrix.N,1024,galois_right);
        // return pad;
        // for(int i = 0; i < size; i++){
        //     polymul<<<N/1024,1024>>>(matrix.data + N * i,pad.a + N * i,pad.a + N * i,q[i],mu[i],q_bit[i]);
        //     polymul<<<N/1024,1024>>>(matrix.data + N * i,pad.b + N * i,pad.b + N * i,q[i],mu[i],q_bit[i]);
        // }  
        polymuldouble<<<dim,1024>>>(matrix.data,pad.a,pad.a,matrix.data,pad.b,pad.b);
        pad.depth = vec.depth;
        // rescale(pad);
        // return pad;
        // return pad;
        auto res = sum_th(pad,1,matrix.N,galois);
        res.depth = vec.depth;
        // rescale(res);
        
        return res;
        // double mask[n];
        // for(int i = 0; i < n;i++){
        //     mask[i] = 0;
        // }
        // for(int i = 0; i < matrix.M;i++){
        //     mask[i*matrix.N] = 1;
        // }
        // auto maskEncode = encoder.encode(mask);
        // mulPlain(res,maskEncode);
        // // return res;
        // res = sum_r(res,matrix.N,galois_right);
        // return res;
    }
    cipherText dot(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        int row = matrix.M;
        unsigned long long *a_tmp,*b_tmp,*a_res,*b_res;
        Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_res, N * size * sizeof(unsigned long long)));
        cipherText res;res.set(a_res,b_res);
        for(int idx = 0; idx < row;idx++){
            cudaMemcpy(a_tmp,vec.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            cudaMemcpy(b_tmp,vec.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            cipherText tmp;tmp.set(a_tmp,b_tmp);tmp.depth = vec.depth;

            rotation(tmp,idx,baby,gaint);
            
            mulPlain(tmp,matrix.data[idx]);

            addcipter(res,tmp);

        }
        res = sum_h(res,row,galois);
        // double mask[n];
        // for(int i = 0;i<n;i++)mask[i]=0;
        // for(int i = 0;i<row;i++)mask[i]=1;
        // auto encodeMask = encoder.encode(mask);
        // mulPlain(res,encodeMask);
        return res;
    }
    // cipherText dot(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right){
        
    //     auto pad = sum_h(vec,matrix.N,galois);
    //     for(int i = 0; i < size; i++){
    //         polymul<<<N/1024,1024>>>(matrix.data + N * i,pad.a + N * i,pad.a + N * i,q[i],mu[i],q_bit[i]);
    //         polymul<<<N/1024,1024>>>(matrix.data + N * i,pad.b + N * i,pad.b + N * i,q[i],mu[i],q_bit[i]);
    //     }
                
    // }


    cipherText dot_bsgs(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint, unsigned long long* encodeMask){
        int row = matrix.M;
        unsigned long long *a_tmp,*b_tmp,*a_res,*b_res,*a_bsgs_tmp,*b_bsgs_tmp;
        unsigned long long **bsgstmp;

        // cipherText bsgs_vec[bsgs];

        Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_bsgs_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_bsgs_tmp, N * size * sizeof(unsigned long long)));
        cipherText res;res.set(a_res,b_res);
        cipherText cip_tmp;cip_tmp.set(a_tmp,b_tmp);
        cipherText cip_sum;cip_sum.set(a_bsgs_tmp,b_bsgs_tmp);
        unsigned long long *a_temp[bsgs];
        unsigned long long *b_temp[bsgs];
        // cipherText bsgs_vec[bsgs];
        // for(int i = 0; i < bsgs; i++){
        //     Check(mempool((void**)&a_temp[i], N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&b_temp[i], N * size * sizeof(unsigned long long)));
        //     bsgs_vec[i].set(a_temp[i],b_temp[i]);
        //     bsgs_vec[i].depth = vec.depth;
        // }
        // auto pad = sum_th_r(vec,1024,4096,galois_right);

        auto bsgs_vec = rotation_comp_hoist_new(vec,1,bsgs,baby,gaint);
        // for(int i = 0; i < bsgs; i++){
        //     // Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
        //     // Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
        //     cudaMemcpy(a_temp[i],vec.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     cudaMemcpy(b_temp[i],vec.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     // bsgs_vec[i].set(a_tmp,b_tmp);bsgs_vec[i].depth = vec.depth;
        //     rotation(bsgs_vec[i],i,baby,gaint);
        // }
        // return bsgs_vec[3];
        // bsgs_vec[2].depth = vec.depth;
        // return bsgs_vec[2];


        for(int i = row/bsgs-1; i >= 0; i--){
            cipherClear(cip_sum);
            for(int j = 0; j < bsgs; j++){
                // mulPlainAddCipher(bsgs_vec[j],matrix.data[i*bsgs+j],cip_tmp);
                mulPlain_new1_lazy(cip_tmp,bsgs_vec[j],matrix.data[i*bsgs+j]);
                // if(i ==3 && j == 7)return cip_tmp;
                addcipter(cip_sum,cip_tmp);
                // muladdPlain(cip_sum,cip_tmp)
            }
            rotation_comp_table(res,bsgs,baby,gaint);

            // rescale(cip_sum);
            addcipter(res,cip_sum);
            // if(i == 7){ rotation_comp_table(res,bsgs,baby,gaint);return res;} 
        }
        res.depth = vec.depth;

        rescale(res);


        res = sum_th(res,row,1024,galois);
        // return res;
        // return res;
        // mulPlain(res,encodeMask);
        // rescale(res);


        // res.depth = vec.depth+2;
        return res;
    }

    cipherText dot_bsgs_test(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint,  unsigned long long* encodeMask){
        int row = matrix.M;
        unsigned long long *a_tmp,*b_tmp,*a_res,*b_res,*a_bsgs_tmp,*b_bsgs_tmp;
        unsigned long long **bsgstmp;

        int bsgst = 4;
        Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_bsgs_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_bsgs_tmp, N * size * sizeof(unsigned long long)));
        cipherText res;res.set(a_res,b_res);
        cipherText cip_tmp;cip_tmp.set(a_tmp,b_tmp);
        cipherText cip_sum;cip_sum.set(a_bsgs_tmp,b_bsgs_tmp);

        auto bsgs_vec = rotation_comp_hoist_new(vec,1,bsgst,baby,gaint);

        for(int i = row/bsgst-1; i >= 0; i--){
            cipherClear(cip_sum);
            for(int j = 0; j < bsgst; j++){
                // mulPlainAddCipher(bsgs_vec[j],matrix.data[i*bsgs+j],cip_tmp);
                mulPlain_new1_lazy(cip_tmp,bsgs_vec[j],matrix.data[i*bsgst+j]);
                // if(i ==3 && j == 7)return cip_tmp;
                addcipter(cip_sum,cip_tmp);
                // muladdPlain(cip_sum,cip_tmp)
            }
            rotation_comp_table(res,bsgst,baby,gaint);
            addcipter(res,cip_sum);
        }
        res.depth = vec.depth ;
        rescale(res);

        res = sum_th(res,matrix.M,matrix.N,galois);
        return res;
    }

    cipherText dot_bsgs_test_test(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint, unsigned long long* encodeMask){
        int row = matrix.M;
        unsigned long long *a_tmp,*b_tmp,*a_res,*b_res,*a_bsgs_tmp,*b_bsgs_tmp;
        unsigned long long **bsgstmp;

        // cipherText bsgs_vec[bsgs];
        int bsgst = 4;
        Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_res, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&a_bsgs_tmp, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b_bsgs_tmp, N * size * sizeof(unsigned long long)));
        cipherText res;res.set(a_res,b_res);
        cipherText cip_tmp;cip_tmp.set(a_tmp,b_tmp);
        cipherText cip_sum;cip_sum.set(a_bsgs_tmp,b_bsgs_tmp);

        // cipherText bsgs_vec[bsgs];
        // for(int i = 0; i < bsgs; i++){
        //     Check(mempool((void**)&a_temp[i], N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&b_temp[i], N * size * sizeof(unsigned long long)));
        //     bsgs_vec[i].set(a_temp[i],b_temp[i]);
        //     bsgs_vec[i].depth = vec.depth;
        // }
        // auto pad = sum_th_r(vec,matrix.N,1024,galois_right);

        auto bsgs_vec = rotation_comp_hoist_new(vec,1,bsgst,baby,gaint);
        // for(int i = 0; i < bsgs; i++){
        //     // Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
        //     // Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
        //     cudaMemcpy(a_temp[i],vec.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     cudaMemcpy(b_temp[i],vec.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     // bsgs_vec[i].set(a_tmp,b_tmp);bsgs_vec[i].depth = vec.depth;
        //     rotation(bsgs_vec[i],i,baby,gaint);
        // }
        // return bsgs_vec[3];
        // bsgs_vec[2].depth = vec.depth;
        // return bsgs_vec[3];
        // return bsgs_vec[7];

        for(int i = row/bsgst-1; i >= 0; i--){
            cipherClear(cip_sum);
            for(int j = 0; j < bsgst; j++){
                // mulPlainAddCipher(bsgs_vec[j],matrix.data[i*bsgs+j],cip_tmp);
                mulPlain_new1_lazy(cip_tmp,bsgs_vec[j],matrix.data[i*bsgst+j]);
                // if(i ==3 && j == 7)return cip_tmp;
                addcipter(cip_sum,cip_tmp);
                // muladdPlain(cip_sum,cip_tmp)
            }
            rotation_comp_table(res,bsgst,baby,gaint);
            addcipter(res,cip_sum);
        }
        res.depth = vec.depth ;
        rescale(res);
        // return res;
        // res.depth = vec.depth+2;
        // return res;
        // return res;
        res = sum_th(res,16,128,galois);
        return res;

        // mulPlain(res,encodeMask);
        
        return res;
    }
    // cipherText dot_stream(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint){
    //     int row = matrix.M;
    //     unsigned long long *a_tmp,*b_tmp,*a_res,*b_res,*a_bsgs_tmp,*b_bsgs_tmp;
    //     unsigned long long **bsgstmp;

    //     // cipherText bsgs_vec[bsgs];

    //     Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&a_res, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b_res, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&a_bsgs_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b_bsgs_tmp, N * size * sizeof(unsigned long long)));
    //     cipherText res;res.set(a_res,b_res);
    //     cipherText cip_tmp;cip_tmp.set(a_tmp,b_tmp);
    //     cipherText cip_sum;cip_sum.set(a_bsgs_tmp,b_bsgs_tmp);
    //     unsigned long long *a_temp[bsgs];
    //     unsigned long long *b_temp[bsgs];
    //     cipherText bsgs_vec[bsgs];
    //     for(int i = 0; i < bsgs; i++){
    //         Check(mempool((void**)&a_temp[i], N * size * sizeof(unsigned long long)));
    //         Check(mempool((void**)&b_temp[i], N * size * sizeof(unsigned long long)));
    //         bsgs_vec[i].set(a_temp[i],b_temp[i]);
    //         bsgs_vec[i].depth = vec.depth;
    //     }
    //     for(int i = 0; i < bsgs; i++){
    //         // Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
    //         // Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
    //         cudaMemcpy(a_temp[i],vec.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         cudaMemcpy(b_temp[i],vec.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         // bsgs_vec[i].set(a_tmp,b_tmp);bsgs_vec[i].depth = vec.depth;
    //         rotation(bsgs_vec[i],i,baby,gaint);
    //     }


    //     const int nStreams = bsgs;
    //     cudaStream_t streams[nStreams];
    //     for (int i = 0; i < nStreams; i++) {
    //         cudaStreamCreate(&streams[i]);
    //     }
    //     for(int i = 0; i < row/bsgs; i++){
    //         cipherClear(cip_sum);
    //         for(int j = 0; j < bsgs; j++){
    //             mulPlain_stream(cip_tmp,bsgs_vec[j],matrix.data[i*bsgs+j],streams[j]);
    //             addcipter_stream(cip_sum,cip_tmp,streams[j]);
    //         }
    //         rotation(cip_sum,i * bsgs,baby,gaint);
    //         addcipter(res,cip_sum);
    //     }

    //     res = sum_h(res,row,galois);
    //     // double mask[n];
    //     // for(int i = 0;i<n;i++)mask[i]=0;
    //     // for(int i = 0;i<row;i++)mask[i]=1;
    //     // auto encodeMask = encoder.encode(mask);
    //     // mulPlain(res,encodeMask);
    //     return res;
    // }

    // cipherText dot_bsgs1(BigMatrix matrix,cipherText vec,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint){
    //     int row = matrix.M;
    //     unsigned long long *a_tmp,*b_tmp,*a_res,*b_res,*a_bsgs_tmp,*b_bsgs_tmp;
    //     unsigned long long **bsgstmp;

    //     // cipherText bsgs_vec[bsgs];

    //     Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&a_res, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b_res, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&a_bsgs_tmp, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&b_bsgs_tmp, N * size * sizeof(unsigned long long)));
    //     cipherText res;res.set(a_res,b_res);
    //     cipherText cip_tmp;cip_tmp.set(a_tmp,b_tmp);
    //     cipherText cip_sum;cip_sum.set(a_bsgs_tmp,b_bsgs_tmp);
    //     unsigned long long *a_temp[bsgs];
    //     unsigned long long *b_temp[bsgs];
    //     cipherText bsgs_vec[bsgs];
    //     for(int i = 0; i < bsgs; i++){
    //         Check(mempool((void**)&a_temp[i], N * size * sizeof(unsigned long long)));
    //         Check(mempool((void**)&b_temp[i], N * size * sizeof(unsigned long long)));
    //         bsgs_vec[i].set(a_temp[i],b_temp[i]);
    //         bsgs_vec[i].depth = vec.depth;
    //     }
    //     for(int i = 0; i < bsgs; i++){
    //         // Check(mempool((void**)&a_tmp, N * size * sizeof(unsigned long long)));
    //         // Check(mempool((void**)&b_tmp, N * size * sizeof(unsigned long long)));
    //         cudaMemcpy(a_temp[i],vec.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         cudaMemcpy(b_temp[i],vec.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         // bsgs_vec[i].set(a_tmp,b_tmp);bsgs_vec[i].depth = vec.depth;
    //         rotation(bsgs_vec[i],i,baby,gaint);
    //     }



    //     for(int i = 0; i < row/bsgs/2; i++){
    //         cipherClear(cip_sum);
    //         for(int j = 0; j < bsgs; j++){
    //             // mulPlainAddCipher(bsgs_vec[j],matrix.data[i*bsgs+j],cip_tmp);
    //             mulPlain_new1(cip_tmp,bsgs_vec[j],matrix.data[i*bsgs+j]);
    //             // if(i ==3 && j == 7)return cip_tmp;
    //             addcipter(cip_sum,cip_tmp);
    //             // muladdPlain(cip_sum,cip_tmp)
    //         }
    //         rotation(cip_sum,i * bsgs,baby,gaint);
    //         addcipter(res,cip_sum);
    //     }

    //     res = sum_h(res,row,galois);
    //     double mask[n];
    //     for(int i = 0;i<n;i++)mask[i]=0;
    //     for(int i = 0;i<row;i++)mask[i]=1;
    //     auto encodeMask = encoder.encode(mask);
    //     mulPlain(res,encodeMask);
    //     return res;
    // }
    //TODO
    //inv
    cipherText conv(cipherText& cipher, ConvKer& convker,int w,int h,int m,int n,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        int len = convker.len;
        // printf("%d,%d,%d\n",m,n,len);

        unsigned long long** kerData = convker.data;

        cipherText* cip = (cipherText*)malloc(len * sizeof(cipherText));


        for(int i = 0; i < len; i++){
            unsigned long long *a_temp,*b_temp;
            Check(mempool((void**)&a_temp, N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&b_temp, N * size * sizeof(unsigned long long)));
            cip[i].set(a_temp,b_temp);
        }
        // for(int i = 0; i < m; i++){
        cudaMemcpy(cip[0].a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        cudaMemcpy(cip[0].b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            // rotation_comp_r(cip[i*n],i*w,baby,gaint);
        // }
        rotation_comp_hoist_conv_giant_new(cip,h,m,m,baby,gaint);

        for(int i = 0; i < m; i++){
            // for(int j = 0; j < n; j++){
                // cudaMemcpy(cip[i*n+j].a,cip[i*n].a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
                // cudaMemcpy(cip[i*n+j].b,cip[i*n].b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
                rotation_comp_hoist_conv_baby_new(&cip[i*n],1,n,0,baby,gaint);                
            // }
                // cip[1].depth = cipher.depth;
                // return cip[1];
        }
        // return cip[1];
        unsigned long long *res_a,*res_b;
        Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
        for(int i = 0; i < len;i++){
            // for(int j = 0; j < size; j++){
            //     polymuladd<<<N/1024,1024>>>(cip[i].a + N * j,kerData[i] + N * j,res_a + N * j,res_a + N * j,q[j],mu[j],q_bit[j]);
            //     polymuladd<<<N/1024,1024>>>(cip[i].b + N * j,kerData[i] + N * j,res_b + N * j,res_b + N * j,q[j],mu[j],q_bit[j]);
            // }
            polymuladd_new<<<dim,1024>>>(cip[i].a,kerData[i],res_a,res_a);
            polymuladd_new<<<dim,1024>>>(cip[i].b,kerData[i],res_b,res_b);
            // printf("%p\n",cip[i].a);
        }
        cipherText res;
        res.set(res_a,res_b);
        res.depth = cipher.depth;
        // double mask[n];
        // for(int i = 0;i<n;i++)mask[i]=0;
        // auto encodeMask = encoder.encode(mask);
        // mulPlain(res,encodeMask);
        return res;
    }
    unsigned long long* Modup(unsigned long long* cipher){
        unsigned long long *res;
        Check(mempool((void**)&res, N * 2 * size * sizeof(unsigned long long)));


        inverseNTT_batch(cipher, N, psiinvList, size, size);

        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        
        cudaConvUp<<<N/1024,1024>>>(N,res,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);

        forwardNTT_batch(res, N, psiList, 2 * size, 2 * size);
        forwardNTT_batch(cipher, N, psiList, size, size);

        return res;
    }

    unsigned long long** Modup_dcomp(unsigned long long* cipher){
        unsigned long long *res,*rest;
        unsigned long long **ress = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
        

        Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(res, N, psiinvList, size, size);
        // cudaStream_t stream[stream_num];
        // for(int i = 0; i < stream_num; i++){
        //     cudaStreamCreate(&stream[i]);
        // }

        for(int i = 0; i < size;i++){
            Check(mempool((void**)&rest, N * (size + size) * sizeof(unsigned long long)));
            cudaMemcpy(rest,res,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            cudaConvUp_dcomp<<<N/1024,1024>>>(N,rest,i,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
            forwardNTT_batch(rest, N, psiList, size + 1, size + 1);
            ress[i] = rest;
        }
        // for(int i = 0; i < stream_num; i++){
        //     cudaStreamSynchronize(stream[i]); 
        //     cudaStreamDestroy(stream[i]);
        // }
        forwardNTT_batch(cipher, N, psiList, size, size);


        return ress;
    }
    unsigned long long** Modup_dcomp_batch2(unsigned long long* cipher){
        unsigned long long *res,*rest;
        unsigned long long **ress = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
        

        Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(res, N, psiinvList, size, size);

        Check(mempool((void**)&rest, N * (size + 1) * size * sizeof(unsigned long long)));

        cudaConvUp_dcomp_batch_batch<<<size * N/1024,1024>>>(N,rest,res,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
        forwardNTT_batch_batch(rest , N, psiList, size * (size + 1), size * (size + 1), size + 1);

        for(int i = 0; i < size;i++){
            ress[i] = rest + i * N * (size + 1);
        }

        // forwardNTT_batch(cipher, N, psiList, size, size);
        return ress;
    }

    unsigned long long** Modup_dcomp_batch2_fusion(unsigned long long* cipher,unsigned long long* a,unsigned long long* b,unsigned long long* suma,unsigned long long* sumb,bool isrot){
        unsigned long long *res;
        unsigned long long **ress ;
        
        static unsigned long long *rest;

        res = cipher;
        // Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        // cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(cipher, N, psiinvList, size, size);

        if(!rest)Check(mempool((void**)&rest, N * (size + 1) * size * sizeof(unsigned long long)));

        // cudaConvUp_dcomp_batch_batch<<<size * N/1024,1024>>>(N,rest,res,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
        // forwardNTT_batch_batch_fusion(rest , res, N, psiList, size * (size + 1), size * (size + 1), size + 1,a,b,suma,sumb,isrot);
        forwardNTT8(rest , res, N, psiList, size * (size + 1), size * (size + 1), size + 1,a,b,suma,sumb,isrot);

        // for(int i = 0; i < size;i++){
        //     ress[i] = rest + i * N * (size + 1);
        // }

        // forwardNTT_batch(cipher, N, psiList, size, size);
        return ress;
    }

    unsigned long long** Modup_dcomp_batch2_fusion_buff(unsigned long long* rest,unsigned long long* cipher,unsigned long long* a,unsigned long long* b,unsigned long long* suma,unsigned long long* sumb,bool isrot){
        unsigned long long *res;
        unsigned long long **ress ;
        
        res = cipher;
        // Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        // cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(cipher, N, psiinvList, size, size);

        // Check(mempool((void**)&rest, N * (size + 1) * size * sizeof(unsigned long long)));

        // cudaConvUp_dcomp_batch_batch<<<size * N/1024,1024>>>(N,rest,res,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
        forwardNTT_batch_batch_fusion(rest , res, N, psiList, size * (size + 1), size * (size + 1), size + 1,a,b,suma,sumb,isrot);

        // for(int i = 0; i < size;i++){
        //     ress[i] = rest + i * N * (size + 1);
        // }

        // forwardNTT_batch(cipher, N, psiList, size, size);
        return ress;
    }
    unsigned long long** Modup_dcomp_batch2(unsigned long long* cipher,unsigned long long* buffer1,unsigned long long* buffer2){
        unsigned long long *res,*rest;
        unsigned long long **ress = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
        

        // Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        res = buffer2;
        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(res, N, psiinvList, size, size);

        // Check(mempool((void**)&rest, N * (size + 1) * size * sizeof(unsigned long long)));
        rest = buffer1;
        cudaConvUp_dcomp_batch_batch<<<size * N/1024,1024>>>(N,rest,res,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
        forwardNTT_batch_batch(rest , N, psiList, size * (size + 1), size * (size + 1), size + 1);

        for(int i = 0; i < size;i++){
            ress[i] = rest + i * N * (size + 1);
        }

        // forwardNTT_batch(cipher, N, psiList, size, size);
        return ress;
    }
    unsigned long long** Modup_dcomp_batch(unsigned long long* cipher){
        unsigned long long *res,*rest;
        unsigned long long **ress = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
        

        Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(res, N, psiinvList, size, size);
        // cudaStream_t stream[stream_num];
        // for(int i = 0; i < stream_num; i++){
        //     cudaStreamCreate(&stream[i]);
        // }
        Check(mempool((void**)&rest, N * size * (size + 1) * sizeof(unsigned long long)));
        for(int i = 0; i < size; i++)cudaMemcpy(rest + i * N * (size + 1),res,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        for(int i = 0; i < size; i++)cudaMemcpy(rest + i * N * (size + 1),res,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        for(int i = 0; i < size; i++)cudaConvUp_dcomp<<<N/1024,1024>>>(N,rest,i,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);

        // cudaConvUp_dcomp_batch<<<N/1024,1024>>>(N,rest,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
        // forwardNTT_batch(rest, N, psiList,  size * (size + 1), size * (size + 1));
        for(int i = 0; i < size;i++){
            forwardNTT_batch(rest+ (size + 1) * N, N, psiList, size + 1, size + 1);
            ress[i] = rest + (size + 1) * N;
        }
        // for(int i = 0; i < stream_num; i++){
        //     cudaStreamSynchronize(stream[i]); 
        //     cudaStreamDestroy(stream[i]);
        // }
        forwardNTT_batch(cipher, N, psiList, size, size);


        return ress;
    }
    unsigned long long** Modup_dcomp_stream(unsigned long long* cipher){
        unsigned long long *res,*rest;
        unsigned long long **ress = (unsigned long long**)malloc(size*sizeof(unsigned long long*));
        

        Check(mempool((void**)&res, N * (size + size) * sizeof(unsigned long long)));
        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        inverseNTT_batch(res, N, psiinvList, size, size);
        cudaStream_t stream[stream_num];
        for(int i = 0; i < stream_num; i++){
            cudaStreamCreate(&stream[i]);
        }

        for(int i = 0; i < size;i++){
            Check(mempool((void**)&rest, N * (size + size) * sizeof(unsigned long long)));
            cudaMemcpyAsync(rest,res,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice,stream[i]);
            cudaConvUp_dcomp<<<N/1024,1024,0,stream[i]>>>(N,rest,i,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);
            forwardNTT_batch(stream[i],rest, N, psiList, size + 1, size + 1);
            ress[i] = rest;
        }
        for(int i = 0; i < stream_num; i++){
            cudaStreamSynchronize(stream[i]); 
            cudaStreamDestroy(stream[i]);
        }
        forwardNTT_batch(cipher, N, psiList, size, size);


        return ress;
    }
    unsigned long long* Modup(unsigned long long* cipher,cudaStream_t stream){
        unsigned long long *res;
        Check(mempool((void**)&res, N * 2 * size * sizeof(unsigned long long)));

        // for(int i = 0; i < size; i++){
        //     inverseNTT(cipher + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        // }
        inverseNTT_batch(cipher, N, psiinvList, size, size);

        cudaMemcpy(res,cipher,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        
        cudaConvUp<<<N/1024,1024,0,stream>>>(N,res,size,cudaParam.q,cudaParam.Qmod,cudaParam.q_hatinv);

        forwardNTT_batch(res, N, psiList, 2 * size, 2 * size);
        forwardNTT_batch(cipher, N, psiList, size, size);

        // for(int i = 0; i < 2 * size; i++){
        //     forwardNTT(res+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }        
        // for(int i = 0; i < size; i++){
        //     forwardNTT(cipher+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        return res;
    }

    unsigned long long* Moddown(unsigned long long* cipher){
        unsigned long long *res;
        Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));
        // for(int i = 0; i < 2 * size; i++){
        //     inverseNTT(cipher + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        // }
        // inverseNTT_batch(cipher, N, psiinvList, size, size);

        // print<<<1,1>>>(cipher,16);
        cudaConvDown<<<N/1024,1024>>>(N,res,cipher,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv);
        // print<<<1,1>>>(res,8);

        // forwardNTT_batch(cipher, N, psiList, 2 * size, 2 * size);
        // forwardNTT_batch(res, N, psiList, size, size);
        // for(int i = 0; i < size; i++){
        //     forwardNTT(res+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        // for(int i = 0; i < 2 * size; i++){
        //     forwardNTT(cipher+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        return res;
    }

    unsigned long long* Moddown_dcomp1(unsigned long long* cipher){
        unsigned long long *res;
        Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));

        // for(int i = 0; i < stream_num; i++){
        //     cudaStreamCreate(&stream[i]);
        // }
        // for(int i = 0; i < size+1; i++){
            // inverseNTT(cipher + N * i,N,streams[i],q[i],mu[i],q_bit[i],psiinvTable[i]);
        // }
        // for(int i = 0; i < stream_num; i++){
            // cudaStreamSynchronize(stream[i]); 
            // cudaStreamDestroy(stream[i]);
        // }
        inverseNTT_batch(cipher, N, psiinvList,  size+1, size+1);

        // print_dd<<<1,1>>>(cipher,16);
        
        cudaConvDown_dcomp<<<N/1024,1024>>>(N,res,cipher,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        // print<<<1,1>>>(res,8);
        // print_dd<<<1,1>>>(cipher,16);

        // forwardNTT_batch(cipher, N, psiList, size+1, size+1);
        forwardNTT_batch(res, N, psiList, size, size);
        // for(int i = 0; i < size; i++){
        //     forwardNTT(res+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        // for(int i = 0; i < 2 * size; i++){
        //     forwardNTT(cipher+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        return res;
    }
    unsigned long long* Moddown_dcomp(unsigned long long* cipher){
        unsigned long long *res;
        Check(mempool((void**)&res,2 * N * size * sizeof(unsigned long long)));

        // inverseNTT_batch(cipher, N, psiinvList,  size + 1, size + 1);
        inverseNTT(cipher + N * size,N,ntt,q[size],mu[size],q_bit[size],psiinvTable[size]);
        // print_dd<<<1,1>>>(cipher,16);
        
        cudaMemcpy(res + N * size,cipher+ N * size,N * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        // print_dd<<<1,1>>>(res,16);
        // cudaConvMov_dcomp<<<N/1024,1024>>>(N,res,res,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        fusionModdown(res, cipher, N, psiList, size,cudaParam.spinv);
        // forwardNTT3(res, N, psiList, size);
        // print_dd<<<1,1>>>(res,16);

        // cudaConvDown_dcomp<<<N/1024,1024>>>(N,res,cipher,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        // cudaFusionDown<<<N/1024,1024>>>(N,cipher,res,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        // cudaConvDown_dcomp<<<N/1024,1024>>>(N,res,cipher,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);



        return cipher;
    }
    unsigned long long* Moddown_dcomp(unsigned long long* cipher,unsigned long long *res){
        ;

        inverseNTT(cipher + N * size,N,ntt,q[size],mu[size],q_bit[size],psiinvTable[size]);
        cudaMemcpy(res + N * size,cipher+ N * size,N * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        fusionModdown(res, cipher, N, psiList, size,cudaParam.spinv);

        return cipher;
    }
    unsigned long long* Moddown_dcomp_double(unsigned long long* cipher){
        unsigned long long *res;
        Check(mempool((void**)&res,2 * N * size * sizeof(unsigned long long)));
        inverseNTT(cipher + N * size,N,ntt,q[size],mu[size],q_bit[size],psiinvTable[size]);
        cudaMemcpy(res + N * size,cipher+ N * size,N * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        fusionModdown(res, cipher, N, psiList, size,cudaParam.spinv);
        return cipher;
    }
    unsigned long long* Moddown_dcomp11(unsigned long long* cipher){
        unsigned long long *res;
        Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));
        inverseNTT(cipher + N * size,N,ntt,q[size],mu[size],q_bit[size],psiinvTable[size]);
        cudaConvMov_dcomp<<<N/1024,1024>>>(N,res,cipher,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        forwardNTT_batch(res, N, psiList, size, size);
        cudaConvDown_dcomp_new<<<N/1024,1024>>>(N,cipher,res,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv,cudaParam.spinv);
        return cipher;
    }
    unsigned long long* Moddown(unsigned long long* cipher,cudaStream_t stream){
        unsigned long long *res;
        Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));
        // for(int i = 0; i < 2 * size; i++){
        //     inverseNTT(cipher + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        // }
        // inverseNTT_batch(cipher, N, psiinvList, size, size);

        // print<<<1,1>>>(cipher,16);
        cudaConvDown<<<N/1024,1024,0,stream>>>(N,res,cipher,size,cudaParam.q,cudaParam.Pmod,cudaParam.p_hatinv,cudaParam.Pinv);
        // print<<<1,1>>>(res,8);

        // forwardNTT_batch(cipher, N, psiList, 2 * size, 2 * size);
        // forwardNTT_batch(res, N, psiList, size, size);
        // for(int i = 0; i < size; i++){
        //     forwardNTT(res+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        // for(int i = 0; i < 2 * size; i++){
        //     forwardNTT(cipher+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        // }
        return res;
    }
    cipherText BootStrapping(cipherText& cipher,galoisKey* galois,galoisKey* galois_right,galoisKeyRoot* baby,galoisKeyRoot* gaint){
        
        printf("coeff2slot\n");

        for(int i = 0; i < size; i++){
            inverseNTT(cipher.a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        for(int i = 0; i < size; i++){
            inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }

        
        print<<<1,1>>>(cipher.a,8);
        print<<<1,1>>>(cipher.b,8);
        cudaModRaise<<<N/1024,1024>>>(cipher.a,N,size,cudaParam.q);
        cudaModRaise<<<N/1024,1024>>>(cipher.b,N,size,cudaParam.q);
        print<<<1,1>>>(cipher.a,8);
        print<<<1,1>>>(cipher.b,8);
        for(int i = 0; i < size; i++){
            forwardNTT(cipher.a +N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        for(int i = 0; i < size; i++){
            forwardNTT(cipher.b +N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        // cipher.depth = 0;
        print<<<1,1>>>(cipher.a,8);
        BigMatrix Coeff2SlotU = encoder.Coeff2SlotMatrixU();
        BigMatrix Coeff2SlotV = encoder.Coeff2SlotMatrixV();
        
        cipherText slota = dot(Coeff2SlotU,cipher,galois,galois_right,baby,gaint);
        // return slota;
        cipherText slotb = dot(Coeff2SlotV,cipher,galois,galois_right,baby,gaint);
        return slota;
        // return slotb;
        printf("slot2coeff\n");
        BigMatrix Slot2CoeffU = encoder.Slot2CoeffMatrixU();
        BigMatrix Slot2CoeffV = encoder.Slot2CoeffMatrixV();
        
        cipherText coeffa = dot(Slot2CoeffU,slota,galois,galois_right,baby,gaint);
        cipherText coeffb = dot(Slot2CoeffV,slotb,galois,galois_right,baby,gaint);
        
        addcipter(coeffa,coeffb);
        return coeffa;
        cipher.a = coeffa.a;
        cipher.b = coeffa.b;
    }
    // void ModRaise(cipherText& cipher){
    //     // unsigned long long *a,*b;
    //     // Check(mempool((void**)&a, N * size * sizeof(unsigned long long)));
    //     // Check(mempool((void**)&b, N * size * sizeof(unsigned long long)));
    //     cudaModRaise<<<N/1024,1024>>>(N,cipher.a,size,cudaParam.q);
    //     cudaModRaise<<<N/1024,1024>>>(N,cipher.b,size,cudaParam.q);
    // }
    // cipherText Mod(cipherText& cipher, unsigned long long modnum){
        
    // }
    // cipherTextUp Modup(cipherText& cipher){
    //     unsigned long long *res_a,*res_b;
    //     Check(mempool((void**)&res_a, N * 2 * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b, N * 2 * size * sizeof(unsigned long long)));
    //     for(int i = 0; i < size; i++){
    //         inverseNTT(cipher.a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //     }

    //     cudaMemcpy(res_a,cipher.a,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //     cudaMemcpy(res_b,cipher.b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        
    //     cudaConvUp<<<N/1024,1024>>>(N,res_a,size,q,Qmod,q_hatinv);
    //     cudaConvUp<<<N/1024,1024>>>(N,res_b,size,q,Qmod,q_hatinv);
    //     for(int i = 0; i < 2 * size; i++){
    //         forwardNTT(res_a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         forwardNTT(res_b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //     }        
    //     for(int i = 0; i < size; i++){
    //         forwardNTT(cipher.a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         forwardNTT(cipher.b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //     }
    //     cipherTextUp res;
    //     res.set(res_a,res_b);
    //     res.depth = cipher.depth;
    //     return res;
    // }
    // cipherText Moddown(cipherTextUp& cipher){
    //     unsigned long long *res_a,*res_b;
    //     Check(mempool((void**)&res_a, N * size * sizeof(unsigned long long)));
    //     Check(mempool((void**)&res_b, N * size * sizeof(unsigned long long)));
    //     for(int i = 0; i < 2 * size; i++){
    //         inverseNTT(cipher.a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //     }
        
    //     cudaConvDown<<<N/1024,1024>>>(N,res_a,cipher.a,size,q,Pmod,p_hatinv,Pinv);
    //     for(int i = 0; i < size; i++){
    //         forwardNTT(cipher.a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         forwardNTT(cipher.b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //     }
    //     cipherText res;
    //     res.set(res_a,res_b);
    //     res.depth = cipher.depth;
    //     return res;
    // }
    
    // void conv(cipherText& cipher, unsigned long long* plain){
    //     for(int i = 0; i < size; i++){
    //         forwardNTT(cipher.a+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         forwardNTT(cipher.b+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         forwardNTT(plain+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //     }
    //     for(int i = 0; i < size; i++){
    //         // print<<<1,1>>>(cipher.a + N * i);
    //         polymul<<<N/1024,1024>>>(cipher.a + N * i,plain + N * i,cipher.a + N * i,q[i],mu[i],q_bit[i]);
    //         polymul<<<N/1024,1024>>>(cipher.b + N * i,plain + N * i,cipher.b + N * i,q[i],mu[i],q_bit[i]);
    //         // print<<<1,1>>>(cipher.a + N * i);
    //     }
    //     for(int i = 0; i < size; i++){
    //         inverseNTT(cipher.a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         inverseNTT(plain + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //     }
    //     // rescale(cipher);
    // }
    // void conv(unsigned long long* plain1, unsigned long long* plain2){
    //     for(int i = 0; i < size; i++){
    //         forwardNTT(plain1+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         forwardNTT(plain2+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         // forwardNTT(plain+N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //     }
    //     for(int i = 0; i < size; i++){
    //         // print<<<1,1>>>(cipher.a + N * i);
    //         polymul<<<N/1024,1024>>>(plain1 + N * i,plain2 + N * i,plain1 + N * i,q[i],mu[i],q_bit[i]);
    //         // polymul<<<N/1024,1024>>>(cipher.b + N * i,plain + N * i,cipher.b + N * i,q[i],mu[i],q_bit[i]);
    //         // print<<<1,1>>>(cipher.a + N * i);
    //     }
    //     for(int i = 0; i < size; i++){
    //         // inverseNTT(cipher.a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         // inverseNTT(cipher.b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         inverseNTT(plain1 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         inverseNTT(plain2 + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);

    //     }
    //     // rescale(cipher);
    // }
};

