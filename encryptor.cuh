#pragma once

#include "random.cuh"
#include "polycalc.cuh"


#define ROT

const int baby_step = 2;



//Nbits babyr 


// #include "evaluator.cuh"
__global__ void print(unsigned long long* a){
    for(int i = 0; i < 128;i++){
        printf("%llu,",a[i]);
    }printf("\n\n\n\n\n");
    // for(int i = 8192 * 2 - 128; i < 8192 * 2;i++){
    //     printf("%llu,",a[i]);
    // }printf("\n\n\n\n\n");
}
__global__ void print_t(unsigned long long* a){
    for(int i = 0; i < 8192 * 2;i++){
        printf("%llu,",a[i+8192 * 2 * 0]);
    }printf("\n\n\n\n\n");
    // for(int i = 8192 * 2 - 128; i < 8192 * 2;i++){
    //     printf("%llu,",a[i]);
    // }printf("\n\n\n\n\n");
}
__global__ void print_ddd(unsigned long long* a){
    for(int i = 0; i < 16;i++){
        printf("%llu,",a[i*8192+0]);
    }printf("\n");
}
__global__ void print(unsigned long long* a,int size);
class doublePoly{
    public:
    unsigned long long* a;
    unsigned long long* b;
    int depth = 0;
    doublePoly(unsigned long long* a,unsigned long long *b){
        this->a = a;
        this->b = b;
    }
    doublePoly(){
        this->a = nullptr;
        this->b = nullptr;
    }
    void set(unsigned long long* a,unsigned long long *b){
        this->a = a;
        this->b = b;
    }
    void set(unsigned long long *b){
        this->b = b;
    }
    // ~doublePoly(){
    //     if(a)cudaFree(a);
    //     if(b)cudaFree(b);
    // }
};
class triplePoly{
    public:
    unsigned long long* a;
    unsigned long long* b;
    unsigned long long* c;
    int depth = 0;
    triplePoly(unsigned long long* a,unsigned long long *b,unsigned long long *c){
        this->a = a;
        this->b = b;
        this->c = c;
    }
    triplePoly(){
        this->a = nullptr;
        this->b = nullptr;
        this->c = nullptr;
    }
    void set(unsigned long long* a,unsigned long long *b,unsigned long long *c){
        this->a = a;
        this->b = b;
        this->c = c;
    }
    
    // ~doublePoly(){
    //     if(a)cudaFree(a);
    //     if(b)cudaFree(b);
    // }
};

class cipherText:public doublePoly{

};

class publicKey:public doublePoly{

};

class privateKey:public doublePoly{

};
class relienKey:public doublePoly{

};
class galoisKey:public doublePoly{

};
class galoisKeyRoot:public doublePoly{
    
};
class cipherTextUp:public doublePoly{
    
};
class keyGen{
    public:
    publicKey pub;
    privateKey pri;
    relienKey relien;
    relienKey* reliendcomp;
    galoisKey* galois;
    galoisKey* galois_right;
    galoisKeyRoot* baby;
    galoisKeyRoot* gaint;
    int N;
    double scale;
    unsigned long long** psiTable;
    unsigned long long** psiinvTable; 
    unsigned long long* psi;
    unsigned long long* psiinv;
    unsigned long long* q;
    unsigned long long* mu;
    // unsigned long long* ninv;
    unsigned long long size;
    unsigned long long* q_bit;
    unsigned long long* Qmod;
    unsigned long long* q_hatinv;
    unsigned long long* Pmod;
    unsigned long long* p_hatinv;
    unsigned long long* Pinv;
    unsigned long long* Ps;

    galoisKeyRoot* gaintcomp;
    galoisKey* galoiscomp;
    galoisKeyRoot* babycomp;
    galoisKeyRoot* babycompr;
    galoisKey* galoiscomp_r;

    int Nbits;
    cudaStream_t ntt;
    RNS rns;

    keyGen(int n,double scale,int size):rns(size,scale){
        // const int baby_step = 32;
        N = n * 2;
        this->scale = scale;
        this->size = size;
        Nbits = log2(N);
        
        Nbits = 1;

    printf("!!%d\n",__LINE__);

        Ps = (unsigned long long *)malloc(size * sizeof(unsigned long long));
        reliendcomp = (relienKey*)malloc(size * sizeof(relienKey));
        galois = (galoisKey*)malloc(Nbits * sizeof(galoisKey));
        galois_right = (galoisKey*)malloc(Nbits * sizeof(galoisKey));
        baby = (galoisKeyRoot*)malloc(baby_step * sizeof(galoisKeyRoot));
        babycomp = (galoisKeyRoot*)malloc(size * baby_step * sizeof(galoisKeyRoot));
        babycompr = (galoisKeyRoot*)malloc(size * baby_step * sizeof(galoisKeyRoot));

        gaintcomp = (galoisKeyRoot*)malloc(size * n/baby_step * sizeof(galoisKeyRoot));
        galoiscomp = (galoisKey*)malloc(size * Nbits * sizeof(galoisKey));
        galoiscomp_r = (galoisKey*)malloc(size * Nbits * sizeof(galoisKey));
        // galoiscomp_r = (galoisKey*)malloc(size * 16 * sizeof(galoisKey));
    printf("!!%d\n",__LINE__);

        gaint = (galoisKeyRoot*)malloc(n/baby_step * sizeof(galoisKeyRoot));
        q = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        psi = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        psiinv = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        q_bit = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
        Qmod = (unsigned long long *)malloc(size * size * sizeof(unsigned long long));
        q_hatinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
        Pmod = (unsigned long long*)malloc(size * size * sizeof(unsigned long long *));
        p_hatinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
        Pinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
    printf("!!%d\n",__LINE__);

        getPs(Ps,size);
    printf("!!%d\n",__LINE__);

        getParams(q, psi, psiinv, q_bit, Qmod,q_hatinv,Pmod,p_hatinv,Pinv,size);
    printf("!!%d\n",__LINE__);

        mu = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long));
        for(int i = 0; i < 2 * size; i++){
            uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
            mu[i] = (mu1 / q[i]).low;
        }
    printf("!!%d\n",__LINE__);

        psiTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));
        psiinvTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));
        for(int i = 0; i <2 * size; i++){
            Check(mempool(&psiTable[i], N * sizeof(unsigned long long)));
            Check(mempool(&psiinvTable[i], N * sizeof(unsigned long long)));
        }
    printf("!!%d\n",__LINE__);

        for(int i = 0; i <2 * size; i++){
            fillTablePsi128<<<N/1024,1024>>>(psi[i], q[i], psiinv[i], psiTable[i], psiinvTable[i], log2(N));
            uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
            mu[i] = (mu1 / q[i]).low;

            // unsigned long long *t;
            // Check(mempool(&t, N * sizeof(unsigned long long)));
            // forwardNTT3(t, N/2, psiTable[0],1);
            // // forwardNTT3(t, N/2, psiTable[0],1);

            // cudaDeviceSynchronize();
            // printf("%d\n",1/0);
        }
        ntt = 0;
    printf("!!%d\n",__LINE__);

        unsigned long long *pub_a,*pub_b,*pri_b,*pri_b_l,*relien_a,*relien_b,*e,*galois_a,*galois_b,*galois_a_r,*galois_b_r,*galois_a_baby,*galois_b_baby,*galois_a_gaint,*galois_b_gaint;
        Check(mempool((void**)&pub_a, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&pub_b, size * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&relien_a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&relien_b, 2 * N * size * sizeof(unsigned long long)));
        // Check(mempool((void**)&galois_a, N * size * sizeof(unsigned long long)));
        
        Check(mempool((void**)&pri_b,2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&e, N * size * sizeof(unsigned long long)));
        // Check(mempool((void**)&e, N * sizeof(unsigned long long)));

        genRandom<<<N/1024,1024>>>(pub_a,0);
        pub_a = rns.decompose(pub_a,N);
        for(int i = 0; i < size; i++){
            forwardNTT(pub_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }

    printf("!!%d\n",__LINE__);

        // genRandom<<<N/1024,1024>>>(pub_b,rns.moduleChain_h[0]);
        genRandom<<<N/1024,1024>>>(pub_b,scale);
        pub_b = rns.decomposeLongBasis(pub_b,N);

        // pub_b = rns.decompose(pub_b,N);
        // rns.decompose(pub_b);
        for(int  i = 0; i < size; i++){
            forwardNTT(pub_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        // print<<<1,1>>>(pub_b);
        genRandom_s<<<N/1024,1024>>>(pri_b,scale);
        pri_b_l = rns.decomposeLongBasis(pri_b,N);
        pri_b = rns.decompose(pri_b,N);
        for(int i = 0; i < size; i++){
            forwardNTT(pri_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        // print<<<1,1>>>(pri_b);
        for(int i = 0; i < 2 * size; i++){
            forwardNTT(pri_b_l + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }

        for(int i = 0; i < size; i++){
            polymuladd<<<N/1024,1024>>>(pri_b + N * i,pub_b + N * i,pub_a + N * i,pub_a + N * i,q[i],mu[i],q_bit[i]);
        }
        for(int i = 0; i < size; i++){
            inverseNTT(pub_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        for(int i = 0; i < size; i++){
            inverseNTT(pub_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        for(int i = 0; i < size; i++){
            inverseNTT(pri_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        }
        // print<<<1,1>>>(pub_a);
        // print<<<1,1>>>(pub_b);
        // print<<<1,1>>>(pri_b);
        for(int i = 0; i < size; i++){
            forwardNTT(pub_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            forwardNTT(pub_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            forwardNTT(pri_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        pub.set(pub_a,pub_b);
        pri.set(pri_b);
    printf("!!%d\n",__LINE__);

        genRandom<<<N/1024,1024>>>(relien_b,0);

        // exit(1);
        relien_b = rns.decomposeLongBasis(relien_b,N);
    
        for(int i = 0; i < 2 * size; i++){
            forwardNTT(relien_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        unsigned long long *s_2;
        Check(mempool((void**)&s_2, 2 * N * size * sizeof(unsigned long long)));

        for(int i = 0; i < 2 * size; i++){            
            polymul<<<N/1024,1024>>>(pri_b_l + N * i,pri_b_l + N * i,s_2 + N * i,q[i],mu[i],q_bit[i]);
        }
        for(int i = 0; i < 2 * size; i++){            
            polymul<<<N/1024,1024>>>(pri_b_l+ N * i,relien_b + N * i,relien_a + N * i,q[i],mu[i],q_bit[i]);
        }
        unsigned long long *relien_a_dcomp,*relien_b_dcomp;

        Check(mempool((void**)&relien_a_dcomp, size * 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&relien_b_dcomp, size * 2 * N * size * sizeof(unsigned long long)));
        genRandom<<<N/1024,1024>>>(e,0);

        e = rns.decomposeLongBasis(e,N);
        for(int i = 0; i < 2 * size; i++){
            forwardNTT(e + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        for(int j = 0; j < 2 * size; j++){            
            polyadd<<<N/1024,1024>>>(relien_a + N * j,e + N * j,relien_a + N * j,N,q[j]);
        }  

        for(int i = 0; i < size; i++){
            cudaMemcpy(relien_a_dcomp + i * 2 * N * size ,relien_a,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            cudaMemcpy(relien_b_dcomp + i * 2 * N * size ,relien_b,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        }
        unsigned long long* s_2_dcomp;
        Check(mempool((void**)&s_2_dcomp, 2 * N * size * sizeof(unsigned long long)));

        cudaMemcpy(s_2_dcomp ,s_2, N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        // print_ddd<<<1,1>>>(relien_b_dcomp);
        for(int i = 0; i <  size; i++){
            polymulInteger<<<N/1024,1024>>>(s_2_dcomp + N * i,q[size],q[i],mu[i],q_bit[i]);
        }
    printf("!!%d\n",__LINE__);

        // print_ddd<<<1,1>>>(s_2_dcomp);
        // print_ddd<<<1,1>>>(relien_b_dcomp + 2 * N * size * 4);
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                if(i == j)polyadd<<<N/1024,1024>>>(s_2_dcomp + N * i,relien_a_dcomp + 2 * N * size * i + j * N,relien_a_dcomp + 2 * N * size * i + j * N,N,q[j]);
                else polyadd<<<N/1024,1024>>>(0,relien_a_dcomp + 2 * N * size * i + j * N,relien_a_dcomp + 2 * N * size * i + j * N,N,q[j]);
            }

            reliendcomp[i].set(relien_a_dcomp + 2 * N * size * i,relien_b_dcomp + i * 2 * N * size);
        }
        // print_ddd<<<1,1>>>(reliendcomp[0].a);
        
        for(int i = 0; i < size; i++){
            polymulInteger<<<N/1024,1024>>>(s_2 + N * i,Ps[i],q[i],mu[i],q_bit[i]);
        }

        for(int i = 0; i < 2 * size; i++){            
            polyadd<<<N/1024,1024>>>(s_2 + N * i,relien_a + N * i,relien_a + N * i,N,q[i]);
        } 
        
    printf("!!%d\n",__LINE__);

        


        genRandom<<<N/1024,1024>>>(e,0);
        e = rns.decomposeLongBasis(e,N);

        for(int i = 0; i < 2 * size; i++){            
            polyadd<<<N/1024,1024>>>(relien_a+ N * i,e + N * i,relien_a + N * i,N,q[i]);
        }        
        relien.set(relien_a,relien_b);
        int elt = 3;  
        unsigned long long *rottmp;
        Check(mempool((void**)&rottmp, 2 * N * size * sizeof(unsigned long long)));
        for(int idx = 0;idx < Nbits; idx++){
            Check(mempool((void**)&galois_b, N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&galois_a, N * size * sizeof(unsigned long long)));
            genRandom<<<N/1024,1024>>>(galois_b,scale);
            galois_b = rns.decompose(galois_b,N);
            for(int i = 0; i < size; i++){
                forwardNTT(galois_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            for(int i = 0; i < size; i++){
                inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }
            for(int i = 0; i < size; i++){
                cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a + N * i,q[i],elt,N);
            }
            for(int i = 0; i < size; i++){
                forwardNTT(galois_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            for(int i = 0; i < size; i++){
                polymulminus<<<N/1024,1024>>>(galois_b + N * i,pri_b + N * i,galois_a + N * i,galois_a + N * i,q[i],mu[i],q_bit[i]);
            }
            elt = (elt * elt) % (2 * N);
            galois[idx].set(galois_a,galois_b);
        }
    printf("!!%d\n",__LINE__);

        // elt = 3;  
        // Check(mempool((void**)&rottmp, 2 * N * size * sizeof(unsigned long long)));
        // for(int idx = 0;idx < Nbits; idx++){
        //     Check(mempool((void**)&galois_b, N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a, N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b,scale);
        //     galois_b = rns.decompose(galois_b,N);
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b_l,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < 2 * size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }
        //     for(int i = 0; i < 2 * size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a + N * i,q[i],elt,N);
        //     }

        //     for(int i = 0; i < 2 * size; i++){
        //         forwardNTT(galois_a + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }

        //     for(int j = 0; j < size; j++){
        //         unsigned long long *galois_b_comp,*galois_a_comp;
        //         Check(mempool((void**)&galois_b_comp,2 * N * size * sizeof(unsigned long long)));
        //         Check(mempool((void**)&galois_a_comp,2 * N * size * sizeof(unsigned long long)));
        //         cudaMemcpy(galois_a_comp,galois_a,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //         cudaMemcpy(galois_b_comp,galois_b,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        //         for(int i = 0; i < 2 * size; i++){
        //             polymulminus<<<N/1024,1024>>>(galois_b_comp + N * i,pri_b + N * j,galois_a_comp + N * i,galois_a_comp + N * i,q[i],mu[i],q_bit[i]);
        //         }
        //         galoiscomp[idx*size + j].set(galois_a_comp,galois_b_comp);
        //     }
        //     elt = (elt * elt) % (2 * N);
        // }
        elt = 3;
        for(int idx = 0;idx < Nbits; idx++){
            Check(mempool((void**)&galois_b,2 * N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&galois_a,2 * N * size * sizeof(unsigned long long)));
            genRandom<<<N/1024,1024>>>(galois_b,scale);
            galois_b = rns.decomposeLongBasis(galois_b,N);
            for(int i = 0; i < 2 * size; i++){
                forwardNTT(galois_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            unsigned long long* rots;
            Check(mempool((void**)&rots, 2 * N * size * sizeof(unsigned long long)));

            cudaMemcpy(rottmp,pri_b_l,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

            for(int i = 0; i < 2 * size; i++){
                inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }

            for(int i = 0; i < 2 * size; i++){
                cudarotation<<<N/1024,1024>>>(rottmp + N * i,rots + N * i,q[i],elt,N);
            }

            for(int i = 0; i < 2 * size; i++){
                forwardNTT(rots + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }


            for(int i = 0; i < 2 * size; i++){            
                polymul<<<N/1024,1024>>>(pri_b_l+ N * i,galois_b + N * i,galois_a + N * i,q[i],mu[i],q_bit[i]);
            }
            unsigned long long *galois_a_comp,*galois_b_comp;

            Check(mempool((void**)&galois_a_comp, size * 2 * N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&galois_b_comp, size * 2 * N * size * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++){
                cudaMemcpy(galois_a_comp + i * 2 * N * size ,galois_a,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
                cudaMemcpy(galois_b_comp + i * 2 * N * size ,galois_b,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            }
            unsigned long long* rots_dcomp;
            Check(mempool((void**)&rots_dcomp, 2 * N * size * sizeof(unsigned long long)));

            cudaMemcpy(rots_dcomp ,rots, N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            for(int i = 0; i <  size; i++){
                polymulInteger<<<N/1024,1024>>>(rots_dcomp + N * i,q[size],q[i],mu[i],q_bit[i]);
            }
            for(int i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                    if(i == j)polyadd<<<N/1024,1024>>>(rots_dcomp + N * i,galois_a_comp + 2 * N * size * i + j * N,galois_a_comp + 2 * N * size * i + j * N,N,q[j]);
                    else polyadd<<<N/1024,1024>>>(0,galois_a_comp + 2 * N * size * i + j * N,galois_a_comp + 2 * N * size * i + j * N,N,q[j]);
                }
                galoiscomp[idx*size + i].set(galois_a_comp,galois_b_comp);
            }
            elt = (elt * elt) % (2 * N);

        }
    printf("!!%d\n",__LINE__);


        elt = modpow128(3,2*N-1,(2*N));
        // for(int idx = 0;idx < 1; idx++){
        for(int idx = 0;idx < Nbits; idx++){
            Check(mempool((void**)&galois_b,2 * N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&galois_a,2 * N * size * sizeof(unsigned long long)));
            genRandom<<<N/1024,1024>>>(galois_b,scale);
            galois_b = rns.decomposeLongBasis(galois_b,N);
            for(int i = 0; i < 2 * size; i++){
                forwardNTT(galois_b + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            unsigned long long* rots;
            Check(mempool((void**)&rots, 2 * N * size * sizeof(unsigned long long)));

            cudaMemcpy(rottmp,pri_b_l,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

            for(int i = 0; i < 2 * size; i++){
                inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
            }

            for(int i = 0; i < 2 * size; i++){
                cudarotation<<<N/1024,1024>>>(rottmp + N * i,rots + N * i,q[i],elt,N);
            }

            for(int i = 0; i < 2 * size; i++){
                forwardNTT(rots + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }


            for(int i = 0; i < 2 * size; i++){            
                polymul<<<N/1024,1024>>>(pri_b_l+ N * i,galois_b + N * i,galois_a + N * i,q[i],mu[i],q_bit[i]);
            }
            unsigned long long *galois_a_comp,*galois_b_comp;

            Check(mempool((void**)&galois_a_comp, size * 2 * N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&galois_b_comp, size * 2 * N * size * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++){
                cudaMemcpy(galois_a_comp + i * 2 * N * size ,galois_a,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
                cudaMemcpy(galois_b_comp + i * 2 * N * size ,galois_b,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            }
            unsigned long long* rots_dcomp;
            Check(mempool((void**)&rots_dcomp, 2 * N * size * sizeof(unsigned long long)));

            cudaMemcpy(rots_dcomp ,rots, N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
            for(int i = 0; i <  size; i++){
                polymulInteger<<<N/1024,1024>>>(rots_dcomp + N * i,q[size],q[i],mu[i],q_bit[i]);
            }
            for(int i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                    if(i == j)polyadd<<<N/1024,1024>>>(rots_dcomp + N * i,galois_a_comp + 2 * N * size * i + j * N,galois_a_comp + 2 * N * size * i + j * N,N,q[j]);
                    else polyadd<<<N/1024,1024>>>(0,galois_a_comp + 2 * N * size * i + j * N,galois_a_comp + 2 * N * size * i + j * N,N,q[j]);
                }
                galoiscomp_r[idx*size + i].set(galois_a_comp,galois_b_comp);
            }
            elt = (elt * elt) % (2 * N);

        }

        // elt = modpow128(3,n-1,(2*N));
        // for(int idx = 0;idx < Nbits; idx++){
        //     Check(mempool((void**)&galois_b_r, N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_r, N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_r,scale);
        //     galois_b_r = rns.decompose(galois_b_r,N);
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_b_r + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a_r + N * i,q[i],elt,N);
        //     }
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_a_r + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         polymulminus<<<N/1024,1024>>>(galois_b_r + N * i,pri_b + N * i,galois_a_r + N * i,galois_a_r + N * i,q[i],mu[i],q_bit[i]);
        //     }
        //     elt = (elt * elt) % (2 * N);
        //     galois_right[idx].set(galois_a_r,galois_b_r);
        // }

        //     printf("!!%d\n",__LINE__);

        // for(int idx = 0;idx < baby_step; idx++){
        //     elt = modpow128(3,n-idx,(2*N));
        //     Check(mempool((void**)&galois_b_baby, N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_baby, N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_baby,scale);
        //     galois_b_baby = rns.decompose(galois_b_baby,N);
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_b_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a_baby + N * i,q[i],elt,N);
        //     }
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_a_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         polymulminus<<<N/1024,1024>>>(galois_b_baby + N * i,pri_b + N * i,galois_a_baby + N * i,galois_a_baby + N * i,q[i],mu[i],q_bit[i]);
        //     }
        //     baby[idx].set(galois_a_baby,galois_b_baby);
        // }

#ifdef ROT
        // for(int idx = 0;idx < baby_step; idx++){
        //     elt = modpow128(3,n-idx,(2*N));
        //     Check(mempool((void**)&galois_b_baby,2 * N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_baby,2 * N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_baby,scale);
        //     galois_b_baby = rns.decomposeLongBasis(galois_b_baby,N);
        //     for(int i = 0; i < 2 * size; i++){
        //         forwardNTT(galois_b_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }

        //     for(int i = 0; i < size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a_baby + N * i,q[i],elt,N);
        //     }

        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_a_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         polymulInteger<<<N/1024,1024>>>(galois_a_baby + N * i,q[size],q[i],mu[i],q_bit[i]);
        //     }

        //     for(int j = 0; j < size; j++){
        //         unsigned long long *galois_b_baby_comp,*galois_a_baby_comp;
        //         Check(mempool((void**)&galois_b_baby_comp,2 * N * size * sizeof(unsigned long long)));
        //         Check(mempool((void**)&galois_a_baby_comp,2 * N * size * sizeof(unsigned long long)));
        //         cudaMemcpy(galois_a_baby_comp,galois_a_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //         cudaMemcpy(galois_b_baby_comp,galois_b_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        //         for(int i = 0; i < 2 * size; i++){
        //             polymuladd<<<N/1024,1024>>>(galois_b_baby + N * i,pri_b_l + N * j,galois_a_baby + N * i,galois_a_baby + N * i,q[i],mu[i],q_bit[i]);
        //         }
        //         babycomp[idx*size + j].set(galois_a_baby,galois_b_baby);
        //     }
        // }
//===================================
    printf("!!%d\n",__LINE__);

    //     for(int idx = 1;idx < baby_step; idx++){
    //         if(idx > 17 && idx % 32 != 0){
    //             continue;
    //         }
    //         elt = modpow128(3,n-idx,(2*N));
    //         Check(mempool((void**)&galois_b_baby,2 * N * size * sizeof(unsigned long long)));
    //         Check(mempool((void**)&galois_a_baby,2 * N * size * sizeof(unsigned long long)));
    //         genRandom<<<N/1024,1024>>>(galois_b_baby,scale);
    //         galois_b_baby = rns.decomposeLongBasis(galois_b_baby,N);
    //         for(int i = 0; i < 2 * size; i++){
    //             forwardNTT(galois_b_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         }
    //         unsigned long long* rots;
    //         Check(mempool((void**)&rots, 2 * N * size * sizeof(unsigned long long)));

    //         cudaMemcpy(rottmp,pri_b_l,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

    //         for(int i = 0; i < 2 * size; i++){
    //             inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         }

    //         for(int i = 0; i < 2 * size; i++){
    //             cudarotation<<<N/1024,1024>>>(rottmp + N * i,rots + N * i,q[i],elt,N);
    //         }

    //         for(int i = 0; i < 2 * size; i++){
    //             forwardNTT(rots + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         }


    //         for(int i = 0; i < 2 * size; i++){            
    //             polymul<<<N/1024,1024>>>(pri_b_l+ N * i,galois_b_baby + N * i,galois_a_baby + N * i,q[i],mu[i],q_bit[i]);
    //         }
    //         unsigned long long *galois_a_baby_comp,*galois_b_baby_comp;

    //         Check(mempool((void**)&galois_a_baby_comp, size * 2 * N * size * sizeof(unsigned long long)));
    //         Check(mempool((void**)&galois_b_baby_comp, size * 2 * N * size * sizeof(unsigned long long)));
    //         for(int i = 0; i < size; i++){
    //             cudaMemcpy(galois_a_baby_comp + i * 2 * N * size ,galois_a_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //             cudaMemcpy(galois_b_baby_comp + i * 2 * N * size ,galois_b_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         }
    //         unsigned long long* rots_dcomp;
    //         Check(mempool((void**)&rots_dcomp, 2 * N * size * sizeof(unsigned long long)));

    //         cudaMemcpy(rots_dcomp ,rots, N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         for(int i = 0; i <  size; i++){
    //             polymulInteger<<<N/1024,1024>>>(rots_dcomp + N * i,q[size],q[i],mu[i],q_bit[i]);
    //         }
    //         for(int i = 0; i < size; i++){
    //             // if(idx == 2){
    //             //     babycomp[idx*size + i].set(galois_a_baby_comp,galois_b_baby_comp);
    //             //     continue;
    //             // }
                
    //             for(int j = 0; j < size; j++){
    //                 if(i == j)polyadd<<<N/1024,1024>>>(rots_dcomp + N * i,galois_a_baby_comp + 2 * N * size * i + j * N,galois_a_baby_comp + 2 * N * size * i + j * N,N,q[j]);
    //                 else polyadd<<<N/1024,1024>>>(0,galois_a_baby_comp + 2 * N * size * i + j * N,galois_a_baby_comp + 2 * N * size * i + j * N,N,q[j]);
    //             }
    //             babycomp[idx*size + i].set(galois_a_baby_comp,galois_b_baby_comp);
    //         }
    //     }
    // printf("!!%d\n",__LINE__);

        // for(int idx = 0;idx < 0; idx++){

    //     for(int idx = 1;idx < baby_step; idx++){
    //         if(idx > 17 && idx % 32 != 0){
    //             continue;
    //         }
    //         elt = modpow128(3,idx,(2*N));
    //         Check(mempool((void**)&galois_b_baby,2 * N * size * sizeof(unsigned long long)));
    //         Check(mempool((void**)&galois_a_baby,2 * N * size * sizeof(unsigned long long)));
    //         genRandom<<<N/1024,1024>>>(galois_b_baby,scale);
    //         galois_b_baby = rns.decomposeLongBasis(galois_b_baby,N);
    //         for(int i = 0; i < 2 * size; i++){
    //             forwardNTT(galois_b_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         }
    //         unsigned long long* rots;
    //         Check(mempool((void**)&rots, 2 * N * size * sizeof(unsigned long long)));

    //         cudaMemcpy(rottmp,pri_b_l,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

    //         for(int i = 0; i < 2 * size; i++){
    //             inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
    //         }

    //         for(int i = 0; i < 2 * size; i++){
    //             cudarotation<<<N/1024,1024>>>(rottmp + N * i,rots + N * i,q[i],elt,N);
    //         }

    //         for(int i = 0; i < 2 * size; i++){
    //             forwardNTT(rots + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
    //         }


    //         for(int i = 0; i < 2 * size; i++){            
    //             polymul<<<N/1024,1024>>>(pri_b_l+ N * i,galois_b_baby + N * i,galois_a_baby + N * i,q[i],mu[i],q_bit[i]);
    //         }
    //         unsigned long long *galois_a_baby_comp,*galois_b_baby_comp;

    //         Check(mempool((void**)&galois_a_baby_comp, size * 2 * N * size * sizeof(unsigned long long)));
    //         Check(mempool((void**)&galois_b_baby_comp, size * 2 * N * size * sizeof(unsigned long long)));
    //         for(int i = 0; i < size; i++){
    //             cudaMemcpy(galois_a_baby_comp + i * 2 * N * size ,galois_a_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //             cudaMemcpy(galois_b_baby_comp + i * 2 * N * size ,galois_b_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         }
    //         unsigned long long* rots_dcomp;
    //         Check(mempool((void**)&rots_dcomp, 2 * N * size * sizeof(unsigned long long)));

    //         cudaMemcpy(rots_dcomp ,rots, N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
    //         for(int i = 0; i <  size; i++){
    //             polymulInteger<<<N/1024,1024>>>(rots_dcomp + N * i,q[size],q[i],mu[i],q_bit[i]);
    //         }
    //         for(int i = 0; i < size; i++){
    //             // if(idx == 2){
    //             //     babycomp[idx*size + i].set(galois_a_baby_comp,galois_b_baby_comp);
    //             //     continue;
    //             // }
                
    //             for(int j = 0; j < size; j++){
    //                 if(i == j)polyadd<<<N/1024,1024>>>(rots_dcomp + N * i,galois_a_baby_comp + 2 * N * size * i + j * N,galois_a_baby_comp + 2 * N * size * i + j * N,N,q[j]);
    //                 else polyadd<<<N/1024,1024>>>(0,galois_a_baby_comp + 2 * N * size * i + j * N,galois_a_baby_comp + 2 * N * size * i + j * N,N,q[j]);
    //             }
    //             babycompr[idx*size + i].set(galois_a_baby_comp,galois_b_baby_comp);
    //         }
    //     }
    // printf("!!%d\n",__LINE__);

//===========================
#else
        //     printf("!!%d\n",__LINE__);

        // for(int idx = 0;idx < 2; idx++){
        //     elt = modpow128(3,n-idx,(2*N));
        //     Check(mempool((void**)&galois_b_baby,2 * N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_baby,2 * N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_baby,scale);
        //     galois_b_baby = rns.decomposeLongBasis(galois_b_baby,N);
        //     for(int i = 0; i < 2 * size; i++){
        //         forwardNTT(galois_b_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }

        //     for(int i = 0; i < size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a_baby + N * i,q[i],elt,N);
        //     }

        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_a_baby + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         polymulInteger<<<N/1024,1024>>>(galois_a_baby + N * i,q[size],q[i],mu[i],q_bit[i]);
        //     }

        //     for(int j = 0; j < size; j++){
        //         unsigned long long *galois_b_baby_comp,*galois_a_baby_comp;
        //         Check(mempool((void**)&galois_b_baby_comp,2 * N * size * sizeof(unsigned long long)));
        //         Check(mempool((void**)&galois_a_baby_comp,2 * N * size * sizeof(unsigned long long)));
        //         cudaMemcpy(galois_a_baby_comp,galois_a_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //         cudaMemcpy(galois_b_baby_comp,galois_b_baby,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        //         for(int i = 0; i < 2 * size; i++){
        //             polymuladd<<<N/1024,1024>>>(galois_b_baby + N * i,pri_b_l + N * j,galois_a_baby + N * i,galois_a_baby + N * i,q[i],mu[i],q_bit[i]);
        //         }
        //         babycomp[idx*size + j].set(galois_a_baby,galois_b_baby);
        //     }
        // }
        //             printf("!!%d\n",__LINE__);

#endif

#ifdef ROT
        // for(int idx = 0;idx < n/baby_step; idx++){
        //     elt = modpow128(3,n-idx*baby_step,(2*N));
        //     Check(mempool((void**)&galois_b_gaint, N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_gaint, N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_gaint,scale);
        //     galois_b_gaint = rns.decompose(galois_b_gaint,N);
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_b_gaint + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a_gaint + N * i,q[i],elt,N);
        //     }
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_a_gaint + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         polymulminus<<<N/1024,1024>>>(galois_b_gaint + N * i,pri_b + N * i,galois_a_gaint + N * i,galois_a_gaint + N * i,q[i],mu[i],q_bit[i]);
        //     }
        //     gaint[idx].set(galois_a_gaint,galois_b_gaint);
        // }
        // for(int idx = 0;idx < 0; idx++){
        //     elt = modpow128(3,n-idx*baby_step,(2*N));
        //     Check(mempool((void**)&galois_b_gaint,2 * N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_gaint,2 * N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_gaint,scale);
        //     galois_b_gaint = rns.decomposeLongBasis(galois_b_gaint,N);
        //     for(int i = 0; i < 2 * size; i++){
        //         forwardNTT(galois_b_gaint + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     unsigned long long* rots;
        //     Check(mempool((void**)&rots, 2 * N * size * sizeof(unsigned long long)));

        //     cudaMemcpy(rottmp,pri_b_l,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        //     for(int i = 0; i < 2 * size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }

        //     for(int i = 0; i < 2 * size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,rots + N * i,q[i],elt,N);
        //     }

        //     for(int i = 0; i < 2 * size; i++){
        //         forwardNTT(rots + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }


        //     for(int i = 0; i < 2 * size; i++){            
        //         polymul<<<N/1024,1024>>>(pri_b_l+ N * i,galois_b_gaint + N * i,galois_a_gaint + N * i,q[i],mu[i],q_bit[i]);
        //     }
        //     unsigned long long *galois_a_gaint_comp,*galois_b_gaint_comp;

        //     Check(mempool((void**)&galois_a_gaint_comp, size * 2 * N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_b_gaint_comp, size * 2 * N * size * sizeof(unsigned long long)));
        //     for(int i = 0; i < size; i++){
        //         cudaMemcpy(galois_a_gaint_comp + i * 2 * N * size ,galois_a_gaint,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //         cudaMemcpy(galois_b_gaint_comp + i * 2 * N * size ,galois_b_gaint,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     }
        //     unsigned long long* rots_dcomp;
        //     Check(mempool((void**)&rots_dcomp, 2 * N * size * sizeof(unsigned long long)));

        //     cudaMemcpy(rots_dcomp ,rots, 2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i <  size; i++){
        //         polymulInteger<<<N/1024,1024>>>(rots_dcomp + N * i,q[size],q[i],mu[i],q_bit[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         for(int j = 0; j < size; j++){
        //             if(i == j)polyadd<<<N/1024,1024>>>(rots_dcomp + N * i,galois_a_gaint_comp + 2 * N * size * i + j * N,galois_a_gaint_comp + 2 * N * size * i + j * N,N,q[j]);
        //             else polyadd<<<N/1024,1024>>>(0,galois_a_gaint_comp + 2 * N * size * i + j * N,galois_a_gaint_comp + 2 * N * size * i + j * N,N,q[j]);
        //         }
        //         gaintcomp[idx*size + i].set(galois_a_gaint_comp,galois_b_gaint_comp);
        //     }
        // }
#endif
        // for(int idx = 0;idx <  n/baby_step; idx++){
        //     elt = modpow128(3,n-idx*baby_step,(2*N));
        //     Check(mempool((void**)&galois_b_gaint, N * size * sizeof(unsigned long long)));
        //     Check(mempool((void**)&galois_a_gaint, N * size * sizeof(unsigned long long)));
        //     genRandom<<<N/1024,1024>>>(galois_b_gaint,scale);
        //     galois_b_gaint = rns.decompose(galois_b_gaint,N);
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_b_gaint + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     cudaMemcpy(rottmp,pri_b,N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //     for(int i = 0; i < size; i++){
        //         inverseNTT(rottmp + N * i,N,ntt,q[i],mu[i],q_bit[i],psiinvTable[i]);
        //     }
        //     for(int i = 0; i < size; i++){
        //         cudarotation<<<N/1024,1024>>>(rottmp + N * i,galois_a_gaint + N * i,q[i],elt,N);
        //     }
        //     for(int i = 0; i < size; i++){
        //         forwardNTT(galois_a_gaint + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        //     }
        //     for(int j = 0; j < size; j++){
        //         unsigned long long *galois_b_gaint_comp,*galois_a_gaint_comp;
        //         Check(mempool((void**)&galois_b_gaint_comp,2 * N * size * sizeof(unsigned long long)));
        //         Check(mempool((void**)&galois_a_gaint_comp,2 * N * size * sizeof(unsigned long long)));
        //         cudaMemcpy(galois_a_gaint_comp,galois_a_gaint,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);
        //         cudaMemcpy(galois_b_gaint_comp,galois_b_gaint,2 * N * size * sizeof(unsigned long long),cudaMemcpyDeviceToDevice);

        //         for(int i = 0; i < 2 * size; i++){
        //             polymulminus<<<N/1024,1024>>>(galois_b_gaint + N * i,pri_b + N * j,galois_a_gaint + N * i,galois_a_gaint + N * i,q[i],mu[i],q_bit[i]);
        //         }
        //         gaintcomp[idx*size + j].set(galois_a_gaint,galois_b_gaint);
        //     }
        // }
    };

};

class Encryptor{
    public:
        // Encoder encode;
        int N;
        double scale;
        unsigned long long** psiTable;
        unsigned long long** psiinvTable; 
        unsigned long long* psi;
        unsigned long long* psiinv;
        unsigned long long* q;
        unsigned long long* mu;
        // unsigned long long ninv;
        unsigned long long* q_bit;
        unsigned long long* Qmod;
        unsigned long long* q_hatinv;
        unsigned long long* Pmod;
        unsigned long long* p_hatinv;
        unsigned long long* Pinv;
        int size;
        cudaStream_t ntt;

        unsigned long long *u,*e;
        RNS rns;
        Encryptor(int n,double scale,int size):rns(size,scale){
            N = 2 * n;
            this->scale = scale;
            this->size = size;
            q = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
            psi = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
            psiinv = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
            q_bit = (unsigned long long *)malloc(2 * size * sizeof(unsigned long long));
            Qmod = (unsigned long long *)malloc(size * size * sizeof(unsigned long long));
            q_hatinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));

            Pmod = (unsigned long long*)malloc(size * size * sizeof(unsigned long long *));
            p_hatinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
            Pinv = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long *));
             
            getParams(q, psi, psiinv, q_bit, Qmod,q_hatinv,Pmod,p_hatinv,Pinv,size);
            
            mu = (unsigned long long*)malloc(2 * size * sizeof(unsigned long long));
            for(int i = 0; i < 2 * size; i++){
                uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
                mu[i] = (mu1 / q[i]).low;
            }

            psiTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));
            psiinvTable = (unsigned long long**)malloc(2 * size * sizeof(unsigned long long *));

            for(int i = 0; i < 2 * size; i++){
                Check(mempool(&psiTable[i], N * sizeof(unsigned long long)));
                Check(mempool(&psiinvTable[i], N * sizeof(unsigned long long)));
            }

            for(int i = 0; i < 2 * size; i++){
                fillTablePsi128<<<N/1024,1024>>>(psi[i], q[i], psiinv[i], psiTable[i], psiinvTable[i], log2(N));
                uint128_t mu1 = uint128_t::exp2(q_bit[i] * 2);
                mu[i] = (mu1 / q[i]).low;
            }
            ntt = 0;

            Check(mempool((void**)&u, N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&e, N * size * sizeof(unsigned long long)));
            
        };
        cipherText encrypt(unsigned long long* plaintext,publicKey key){
            unsigned long long *a,*b;

            Check(mempool((void**)&b, N * size * sizeof(unsigned long long)));
            Check(mempool((void**)&a, N * size * sizeof(unsigned long long)));

            
            genRandom_u<<<N/1024,1024>>>(u,scale);
            u = rns.decompose(u,N);
            for(int i = 0; i < size; i++){
                forwardNTT(u + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            
            // print<<<1,1>>>(u);
            genRandom<<<size * N/1024,1024>>>(e,0);
            e = rns.decompose(e,N);
            // print<<<1,1>>>(e);
            for(int i = 0; i < size; i++){
                forwardNTT(e + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
            }
            
            // print<<<1,1>>>(key.b);
            // print<<<1,1>>>(u);
            for(int i = 0; i < size; i++){            
                polymul<<<N/1024,1024>>>(key.a + N * i,u + N * i,a + N * i,q[i],mu[i],q_bit[i]);
                polymul<<<N/1024,1024>>>(key.b + N * i,u + N * i,b + N * i,q[i],mu[i],q_bit[i]);
                polyadd<<<N/1024,1024>>>(a + N * i,plaintext + N * i,a + N * i,N,q[i]);
                polyadd<<<N/1024,1024>>>(a + N * i,e + N * i,a + N * i,N,q[i]);
            }
            cipherText res;
            res.set(a,b);
            // print<<<1,1>>>(key.a);
            // print<<<1,1>>>(a);
            // // print<<<1,1>>>(plaintext);
            // print<<<1,1>>>(b);
            // cudaDeviceSynchronize();
            // printf("======\n");
            // print<<<1,1>>>(b);
            return res;
        }
        unsigned long long* decrypt(cipherText& cipher,privateKey key){
            unsigned long long *a,*b,*s;
            unsigned long long *res;
            // unsigned long long *t;
            // Check(mempool((void**)&e, N * sizeof(unsigned long long)));
            // Check(mempool((void**)&t, N * sizeof(unsigned long long)));
            Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));
            a = cipher.a;
            b = cipher.b;
            s = key.b;
            
            for(int i = 0; i < size; i++){
                polymul<<<N/1024,1024>>>(b + N * i,s + N * i,res + N * i,q[i],mu[i],q_bit[i]);
                polyminus<<<N/1024,1024>>>(a + N * i,res + N * i,res + N * i,N,q[i]);
            }
            // print<<<1,1>>>(res);
            return res;
        }
        unsigned long long* decrypt(triplePoly& cipher,privateKey key){
            unsigned long long *a,*b,*c,*s;
            unsigned long long *res;

            a = cipher.a;
            b = cipher.b;
            c = cipher.c;
            s = key.b;
            Check(mempool((void**)&res, N * size * sizeof(unsigned long long)));
            for(int i = 0; i < size; i++){
                polymul<<<N/1024,1024>>>(c + N * i,s + N * i,res + N * i,q[i],mu[i],q_bit[i]);
                polymul<<<N/1024,1024>>>(res + N * i,s + N * i,res + N * i,q[i],mu[i],q_bit[i]);
                polymulminus<<<N/1024,1024>>>(b + N * i,s + N * i,res + N * i,res + N * i,q[i],mu[i],q_bit[i]);
                polyminus<<<N/1024,1024>>>(a + N * i,res + N * i,res + N * i,N,q[i]);   
            }
            // print<<<1,1>>>(res);
            // printf("aaa\n");
            return res;
        }

    cipherText relien(triplePoly cipher,privateKey pri){


        unsigned long long  *reliena,*relienb;
        unsigned long long  *s,*s_2;
        s = pri.b;
        unsigned long long* d2 = cipher.c;
        Check(mempool((void**)&s_2, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&reliena, N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&relienb, N * size * sizeof(unsigned long long)));
        
        genRandom<<<N/1024,1024>>>(relienb,scale);
        relienb = rns.decompose(relienb,N);
        for(int i = 0; i < size; i++){
            forwardNTT(relienb + N * i,N,ntt,q[i],mu[i],q_bit[i],psiTable[i]);
        }
        for(int i = 0; i < size; i++){
            polymul<<<N/1024,1024>>>(s + N * i,s + N * i,s_2 + N * i,q[i],mu[i],q_bit[i]);
        }
        dim3 dim(N/1024, size);
        polymuladd_new<<<dim,1024>>>(s,relienb,s_2,reliena);


        unsigned long long *a,*b;

        Check(mempool((void**)&a, 2 * N * size * sizeof(unsigned long long)));
        Check(mempool((void**)&b, 2 * N * size * sizeof(unsigned long long)));
        // for(int i = 0; i < 2 * size; i++){
        //     polymul<<<N/1024,1024>>>(d2 + N * i,key.a + N * i,a + N * i,q[i],mu[i],q_bit[i]);
        //     polymul<<<N/1024,1024>>>(d2 + N * i,key.b + N * i,b + N * i,q[i],mu[i],q_bit[i]);
        // }
        polymuldouble<<<dim,1024>>>(d2,reliena,a,d2,relienb,b);
        // print<<<1,1>>>(a);
        // printf("$$$$$$$$$$$$$$$$$$$$\n");
        // print<<<1,1>>>(d2,8);
        // print<<<1,1>>>(reliena,8);
        // print<<<1,1>>>(reliena,8);
        // print<<<1,1>>>(relienb,8);

        unsigned long long* a_down = a;//Moddown(a);
        unsigned long long* b_down = b;//Moddown(b);
        // // for(int i = 0; i < size; i++){
        // //     polyadd<<<N/1024,1024>>>(cipher.a + N * i,a_down + N * i,a_down + N * i, N,q[i]);
        // //     polyadd<<<N/1024,1024>>>(cipher.b + N * i,b_down + N * i,b_down + N * i, N,q[i]);
        // // }
        polyadddouble<<<dim,1024>>>(cipher.a,a_down,a_down,cipher.b,b_down,b_down);
        // polyaddsingle<<<dim,1024>>>(cipher.a,a_down,a_down);
        // polyaddsingle<<<dim,1024>>>(b_down,cipher.b,b_down);

        // print<<<1,1>>>(a_down,8);

        cipherText res;
        res.set(a_down,b_down);
        res.depth = cipher.depth;
        return res;
    }
};
