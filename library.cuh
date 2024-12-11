#include"cufft.h"

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
void init(int N, double scale, int SCALE);

unsigned long long* encode(double* a);

cipherText encrypt(unsigned long long* a, publicKey b);

triplePoly mulcipter(cipherText a, cipherText b);

cipherText relien_dcomp(triplePoly &ciptertextc,relienKey* reliendcomp);


void rotation_comp_table(cipherText& cipher,int step,galoisKey* galois);
 
void rotation_to_another(cipherText& cipher,cipherText& cipher2,int step,galoisKey* galois);

void rotation_to_anotherinv(cipherText& cipher,cipherText& cipher2,int step,galoisKey* galois);

void addcipher(cipherText& cipher1, cipherText cipher2);

void rescale(cipherText& ciptertextd);

unsigned long long* decrypt(cipherText ciptertextd,privateKey pri);

cuDoubleComplex* decode(unsigned long long* dec,int depth);
publicKey getpub();

privateKey getpri();

relienKey* getrelien();

galoisKey* getgalois();

galoisKey* getgaloisinv();
void mulPlain(cipherText& cipter, unsigned long long* plain);

void copyCipher(cipherText& cipher2, cipherText& cipher1,int dataSize){
    cipher2.depth = cipher1.depth;
    cudaMemcpy(cipher2.a,cipher1.a,dataSize,cudaMemcpyDeviceToDevice);
    cudaMemcpy(cipher2.b,cipher1.b,dataSize,cudaMemcpyDeviceToDevice);
}