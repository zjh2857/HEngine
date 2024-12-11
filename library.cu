#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
int rotcnt = 0;
keyGen* keygen;
Encoder* encoder;
Encryptor* encryptor;
Evaluator* evaluator;

void init(int N, double scale, int SCALE){
    keygen = new keyGen(N,scale,SCALE);
    encoder = new Encoder(N,scale,SCALE);
    encryptor = new Encryptor(N,scale,SCALE);
    evaluator = new Evaluator(N,scale,SCALE);
}

unsigned long long* encode(double* a){
    return encoder->encode(a);
}

cipherText encrypt(unsigned long long* a, publicKey b){
    return encryptor->encrypt(a,b);
}
triplePoly mulcipter(cipherText a, cipherText b){
    return evaluator->mulcipter(a,b);
}

cipherText relien_dcomp(triplePoly &ciptertextc,relienKey* reliendcomp){
    return evaluator->relien_dcomp(ciptertextc,reliendcomp); 
}

void rescale(cipherText& ciptertextd){
    evaluator->rescale(ciptertextd);
}

unsigned long long* decrypt(cipherText ciptertextd,privateKey pri){
    return encryptor->decrypt(ciptertextd,pri);
}

cuDoubleComplex* decode(unsigned long long* dec,int depth){
    return encoder->decode(dec,depth);
}
privateKey getpri(){
    return keygen->pri;
}
relienKey* getrelien(){
    return keygen->reliendcomp;
}
relienKey getrelien_simple(){
    return keygen->relien;
}
publicKey getpub(){
    return keygen->pub;
}

galoisKey* getgalois(){
    return keygen->galoiscomp;
}
galoisKey* getgaloisinv(){
    return keygen->galoiscomp_r;
}
void addPlain(cipherText& cipter, unsigned long long* plain){
    evaluator->addPlain(cipter,plain); 
}

void mulPlain(cipherText& cipter, unsigned long long* plain){
    evaluator->mulPlain(cipter,plain); 
}
void mulPlain_new1_lazy(cipherText& ciphertmp,cipherText& cipher, unsigned long long* plain){
    evaluator->mulPlain_new1_lazy(ciphertmp,cipher, plain);
}



void addcipher(cipherText& cipher1, cipherText cipher2){
    evaluator->addcipter(cipher1,cipher2);
} 

void rotation_comp_table(cipherText& cipher,int step,galoisKey* galois){
        if(step != 0){
        rotcnt++;
        printf("%d\n",rotcnt);
    }
    evaluator->rotation_comp_table(cipher,step,galois);
}

void rotation_to_another(cipherText& cipher,cipherText& cipher2,int step,galoisKey* galois){
    if(step != 0){
        rotcnt++;
        printf("%d\n",rotcnt);
    }
    evaluator->rotation_comp_table(cipher,cipher2,step,galois);
}

void rotation_to_anotherinv(cipherText& cipher,cipherText& cipher2,int step,galoisKey* galois){
        if(step != 0){
        rotcnt++;
        printf("%d\n",rotcnt);
    }
    evaluator->rotation_comp_tableinv(cipher,cipher2,step,galois);
}