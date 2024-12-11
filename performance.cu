#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "freshman.h"
const int testTimes = 2;

cudaEvent_t st, stop;

void startTiming() {
    cudaEventCreate(&st);
    cudaEventCreate(&stop);
    cudaEventRecord(st);
}

double stopTiming() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st, stop);
    cudaEventDestroy(st);
    cudaEventDestroy(stop);
    return (double)milliseconds / 1000;
}

int main(int argc,char** argv){
    printf("===\n");
    int N = 1 << int(atoi(argv[1]));
    int len = N >> 11;
    N /= 2;
    
    if(N == 8192 * 4){
        len = 42;
    }
    if(N == 8192 * 2){
        len = 20;
    }
    if(N == 8192 ){
        len = 8;
    }
    if(N == 4096 ){
        len = 3;
    }
    unsigned long long *p;
    cudaDeviceSynchronize();

    double scale = 1llu << 30;
    double* a = (double*)malloc(sizeof(double) * N);
    for(int i = 0; i < N; i++){
        a[i] = i ;
    }
    double* b = (double*)malloc(sizeof(double) * N);
    for(int i = 0; i < N; i++){
        b[i] = i;
    }

    keyGen keygen(N,scale,len);
    printf("!!%d\n",__LINE__);

    Encryptor encryptor(N,scale,len);
        printf("!!%d\n",__LINE__);

    Evaluator evaluator(N,scale,len);

    auto encodeVeca = evaluator.encoder.encode(a);
    auto encodeVecb = evaluator.encoder.encode(b);
    auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
    auto ciptertextb = encryptor.encrypt(encodeVecb,keygen.pub);
    auto ciptertextmul = evaluator.mulcipter(ciptertexta,ciptertextb);

    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.addcipter(ciptertexta,ciptertextb);
    }
    double encryptoTimes = stopTiming();
    printf("add Times:%lf microseconds\n", encryptoTimes*1000*1000/1);
    
    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.mulPlain_lazy(ciptertexta,encodeVecb);
    }
    encryptoTimes = stopTiming();
    printf("mul plain Times:%lf microseconds\n", encryptoTimes*1000*1000/1);

    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.mulcipter(ciptertexta,ciptertextb);
    }
    double mulTimes = stopTiming();
    printf("mul Times:%lf microseconds\n", mulTimes*1000*1000/1);

    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.relien_dcomp(ciptertextmul,keygen.reliendcomp);
    }
    double relienTimes = stopTiming();
    printf("relien Times:%lf microseconds\n", relienTimes*1000*1000/1);

    printf("HMUL Times:%lf microseconds\n", (relienTimes+mulTimes)*1000*1000/1);

    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.rotation_comp_table(ciptertexta,1,keygen.galoiscomp);
    }
    encryptoTimes = stopTiming();
    printf("rotation Times:%lf microseconds\n", encryptoTimes*1000*1000/1);

    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.rescale(ciptertextb);
    }
    encryptoTimes = stopTiming();
    printf("rescale Times:%lf microseconds\n", encryptoTimes*1000*1000);

    printf("\n\n\n\n\n\n\n");

    startTiming();
    for(int i = 0; i < testTimes; i++){
        evaluator.addcipter(ciptertexta,ciptertextb);
    }
    encryptoTimes = stopTiming();
    printf("add Times:%lf microseconds\n", encryptoTimes*1000*1000/testTimes);

    startTiming();
    for(int i = 0; i < testTimes; i++){
        evaluator.mulcipter(ciptertexta,ciptertextb);
    }
    mulTimes = stopTiming();
    // printf("mul Times:%lf microseconds\n", mulTimes*1000*1000/testTimes);

    startTiming();
    for(int i = 0; i < testTimes; i++){
        evaluator.relien_dcomp(ciptertextmul,keygen.reliendcomp);
    }
    relienTimes = stopTiming();
    // printf("relien Times:%lf microseconds\n", relienTimes*1000*1000/testTimes);


    printf("HMUL Times:%lf microseconds\n", (relienTimes+mulTimes)*1000*1000/testTimes);

    startTiming();
    for(int i = 0; i < testTimes; i++){
        evaluator.rotation_comp_table(ciptertexta,1,keygen.babycomp,keygen.gaintcomp);
    }
    encryptoTimes = stopTiming();
    printf("rotation Times:%lf microseconds\n", encryptoTimes*1000*1000/testTimes);

    startTiming();
    for(int i = 0; i < 1; i++){
        evaluator.rescale(ciptertexta);
    }
    encryptoTimes = stopTiming();
    printf("rescale Times:%lf microseconds\n", encryptoTimes*1000*1000);

}
