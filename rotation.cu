#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"
int main(int argc,char *argv[]){
    printf("===\n");
    int N = 2048*2;
    double scale = 1llu << 30;
    scale = 1073872897;
    double a[N];
    for(int i = 0; i < N; i++)a[i] = i;
    keyGen keygen(N,scale,8);
    Encryptor encryptor(N,scale,8);
    Evaluator evaluator(N,scale,8);
    auto encodeVeca = evaluator.encoder.encode(a);

    auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);

    evaluator.rotation(ciptertexta,1,keygen.baby,keygen.gaint);
    unsigned long long* dec = encryptor.decrypt(ciptertexta,keygen.pri);

    auto plaina = evaluator.encoder.decode(dec,ciptertexta.depth+1);
    
    
    for(int i = 0; i < 4096; i++){
        // printf("%d,%lf\n",i,plaina[i].x / scale);
    }



    cudaDeviceSynchronize();
}
