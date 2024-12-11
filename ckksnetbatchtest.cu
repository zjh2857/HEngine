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

    keyGen keygen(N,scale,8);
    Encryptor encryptor(N,scale,8);
    Evaluator evaluator(N,scale,8);

    cipherText image[1024];
    double a[N];
    double fc1_w[16 * 4];
    for(int i = 0; i < 16; i++){
        a[0] = i;
        // printf("%d,%lf\n",cnt-1,a[0]);
        auto encodeVeca = evaluator.encoder.encode(a);
        auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
        image[i] = ciptertexta;
    }
    evaluator.dotbatch(8,8,image,fc1_w);
    evaluator.dotbatch_old(8,8,image,fc1_w);
    evaluator.encoder.encode(a);
    cudaDeviceSynchronize();

}
