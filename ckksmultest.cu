#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"

__global__ void print(unsigned long long* a,int size){
    for(int i = 0; i < size; i++){
        printf("%llu\n",a[i *4096 + 0]);
    }
    printf("=======\n");
}

int main(int argc,char *argv[]){
    printf("===\n");
    int N = 1 << (atoi(argv[1]));
    double scale = 1llu << 30;
    scale = 1073872897;
    double a[N];
    for(int i = 0; i < N; i++){
        a[i] = i;
        // a[i].y = 0;
    }
    // a[0] = 1;
    double b[N];
    for(int i = 0; i < N; i++){
        b[i] = 1;
    }

    // keyGen keygen(N,scale,8);
    // EncoderT encoder(N,scale,8);
    // Encryptor encryptor(N,scale,8);
    // EvaluatorT evaluator(N,scale,8);

    // auto encodeVeca = encoder.encode(a);
    // auto encodeVecb = encoder.encode(b);
    





    // auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
    struct aa{
        unsigned long long *a;
    };
    aa ciptertexta;
    cudaMalloc(&ciptertexta.a, N * 8 * sizeof(unsigned long long));
    double start = cpuSecond();
    for(int i = 0; i < 0; i++){
        polyaddsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,1);
        polymulsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,1);
    }
    cudaDeviceSynchronize();
    printf("1Times: %lf\n",cpuSecond() - start);

    start = cpuSecond();
    for(int i = 0; i < 1; i++){
        polyaddsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,1);
        polymulsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,1);
    }
    cudaDeviceSynchronize();
    printf("1Times: %lf\n",cpuSecond() - start);

    start = cpuSecond();
    for(int i = 0; i < 1; i++){
        // polyaddsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,8);
        polymulsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,8);
    }
    cudaDeviceSynchronize();
    printf("2Times: %lf\n",cpuSecond() - start);

    start = cpuSecond();
    polymuladdsingle<<<2*N/1024,1024>>>(ciptertexta.a,ciptertexta.a,ciptertexta.a,ciptertexta.a,8);
    cudaDeviceSynchronize();
    printf("3Times: %lf\n",cpuSecond() - start);
    // auto ciptertextb = encryptor.encrypt(encodeVecb,keygen.pub);


    // auto ciptertextc = evaluator.mulcipter(ciptertexta,ciptertextb);
    
    // auto ciptertextd = evaluator.relien(ciptertextc,keygen.relien);
    // evaluator.rescale(ciptertextd);
    // unsigned long long* dec = encryptor.decrypt(ciptertextd,keygen.pri);

    // auto plaina = encoder.decode(dec,2);

    

    // for(int i = 0; i < 16 ; i++){
    //     printf("%lf+%lf i\n",plaina[i].x/scale,plaina[i].y/scale);
    // }
}