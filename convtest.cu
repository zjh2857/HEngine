#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"

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

int main(int argc,char *argv[]){
    printf("===\n");
    const int N = 2048*2;
    int len = atoi(argv[1]);
    
    double scale = 1llu << 50;
    // scale = 1073872897;

    keyGen keygen(N,scale,4);
    Encoder encoder(N,scale,4);
    Encryptor encryptor(N,scale,4);
    Evaluator evaluator(N,scale,4);

    int imagesize = 32;
    double convker[N];
    for(int i = 0; i < 32;i++)convker[i] = 1;
    double a[N];
    for(int i = 0; i < N; i++)a[i] = 0;
    int cnt = 0;
    for(int i = 0; i < 32*32; i++){
        if(i%32>=imagesize || i/32 >= imagesize)continue;
        a[i] = cnt++;
    }
    
    cnt = 13;

    // for(int i = 0; i < 32*32; i++){
    //     if(i%32>=imagesize || i/32 >= imagesize)continue;
    //     a[i+1024] = cnt++;
    // }
    // cnt = 52;

    // for(int i = 0; i < 32*32; i++){
    //     if(i%32>=imagesize || i/32 >= imagesize)continue;
    //     a[i+2048] = cnt++;
    // }
    // cnt = 22;

    // for(int i = 0; i < 32*32; i++){
    //     if(i%32>=imagesize || i/32 >= imagesize)continue;
    //     a[i+3072] = cnt++;
    // }
    
    auto convkerEncode = encoder.encode(convker,len*len);

    auto encodeVeca = encoder.encode(a);
    auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
    
    cudaDeviceSynchronize();
    double start = cpuSecond();
    startTiming();
    auto ciptertextr = evaluator.conv(ciptertexta,convkerEncode,32,32,len,len,keygen.babycompr,keygen.gaintcomp);
    double endtime = stopTiming();
    printf("Event Time: %lf\n",endtime * 1000 * 1000);
    // cudaDeviceSynchronize();
    printf("Time: %lf\n",(cpuSecond() - start)*1000);
    evaluator.rescale(ciptertextr);
    
    unsigned long long* dec = encryptor.decrypt(ciptertextr,keygen.pri);

    auto plaina = encoder.decode(dec,2);

    for(int i = 0; i < 64;i++){
        printf("%d,%lf\t",i,plaina[i].x /scale);
    }
}