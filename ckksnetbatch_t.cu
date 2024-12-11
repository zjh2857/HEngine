#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"
int main(int argc,char *argv[]){
    printf("===\n");
    int N = 2048;
    double scale = 1llu << 30;
    scale = 1073872897;
    double a[N];
    for(int i = 0; i < N; i++)a[i] = 0;
    for(int i = 0; i < N; i++){
        a[i] = 0;
    }
    double b[N];
    for(int i = 0; i < N; i++)b[i] = conv1_bias;
    double* convker = new double[16];
    for(int i = 0; i < 16;i++){
        convker[i]=conv1[i];
    }
    double c[N];
    for(int i = 0; i < N; i++)c[i] = -1;

    double fc1_bias[N];for(int i = 0;i < N;i++)fc1_bias[i] = 0;
    double fc2_bias[N];for(int i = 0;i < N;i++)fc2_bias[i] = 0;
    for(int i = 0;i < 64;i++)fc1_bias[i] = fc1_b[i];
    for(int i = 0;i < 10;i++)fc2_bias[i*64] = fc2_b[i];
    keyGen keygen(N,scale,8);
    Encoder encoder(N,scale,8);
    Encryptor encryptor(N,scale,8);

    Evaluator evaluator(N,scale,8);

    cipherText image[1024];
    auto enncodefc1_b = encoder.encode(fc1_bias);
    auto enncodefc2_b = encoder.encode(fc2_bias);

    auto encodeVecc = encoder.encode(c);

    int cnt = 0;
    for(int i = 0; i < 32*32; i++){
        a[0] = pic[cnt++];
        if(i%32>=28 || i/32 >= 28)a[0] = 0;
        auto encodeVeca = encoder.encode(a);
        auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
        image[i] = ciptertexta;
    }
    double start = cpuSecond();
    printf("start\n");
    auto convres = evaluator.convbatch(32,32,4,4,image,convker);



    // // auto encodeVeca = encoder.encode(a);
    // // for(int i = 0;i < 64 * 784; i++){
    // //     fc1_w[i] = i;
    // // }
    // double c1bias[784];
    // double scalelevel = 10000.0;

    // for(int i = 0; i < N; i++)c1bias[i] = conv1_bias;
    // evaluator.addbatch(784,convres,scalelevel,c1bias);
    
    // // cudaStream_t stream[784];

    // // for(int i = 0; i < 784; i++){

    // //     cudaStreamCreate(&stream[i]); 
    // //     auto res = evaluator.mulcipter(convres[i],convres[i],stream[i]);
    // //     convres[i] = evaluator.relien(res,keygen.relien,stream[i]);
    // //     evaluator.rescale(convres[i],stream[i]);
    // // }

    // for(int i = 0; i < 784; i++){

    //     // cudaStreamCreate(&stream[i]); 
    //     auto res = evaluator.mulcipter(convres[i],convres[i]);
    //     convres[i] = evaluator.relien(res,keygen.relien);
    //     evaluator.rescale(convres[i]);
    // }
    // // for(int i = 0; i < 784; i++){
    // //     cudaStreamSynchronize(stream[i]); 
    // //     cudaStreamDestroy(stream[i]);
    // // }
    // //  printf("Time: %lf\n",cpuSecond() - start);

    // scalelevel = 10000.0 * 10000.0;

    // auto layer1 = evaluator.dotbatch(64,784,convres,fc1_w);

    // scalelevel = 10000.0 * 10000.0 * 10000.0;

    // evaluator.addbatch(64,layer1,scalelevel,fc1_b);
    // for(int i = 0; i < 64; i++){
    //     auto res = evaluator.mulcipter(layer1[i],layer1[i]);
    //     layer1[i] = evaluator.relien(res,keygen.relien);
    //     evaluator.rescale(layer1[i]);
    //     evaluator.rescale(layer1[i]);
    // }
    // scalelevel = 10000.0 * 10000.0 * 10000.0 * 10000.0 * 10000.0 * 10000.0 / scale;

    // auto layer2 = evaluator.dotbatch(10,64,layer1,fc2_w);
    // scalelevel *= 10000.0;
    // printf("scalelevel: %lf\n",scalelevel);

    // evaluator.addbatch(10,layer2,scalelevel,fc2_b);
    // printf("Time: %lf\n",cpuSecond() - start);

    // unsigned long long* dec = encryptor.decrypt(layer2[0],keygen.pri);

    // auto plaina = encoder.decode(dec,3);

    // for(int i = 0; i < 8; i++){
    //     printf("%d,%lf\n",i,plaina[i].x / scale/scalelevel);
    // }

}
