#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"
int main(int argc,char *argv[]){
    printf("===\n");
    int N = 2048;
    double scale = 1llu << 30;
    scale = 10000.0;
    double scale2 = 1073872897;
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



    double fc1_bias[N];for(int i = 0;i < N;i++)fc1_bias[i] = 0;
    double fc2_bias[N];for(int i = 0;i < N;i++)fc2_bias[i] = 0;
    for(int i = 0;i < 64;i++)fc1_bias[i] = fc1_b[i];
    for(int i = 0;i < 10;i++)fc2_bias[i*64] = fc2_b[i];
    keyGen keygen(N,scale,4);
    // Encoder encoder(N,scale,8);
    Encryptor encryptor(N,scale,4);

    Evaluator evaluator(N,scale,4);

    cipherText image[1024];
    auto enncodefc1_b = evaluator.encoder.encode(fc1_bias);
    auto enncodefc2_b = evaluator.encoder.encode(fc2_bias);
    printf("start\n");

    int cnt = 0;
    for(int i = 0; i < 32*32; i++){
        if(i%32>=28 || i/32 >= 28){ a[0] = 0;}
        else a[0] = pic[cnt++];
        // printf("%d,%lf\n",cnt-1,a[0]);
        auto encodeVeca = evaluator.encoder.encode(a);
        auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
        image[i] = ciptertexta;
    }
    double start = cpuSecond();
    printf("start\n");
    cudaDeviceSynchronize();
    printf("%d\n",__LINE__);
    auto convres = evaluator.convbatch(32,32,4,4,image,convker);


    double c1bias[784];
    double scalelevel = 10000.0*10000.0;
    // double scalelevel2 = 10000.0*10000.0;
    for(int i = 0; i < N; i++)c1bias[i] = conv1_bias;
    evaluator.addbatch(784,convres,scalelevel,c1bias);
    
    printf("%p,%p\n",start,point);
    printf("%d\n",__LINE__);

    for(int i = 0; i < 784; i++){
        
        evaluator.rescale(convres[i]);
        // cudaStreamCreate(&stream[i]); 
        auto res = evaluator.mulcipter(convres[i],convres[i]);

        convres[i] = evaluator.relien_dcomp(res,keygen.reliendcomp);
        
    }

    printf("%d\n",__LINE__);

    scalelevel = scalelevel * scalelevel / scale2/scale2;
    // auto layer1 = evaluator.dotbatch(64,784,convres,fc1_w);

    // scalelevel = scalelevel*scale;

    // evaluator.addbatch(64,layer1,scalelevel,fc1_b);
    // for(int i = 0; i < 64; i++){
    //     auto res = evaluator.mulcipter(layer1[i],layer1[i]);
    //     layer1[i] = evaluator.relien_dcomp(res,keygen.reliendcomp);
    //     // evaluator.rescale(layer1[i]);
    //     // evaluator.rescale(layer1[i]);
    // }
    // scalelevel = scalelevel * scalelevel ;
    // printf("%d\n",__LINE__);

    // auto layer2 = evaluator.dotbatch(10,64,layer1,fc2_w);
    // scalelevel *= 10000.0;

    // evaluator.addbatch(10,layer2,scalelevel,fc2_b);
    // printf("Time: %lf\n",cpuSecond() - start);
    // printf("%d\n",__LINE__);
    printf("scalelevel: %lf\n",scalelevel);
    for(int k = 0; k < 10; k++){
        // if(image[k].a == 0)continue;
        unsigned long long* dec = encryptor.decrypt(convres[k],keygen.pri);

        auto plaina = evaluator.encoder.decode(dec,2);

        for(int i = 0; i < 1; i++){
            printf("%d,%lf\n",k,plaina[i].x /scalelevel/scale);

        }

    }

}
