#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"
int main(int argc,char *argv[]){
    printf("===\n");
    int N = 2048*2;
    double scale = 1llu << 30;
    // scale = 10000.0;
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



    double fc1_bias[N];for(int i = 0;i < N;i++)fc1_bias[i] = 0;
    double fc2_bias[N];for(int i = 0;i < N;i++)fc2_bias[i] = 0;
    for(int i = 0;i < 64;i++)fc1_bias[i] = fc1_b[i];
    for(int i = 0;i < 10;i++)fc2_bias[i*64] = fc2_b[i];
    keyGen keygen(N,scale,8);
    // Encoder encoder(N,scale,8);
    Encryptor encryptor(N,scale,8);

    Evaluator evaluator(N,scale,8);

    cipherText image[1024];
    auto enncodefc1_b = evaluator.encoder.encode(fc1_bias);
    auto enncodefc2_b = evaluator.encoder.encode(fc2_bias);
    printf("start\n");
    cuDoubleComplex *fft_in;
    cuDoubleComplex *fft_out;
    Check(mempool((void**)&fft_in,2 * N * sizeof(cuDoubleComplex)));
    Check(mempool((void**)&fft_out,2 * N * sizeof(cuDoubleComplex)));
    int cnt = 0;
    for(int i = 0; i < 32*32; i++){
        // if(i >= 10){
        //     image[i] = image[0];
        //     continue;

        // }
        if(i%32>=28 || i/32 >= 28){
            a[0] = 0;
            image[i].a = 0;
            continue;
        }
        else a[0] = pic[cnt++];
        // a[i] = i;
        // printf("%d,%lf\n",cnt-1,a[0]);
        auto encodeVeca = evaluator.encoder.encode_buff(a,fft_in,fft_out);
        auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
        image[i] = ciptertexta;
    }
    unsigned long long* buffer1,*buffer2,*buffer3,*buffer4,*buffer5;
    Check(mempool((void**)&buffer1,2 * N * (8 + 1) * 8 * sizeof(unsigned long long)));
    Check(mempool((void**)&buffer2,2 * N * 16 * sizeof(unsigned long long)));
    Check(mempool((void**)&buffer3,2 * N * 16 * sizeof(unsigned long long)));
    Check(mempool((void**)&buffer4,2 * N * 16 * sizeof(unsigned long long)));
    Check(mempool((void**)&buffer5,2 * N * 16 * sizeof(unsigned long long)));
    unsigned long long* buff1,*buff2,*buff3,*buff4,*buff5;
    Check(mempool((void**)&buff1,2 * N *  8 * sizeof(unsigned long long)));
    Check(mempool((void**)&buff2,2 * N *  8 * sizeof(unsigned long long)));
    Check(mempool((void**)&buff3,2 * N *  8 * sizeof(unsigned long long)));
    Check(mempool((void**)&buff4,2 * N *  8 * sizeof(unsigned long long)));
    Check(mempool((void**)&buff5,2 * N *  8 * sizeof(unsigned long long)));
    double start = cpuSecond();
    printf("start\n");
    cudaDeviceSynchronize();
    printf("%d\n",__LINE__);
    auto convres = evaluator.convbatch(32,32,4,4,image,convker);

    for(int i = 0; i < 784; i++){
        evaluator.rescale_buff(convres[i],buff4,buff5);
    }
    double c1bias[784];
    // double scalelevel = 10000.0*scale;


    for(int i = 0; i < N; i++)c1bias[i] = conv1_bias;
    evaluator.addbatch(784,convres,scale,c1bias);
    
    // printf("%p,%p\n",start,point);
    // printf("%d\n",__LINE__);


    for(int i = 0; i < 784; i++){

        // cudaStreamCreate(&stream[i]); 
        auto res = evaluator.mulcipter(convres[i],convres[i],buff1,buff2,buff3);

        convres[i] = evaluator.relien_dcomp_fusion_buff(res,keygen.reliendcomp,buffer1,buffer2,buffer3,buffer4,buffer5);
        evaluator.rescale_buff(convres[i],buff4,buff5);
        
    }

    // printf("%d\n",__LINE__);

    // printf("%d\n",__LINE__);

    // scalelevel = scalelevel * scalelevel / scale2;

    auto layer1 = evaluator.dotbatch(64,784,convres,fc1_w);

    for(int i = 0; i < 64; i++){
        evaluator.rescale_buff(layer1[i],buff4,buff5);
    }
    // scalelevel = scalelevel * scale;

    evaluator.addbatch(64,layer1,scale,fc1_b);
    for(int i = 0; i < 64; i++){
        // evaluator.rescale(layer1[i]);
        auto res = evaluator.mulcipter(layer1[i],layer1[i],buff1,buff2,buff3);
        layer1[i] = evaluator.relien_dcomp_fusion_buff(res,keygen.reliendcomp,buffer1,buffer2,buffer3,buffer4,buffer5);
        evaluator.rescale_buff(layer1[i],buff4,buff5);
    }
    // scalelevel = (scalelevel/scale2) * (scalelevel/scale2);
    // scalelevel = 10000.0 * 10000.0 * 10000.0 * 10000.0 * 10000.0 * 10000.0 / scale;
    // printf("%d\n",__LINE__);

    auto layer2 = evaluator.dotbatch(10,64,layer1,fc2_w);


    for(int i = 0; i < 10; i++){
        evaluator.rescale(layer2[i]);
    }
    // scalelevel *= 10000.0;

    evaluator.addbatch(10,layer2,scale,fc2_b);
    printf("Time: %lf\n",cpuSecond() - start);
    // printf("%d\n",__LINE__);
    // printf("scalelevel: %lf\n",scalelevel);
    for(int k = 0; k < 10; k++){
        unsigned long long* dec = encryptor.decrypt(layer2[k],keygen.pri);

        auto plaina = evaluator.encoder.decode(dec,6);

        for(int i = 0; i < 1; i++){
            printf("%d,%lf\n",k,plaina[i].x /scale);

        }

    }

}
