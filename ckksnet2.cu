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
    for(int i = 0; i < N; i++)a[i] = 0;
    int cnt = 0;
    for(int i = 0; i < 32*32; i++){
        if(i%32>=28 || i/32 >= 28)continue;
        a[i] = pic[cnt++];
    }
    cnt = 0;

    for(int i = 0; i < 32*32; i++){
        if(i%32>=28 || i/32 >= 28)continue;
        a[i+1024] = pic1[cnt++];
    }

    double b[N];
    for(int i = 0; i < N; i++)b[i] = conv1_bias;
    double convker[16];
    for(int i = 0; i < 16;i++){
        convker[i]=conv1[i];
    }


    double fc1_bias[N];for(int i = 0;i < N;i++)fc1_bias[i] = 0;
    double fc2_bias[N];for(int i = 0;i < N;i++)fc2_bias[i] = 0;
    for(int i = 0;i < N;i++){
        if(i % 1024 < 64)
        fc1_bias[i] = fc1_b[i % 1024];
    }
    for(int i = 0;i < N;i++){
        if(i % 1024 < 10)
        fc2_bias[i/1024 * 1024 + (i%1024)*64] = fc2_b[i % 1024];
    }
    keyGen keygen(N,scale,8);
    // Encoder encoder(N,scale,16);
    Encryptor encryptor(N,scale,8);

    Evaluator evaluator(N,scale,8);
    auto encodeVeca = evaluator.encoder.encode(a);
    // for(int i = 0;i < 64 * 784; i++){
    //     fc1_w[i] = i;
    // }
    auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
    auto enncodefc1_w = evaluator.encoder.EncodeBigMatrix_bsgs(fc1_w,64,784,28);
    auto enncodefc1_b = evaluator.encoder.encode(fc1_bias);
    auto encodefc2_w = evaluator.encoder.EncodeBigMatrix_bsgs_test(fc2_w,10,64);

    auto enncodefc2_b = evaluator.encoder.encode(fc2_bias);


    unsigned long long* encodeVecb = evaluator.encoder.encode(b);
    auto convkerEncode = evaluator.encoder.encode(convker,16);

    double mask[N];
    for(int i = 0;i<N;i++)mask[i]=0;
    for(int i = 0;i<N;i++){
        if((i % 1024) < 64)mask[i]=1;
    }
    auto encodeMask = evaluator.encoder.encode(mask);
    cudaDeviceSynchronize();

    double start = cpuSecond();
    double start1 = cpuSecond();
    auto ciptertextr = evaluator.conv(ciptertexta,convkerEncode,32,32,4,4,keygen.babycompr,keygen.gaintcomp);
    
    
    evaluator.rescale(ciptertextr);
    printf("1...%d\n",ciptertextr.depth);
    evaluator.addPlain(ciptertextr,encodeVecb);
    printf("Time: %lf\n",cpuSecond() - start);
    start = cpuSecond();
    auto res = evaluator.mulcipter(ciptertextr,ciptertextr);
    // auto fc1_input = evaluator.relien(res,keygen.relien);
    auto fc1_input = evaluator.relien_dcomp(res,keygen.reliendcomp);

    evaluator.rescale(fc1_input);

    printf("Time: %lf\n",cpuSecond() - start);
    start = cpuSecond();

    printf("2...%d\n",fc1_input.depth);
    
    auto fc1_out = evaluator.dot_bsgs(enncodefc1_w,fc1_input,keygen.galoiscomp,keygen.galois_right,keygen.babycomp,keygen.gaintcomp,encodeMask);


    evaluator.addPlain(fc1_out,enncodefc1_b);

    printf("Time: %lf\n",cpuSecond() - start);
    start = cpuSecond();
    printf("3...%d\n",fc1_out.depth);
    res = evaluator.mulcipter(fc1_out,fc1_out);
    printf("Time: %lf\n",cpuSecond() - start);
    // auto fc2_input = evaluator.relien(res,keygen.relien);
    auto fc2_input = evaluator.relien_dcomp(res,keygen.reliendcomp);

    printf("Time: %lf\n",cpuSecond() - start);
    evaluator.rescale(fc2_input);
    printf("4...%d\n",fc2_input.depth);
    printf("Time: %lf\n",cpuSecond() - start);
    start = cpuSecond();

    auto fc2_out = evaluator.dot_bsgs_fc2(encodefc2_w,fc2_input,keygen.galoiscomp,keygen.galoiscomp_r);
    // evaluator.addPlain(fc2_out,enncodefc2_b);

    cudaDeviceSynchronize();
    printf("Time: %lf\n",cpuSecond() - start);

    printf("6...%d\n",fc2_out.depth);
     printf("Time: %lf\n",cpuSecond() - start);
     printf("Time: %lf\n",cpuSecond() - start1);

    unsigned long long* dec = encryptor.decrypt(fc2_out,keygen.pri);

    auto plaina = evaluator.encoder.decode(dec,fc2_out.depth);
    
    
    for(int i = 0; i < 10; i++){
        printf("%d,%lf\t",i,plaina[i*64].x /scale/scale);
    }



    cudaDeviceSynchronize();
}
