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
    int cnt = 0;
    for(int i = 0; i < 32*32; i++){
        if(i%32>=28 || i/32 >= 28)continue;
        a[i] = pic[cnt++];
    }
    double b[N];
    for(int i = 0; i < N; i++)b[i] = conv1_bias;
    double convker[16];
    for(int i = 0; i < 16;i++){
        convker[i]=conv1[i];
    }
    double c[N];
    for(int i = 0; i < N; i++)c[i] = i;

    double fc1_bias[N];for(int i = 0;i < N;i++)fc1_bias[i] = 0;
    double fc2_bias[N];for(int i = 0;i < N;i++)fc2_bias[i] = 0;
    for(int i = 0;i < 64;i++)fc1_bias[i] = fc1_b[i];
    for(int i = 0;i < 10;i++)fc2_bias[i*64] = fc2_b[i];
    keyGen keygen(N,scale,8);
    Encoder encoder(N,scale,8);
    Encryptor encryptor(N,scale,8);
    Evaluator evaluator(N,scale,8);

    auto encodeVeca = encoder.encode(a);
    // for(int i = 0;i < 64 * 784; i++){
    //     fc1_w[i] = i;
    // }
    auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
    auto enncodefc1_w = encoder.EncodeBigMatrix(fc1_w,64,784,28);
    auto enncodefc1_b = encoder.encode(fc1_bias);
    auto encodefc2_w = encoder.EncodeMatrix(fc2_w,10,64);
    auto enncodefc2_b = encoder.encode(fc2_bias);

    unsigned long long* encodeVecb = encoder.encode(b);
    auto convkerEncode = encoder.encode(convker,16);
    double start = cpuSecond();
    auto ciptertextr = evaluator.conv(ciptertexta,convkerEncode,32,32,4,4,keygen.galois);
    
    evaluator.rescale(ciptertextr);
    printf("1...%d\n",ciptertextr.depth);
    evaluator.addPlain(ciptertextr,encodeVecb);
    auto res = evaluator.mulcipter(ciptertextr,ciptertextr);
    auto fc1_input = evaluator.relien(res,keygen.relien);
    evaluator.rescale(fc1_input);
    printf("Time: %lf\n",cpuSecond() - start);
    printf("2...%d\n",fc1_input.depth);
    auto fc1_out = evaluator.dot(enncodefc1_w,fc1_input,keygen.galois,keygen.galois_right,keygen.baby,keygen.gaint);

    evaluator.addPlain(fc1_out,enncodefc1_b);
    printf("Time: %lf\n",cpuSecond() - start);
    printf("3...%d\n",fc1_out.depth);
    res = evaluator.mulcipter(fc1_out,fc1_out);
    auto fc2_input = evaluator.relien(res,keygen.relien);
    evaluator.rescale(fc2_input);
    printf("4...%d\n",fc2_input.depth);
    
    auto fc2_out = evaluator.dot(encodefc2_w,fc2_input,keygen.galois,keygen.galois_right);
    evaluator.addPlain(fc2_out,enncodefc2_b);
    printf("6...%d\n",fc2_out.depth);
    unsigned long long* dec = encryptor.decrypt(fc2_out,keygen.pri);

    auto plaina = encoder.decode(dec,fc2_out.depth);
    
    
    for(int i = 0; i < 10; i++){
        printf("%d,%lf\n",i,plaina[i*64]);
    }

    cudaDeviceSynchronize();
}