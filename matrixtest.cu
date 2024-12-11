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
    int N = 1 << 15;
    int len = atoi(argv[1]);
    int len2 = atoi(argv[2]);
    printf("%d,%d\n",len,len2);
    double scale = 1llu << 50;
    // scale = 1073872897;

    // double a[512*512];
    printf("%d\n",__LINE__);

    double* a = (double*)malloc(1024*128*sizeof(double));
    double b[N];
    printf("%p\n",a);
    for(int i = 0; i < 1024*128; i++)a[i] = 0;
    printf("%d\n",__LINE__);

    for(int i = 0; i < N; i++)b[i] = 0;
    for(int i = 0; i < len*len2; i++){
        a[i] = i/len;
    }
    for(int i = 0; i < N; i++){
        b[i] = 1;   
        if(i < len)b[i] = 1;
        else if (i < len*2)b[i] = 2;
        else if (i < len*3)b[i] = 2;
        // else b[i] = 2; 
    }
    printf("%d\n",__LINE__);

    keyGen keygen(N,scale,6);
    printf("%d\n",__LINE__);

    Encoder encoder(N,scale,6);
    printf("%d\n",__LINE__);

    Encryptor encryptor(N,scale,6);
    printf("%d\n",__LINE__);

    Evaluator evaluator(N,scale,6);
    printf("%d\n",__LINE__);
    if(false){
        auto encodefc2_w = encoder.EncodeMatrix(a,1,2048);
        auto b_encode = encoder.encode(b);
        auto ciptertextb = encryptor.encrypt(b_encode,keygen.pub);
        double start = cpuSecond();
        auto fc2_out = evaluator.dot(encodefc2_w,ciptertextb,keygen.galois,keygen.galois_right);
        // cudaDeviceSynchronize();
        printf("Time: %lf\n",(cpuSecond() - start)*1000);
    } else {
        auto encodefc2_w = encoder.EncodeBigMatrix_bsgs_test(a,len2,len);
        auto b_encode = encoder.encode(b);
        auto ciptertextb = encryptor.encrypt(b_encode,keygen.pub);

        cudaDeviceSynchronize();
        
        double start = cpuSecond();
        startTiming();
        auto fc2_out = evaluator.dot_bsgs_test(encodefc2_w,ciptertextb,keygen.galoiscomp,keygen.galoiscomp_r,keygen.babycomp,keygen.gaintcomp,b_encode);
        double endtime = stopTiming();
        printf("Event Time: %lf\n",endtime * 1000 * 1000);        
        // evaluator.rescale(fc2_out);



        printf("Time: %lf\n",(cpuSecond() - start)*1000*1000);
        cudaDeviceSynchronize();


        unsigned long long* dec = encryptor.decrypt(fc2_out,keygen.pri);
        auto plaina = evaluator.encoder.decode(dec,2);
        for(int i = 0; i < len2; i++){
            printf("%d,%lf\t",i,plaina[i].x /scale);
        }
        for(int i = 0; i < len2; i++){
            printf("%d,%lf\t",i,plaina[i+len*2].x /scale);
        }
        
    }
}
/*
4096
4 * 4 9.760857
8 * 8 9.696960
16 * 16 9.741783
32 * 32  9.783030
64 * 64 47.437906
128 * 128 91.391087
64 * 1024 49.424171
128 * 1024 86.578846 
*/

/*
4    4.308939
8    5.364180
16   5.473137
32   6.052017
64   7.559061
128  9.896994
256  14.219046
512  21.286011
1024 34.579992
*/