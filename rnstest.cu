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

// __global__ void print(unsigned long long* a,int size){
//     for(int i = 0; i < size; i++){
//         printf("%llu\n",a[i *4096 + 5]);
//     }
// }
int main(int argc,char *argv[]){
    printf("===\n");
    int N = 2048 * 2 * 2 * 2;
    int modulen = N / 2048 * 4;
    double scale = 1llu << 30;
    // scale = 1073872897;
    double a[N];
    for(int i = 0; i < N; i++){
        a[i] = i;
        // a[i].y = 0;
    }
    // a[0] = 1;
    cuDoubleComplex b[N];
    for(int i = 0; i < N; i++){
        b[i].x = 1;
        b[i].y = 0;
    }
    b[0].x = 1;
    double d[N];
    for(int i = 0; i < N; i++){
        d[i] = -1156019777784512513;
    }

    // keyGen keygen(N,scale,modulen);

    EncoderT encoder(N,scale,modulen);

    // Encryptor encryptor(N,scale,modulen);

    // EvaluatorT evaluator(N,scale,modulen);

    auto encodeVeca = encoder.encode(a);
    auto encodeVecb = encoder.encode(b);

    auto plaina = encoder.decode(encodeVeca,0);

    for(int i = 0; i < 16 ; i++){
        printf("%lf+%lf i\n",plaina[i].x/scale,plaina[i].y/scale);
    }
}