#include "encoder.cuh"
#include"cufft.h"
#include "encryptor.cuh"
#include "evaluator.cuh"
#include "net.h"


__global__ void print(unsigned long long* a,int size){
    for(int i = 0; i < size; i++){
        printf("%llu\n",a[i *8192 + 0]);
    }
    printf("=======\n");
}

// __global__ void print(unsigned long long* a,int size){
//     for(int i = 0; i < size; i++){
//         printf("%llu\n",a[i *4096 + 5]);
//     }
// }


__global__ void print_z(unsigned long long* a,unsigned long long* b,int N){
    printf("zz\n");

    for(int j = 0; j < 6;j++){
    for(int i = 0; i < N; i++){
        if(a[j * N + i] == b[j * N + 0])printf("%d,%d,%llu   ",j,i,a[i]);
    }

    }
    printf("=======\n");
}
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
    int N = 1 << int(atoi(argv[1]));
    
    int SCALE = N >> 10;






    printf("%d\n",SCALE);
    // if(N == 8192 * 4){
    //     SCALE /= 2;
    // }
    N /= 2;
    

    if(N == 8192 * 4){
        SCALE = 42;
    }
    if(N == 8192 * 2){
        SCALE = 20;
    }
    if(N == 8192 ){
        SCALE = 8;
    }
    if(N == 4096 ){
        SCALE = 3;
    }
    if(N == 2048 ){
        SCALE = 5;
    }
    // SCALE = 8 ;
    // int N = 2048*SCALE;
    double scale = 1llu << 30;

    // scale = 1073872897;
    double a[N];
    for(int i = 0; i < N; i++){
        a[i] = i %  3;
        // a[(i-1)%N] = i%10;
        // a[i].y = 0;
    }
    a[0] = 0;
    double b[N];
    for(int i = 0; i < N; i++){
        b[i] =  2;
        // b[i].y = 0;
    }
    printf("%s,%d\n",__FILE__,__LINE__);
    // b[0] = 2;
    // double d[N];
    // for(int i = 0; i < N; i++){
    //     d[i] = -1156019777784512513;
    // }
    keyGen keygen(N,scale,SCALE);
    printf("!!%d\n",__LINE__);
    Encoder encoder(N,scale,SCALE);
    printf("!!%d\n",__LINE__);

    Encryptor encryptor(N,scale,SCALE);
    printf("!!%d\n",__LINE__);
    Evaluator evaluator(N,scale,SCALE);

    auto encodeVeca = encoder.encode(a);
    cudaDeviceSynchronize();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}
    auto encodeVecb = encoder.encode(b);


    // auto encodeVecd = encoder.encode(d);
    // print_z<<<1,1>>>(encodeVeca, encodeVecb, 2 * N);
    // print_z<<<1,1>>>(encodeVecb,2 * N);

    // auto tmp = evaluator.Modup(encodeVeca);
    // print<<<1,1>>>(tmp,16);
    // encodeVeca = evaluator.Moddown(tmp);
    // print<<<1,1>>>(encodeVeca,8);
    
    auto ciptertexta = encryptor.encrypt(encodeVeca,keygen.pub);
    // printf("depth: %d\n",ciptertexta.depth);

    // evaluator.rotation_comp(ciptertexta,1,keygengaintcomp);


    auto ciptertextb = encryptor.encrypt(encodeVecb,keygen.pub);
    // for(int i = 0; i < 1;i++){
        // evaluator.mulPlain(ciptertexta,encodeVecb);


    // }
    // evaluator.rotation_comp_table(ciptertexta,1,keygen.babycomp,keygen.gaintcomp);
for(int i = 0; i < 100; i++){
    startTiming();
    evaluator.rotation_comp_table(ciptertexta,1,keygen.galoiscomp);


    double encryptoTimes = stopTiming();
    printf("rotation Times:%lf microseconds\n", encryptoTimes*1000*1000/1);

    // auto ciptertextc = evaluator.BootStrapping(ciptertexta,keygen.galois,keygen.galois_right,keygen.baby,keygen.gaint);
    // evaluator.addPlain(ciptertextc,encodeVecd);
    // printf("%d\n",ciptertextc.depth);
    // print<<<1,1>>>(ciptertexta.a);
    // print<<<1,1>>>(encodeVecb);
}
    startTiming();

    auto ciptertextc = evaluator.mulcipter(ciptertexta,ciptertextb);
    // print<<<1,1>>>(ciptertextc.a);
    // print<<<1,1>>>(ciptertextb.a);
    // print<<<1,1>>>(ciptertextc.a,8);
    // print<<<1,1>>>(ciptertextc.c);
    // ciptertextc.a = evaluator.Moddown(evaluator.Modup(ciptertextc.a));
    // print<<<1,1>>>(ciptertextc.a,16);
    auto ciptertextd = evaluator.relien_dcomp(ciptertextc,keygen.reliendcomp);
    // auto ciptertextd = evaluator.relien(ciptertextc,keygen.relien);
    double encryptoTimes = stopTiming();
    printf("Hmul Times:%lf microseconds\n", encryptoTimes*1000*1000/1);

    // ciptertextd.depth = 2;
    // evaluator.rescale1(ciptertextd);
    startTiming();

    evaluator.rescale(ciptertextd);
    encryptoTimes = stopTiming();
    printf("rescale Times:%lf microseconds\n", encryptoTimes*1000*1000/1);
    // printf("depth: %d\n",ciptertextd.depth);
    unsigned long long* dec = encryptor.decrypt(ciptertexta,keygen.pri);



    // auto plaina = encoder.decode(ciptertexta.a,2);
    // print<<<1,1>>>(dec,8);
    auto plaina = encoder.decode(dec,2);





    for(int i = 0; i < 16 ; i++){
        printf("%lf+%lf i\n",plaina[i].x/scale,plaina[i].y/scale);
    }
}