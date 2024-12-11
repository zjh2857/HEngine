#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
__device__ unsigned long long bitReverse(unsigned long long a, int bit_length)
{
    unsigned long long res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    return res;
}

__device__ __host__ unsigned modpow64(unsigned a, unsigned b, unsigned mod)
{
    unsigned res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        unsigned long long t64 = (unsigned long long)a * a;
        a = t64 % mod;
        if (b & 1)
        {
            unsigned long long r64 = (unsigned long long)a * res;
            res = r64 % mod;
        }

    }
    return res;
}

__global__ void fft(cuDoubleComplex* h_A, cuDoubleComplex* h_B, int N,int logN) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N/2) return;

    h_B[tid] = h_A[tid];
    h_B[tid+N/2] = h_A[tid + N / 2];
    cuDoubleComplex W, even, odd;

    for(int length = 1; length < N / 2; length*=2){
        int step = N / length / 2;
        // id = bitReverse(id, logN);


        double t = -1 * M_PI / N / length * (tid /(step) + 1);
        if(length == 1){
            printf("%d,%lf,%d\n",tid,t,step);
        }
        W.x = cos(t);
        W.y = sin(t);
        int loc = tid / step * 2 * step + tid % step;
        even = h_B[loc];
        odd = cuCmul(W, h_B[loc + step]);


        h_B[loc] = cuCadd(even, odd);
        h_B[loc + step] = cuCsub(odd, even);

        __syncthreads();

    }
}
__global__ void fft2(cuDoubleComplex* h_A,cuDoubleComplex* h_B,int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid;
    h_B[i].x = 0;
    h_B[i].y = 0;
    long double hx = 0;
    long double hy = 0;
    for(int j = 0; j < N; j++){
        
        int k = modpow64(3,j,2*N);
        if(j >= N / 2){
            k = (2*N-modpow64(3,N-j-1,2*N));
        }
        cuDoubleComplex t;
        long double tx = cos(-M_PI/N*i*(k));
        long double ty = sin(-M_PI/N*i*(k));
        // if(tid == 100 && j == 100){
        //     printf("%lf\n",t.x);
        // }
        hx += tx * h_A[j].x - ty * h_A[j].y;
        hy += tx * h_A[j].y + ty * h_A[j].x;
    }
    h_B[i].x = hx;
    h_B[i].y = hy;
}
int main() {
    int N = 4; // Small size for demonstration
    cuDoubleComplex* h_A_host = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    cuDoubleComplex* h_B_host = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    cuDoubleComplex* h_A_device;
    cuDoubleComplex* h_B_device;

    cudaMalloc(&h_A_device, N * sizeof(cuDoubleComplex));
    cudaMalloc(&h_B_device, N * sizeof(cuDoubleComplex));

    for (int i = 0; i < N; i++) {
        h_A_host[i].x = 0; // Example: real part as index
        h_A_host[i].y = 0; // Example: imaginary part as 0
    }
    h_A_host[0].x = 1;
    cudaMemcpy(h_A_device, h_A_host, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    fft<<<gridSize, blockSize>>>(h_A_device, h_B_device, N, log2(N));

    cudaMemcpy(h_B_host, h_B_device, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < N; i++) {
        printf("h_A[%d] = (%f, %f)\n", i, h_A_host[i].x, h_A_host[i].y);
    }

    printf("\nOutput:\n");
    for (int i = 0; i < N; i++) {
        printf("h_B[%d] = (%f, %f)\n", i, h_B_host[i].x, h_B_host[i].y);
    }


    fft2<<<gridSize, blockSize>>>(h_A_device, h_B_device, N);

    cudaMemcpy(h_B_host, h_B_device, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    printf("\nOutput:\n");
    for (int i = 0; i < N; i++) {
        printf("h_B[%d] = (%f, %f)\n", i, h_B_host[i].x, h_B_host[i].y);
    }

    free(h_A_host);
    free(h_B_host);
    cudaFree(h_A_device);
    cudaFree(h_B_device);

    return 0;
}
