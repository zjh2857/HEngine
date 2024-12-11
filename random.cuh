
__device__ unsigned long long devData = 1;
__global__ void genRandom(unsigned long long *randomVec,unsigned long long scale){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // randomVec[tid] = (1) * scale;
    // randomVec[tid] = 0;
    // if(scale == 114514){
    //     if(tid == 1){
    //         // randomVec[tid] = 1179648;
    //         randomVec[tid] = 1;

    //     }else{
    //         randomVec[tid] = 0;
    //     }
    //     return ;
    // }
        //     randomVec[tid] = 0;
        // return ;
    if(scale == 0){
        randomVec[tid] = 0;
        return ;
    }
    if(scale == 1){
        randomVec[0] = 0;
        return ;        
    }
    // if(scale < 10){
    //     randomVec[tid] = 0;
    //     return ;
    // }
    randomVec[tid] = (44553453456 * tid + 5134554) % 4123123;
    // randomVec[tid] = (44553453456 * tid + 5134554) % 5;
        // randomVec[tid] = 100;



    // randomVec[tid] = (44553453456 * tid + 5134554) % 55;
    
    // randomVec[0] = 1;
    

    // randomVec[tid] = 0;
    
        // randomVec[0] = 1;
// 


    devData = randomVec[tid];
    // if(scale == 0){
    //     randomVec[tid] = 0;
    // }
    // else if(scale == 1){
    //     randomVec[tid] = 1;
    // }
    // else{
    //     randomVec[tid] = 1179649 * 2 ;
    // }
}

__global__ void genRandom_s (unsigned long long *randomVec,unsigned long long scale){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
        //     randomVec[tid] = 0;
        // return ;
    if(tid % 100== 0){
        randomVec[tid] = 1;
    }
    else{
        randomVec[tid] = 0;
    }
    // randomVec[tid] = (tid) % 1;
    // printf("%lld\n",devData);
    devData = randomVec[tid];
}
__global__ void genRandom_u (unsigned long long *randomVec,unsigned long long scale){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
        //     randomVec[tid] = 0;
        // return ;
    if(tid == 0)
        randomVec[tid] = 1;
    else
        randomVec[tid] = 0;
    // randomVec[tid] = (tid) % 2;
    // printf("%lld\n",devData);
    // devData = randomVec[tid];
}