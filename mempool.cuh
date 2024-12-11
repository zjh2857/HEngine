
#pragma once
char* point = 0;

char* start = 0;
unsigned long long poolsize = (1llu << 33  );
int sumsize = 0;
#include <iostream>
using namespace std;
#define Check(call)														\
{																		\
	cudaError_t status = call;											\
	if (status != cudaSuccess)											\
	{																	\
		cout << "行号" << __FILE__ << __LINE__ << endl;							\
		cout << "错误:" << cudaGetErrorString(status) << endl;			\
	}																	\
}
cudaError_t mempool(unsigned long long** p, unsigned long long size){
    if(size % 16 != 0){
        size = size + (16 - size % 16);
    }
    // return cudaMalloc(p,size);
    if(point - start > poolsize / 16 * 15){
        point = start + poolsize / 4;
        printf("memory!!\n");
    }
    // printf("QQQQ\n");
    // printf("%p\n",*p);
    if (point!=0){
        
        
        *p = (unsigned long long*)point;
        point = (point + size);
    }else{
        // unsigned long long poolsize = (1llu << 31);
        Check(cudaMalloc((void**)&point, poolsize));
        start = point;
        *p = (unsigned long long*)point;
        point = (point + size);
    }
    // printf("%p,%p\n",point-start,(1llu << 31));

    return cudaError_t(0);
}

cudaError_t mempool(void** p, unsigned long long size){
    // return cudaMalloc(p,size);
    if(size % 16 != 0){
        size = size + (16 - size % 16);
    }
    if(point - start > poolsize / 16 * 15){
        point = start + poolsize / 8;
        printf("memory!!\n");

    }

    if (point!=0){
        *p = (void*)point;
        point = (point + size);
    }else{
        // unsigned long long poolsize = 1llu << 31;
        Check(cudaMalloc((void**)&point, poolsize));
                start = point;

        *p = (void*)point;
        point = (point + size);
    }
    // printf("%p,%p\n",point-start,(1llu << 31));

    return cudaError_t(0);
}

cudaError_t mempool(cuDoubleComplex ** p, unsigned long long size){
    if(size % 16 != 0){
        size = size + (16 - size % 16);
    }
    // return cudaMalloc(p,size);
    if(point - start > poolsize / 16 * 15){
        point = start + poolsize / 8;
        printf("memory!!\n");

    }
    if (point!=0){
        *p = (cuDoubleComplex *)point;
        point = (point + size);
    }else{
        // unsigned long long poolsize = 1llu << 31;
        Check(cudaMalloc((void**)&point, poolsize));
                start = point;

        *p = (cuDoubleComplex*)point;
        point = (point + size);
    }
    // printf("%p,%p\n",point-start,(1llu << 31));
    return cudaError_t(0);
}