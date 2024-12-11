#include"iostream"
#include"cuda_runtime_api.h"
#include"device_launch_parameters.h"
#include"cufft.h"
using namespace std;
__global__ void normalizing(cufftDoubleComplex* data,int data_len)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	data[idx].x /= data_len;
	data[idx].y /= data_len;
}
void Check(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		cout << "line:" << __LINE__ << endl;
		cout << "error:" << cudaGetErrorString(status) << endl;
	}
}
int main()
{
	return 0;
}