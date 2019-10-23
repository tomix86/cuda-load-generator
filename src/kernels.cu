#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void touchMemory(float* memory) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
}