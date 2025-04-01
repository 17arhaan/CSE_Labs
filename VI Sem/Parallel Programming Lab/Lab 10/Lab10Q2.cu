#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
__constant__ int d_filter[3];
__global__ void convolution(int* input, int* output, int width) {
    __shared__ int ds_input[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    ds_input[tid] = input[idx];
    __syncthreads();
    if (idx < width - 2) {
        output[idx] = ds_input[tid] * d_filter[0] + ds_input[tid + 1] * d_filter[1] + ds_input[tid + 2] * d_filter[2];
    }
}
int main() {
    int width;
    printf("Enter the width of the input array: ");
    scanf("%d", &width);
    size_t size = width * sizeof(int);
    int* h_input = (int*)malloc(size);
    int* h_output = (int*)malloc(size);
    int h_filter[3];
    printf("Enter the elements of the input array:\n");
    for (int i = 0; i < width; ++i) {
        scanf("%d", &h_input[i]);
    }
    printf("Enter the elements of the filter (3 elements):\n");
    for (int i = 0; i < 3; ++i) {
        scanf("%d", &h_filter[i]);
    }
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, 3 * sizeof(int));
    int blockSize = 256;
    int numBlocks = (width + blockSize - 1) / blockSize;
    convolution<<<numBlocks, blockSize>>>(d_input, d_output, width);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("Resulting array after convolution:\n");
    for (int i = 0; i < width - 2; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    return 0;
}