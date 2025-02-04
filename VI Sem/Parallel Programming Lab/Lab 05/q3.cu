#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#define THREADS_PER_BLOCK 256
__global__ void computeSine(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = sinf(input[i]);
    }
}
int main() {
    int N;
    printf("Enter Size : ");
    scanf("%d", &N);
    float *input, *output;
    float *d_input, *d_output;
    size_t size = N * sizeof(float);
    input = (float*)malloc(size);
    output = (float*)malloc(size);
    printf("Enter %d Elements (In rad): ", N);
    for (int i = 0; i < N; i++) {
        scanf("%f", &input[i]);
    }
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    computeSine<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    printf("Sine values of the input angles:\n[ ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("%f ", output[i]);
    }
    printf("]\n");
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
