#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>

#define N 1024

__global__ void CUDACount(char* A, unsigned int *d_count){
    int i = threadIdx.x;
    if (A[i] == 'a')
        atomicAdd(d_count, 1);
}

int main() {
    char A[N];
    char *d_A;
    unsigned int d_count = 0, *d_result;
    
    printf("Enter a string: ");
    gets(A);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc((void**)&d_A, strlen(A) * sizeof(char));
    cudaMalloc((void**)&d_result, sizeof(unsigned int));
    
    cudaMemcpy(d_A, A, strlen(A) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &d_count, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start, 0);
    CUDACount<<<1, strlen(A)>>>(d_A, d_result);
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cudaMemcpy(&d_count, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    printf("Total occurrences of 'a': %d\n", d_count);
    printf("Time Taken: %f ms\n", elapsedTime);
    
    cudaFree(d_A);
    cudaFree(d_result);
    
    return 0;
}
