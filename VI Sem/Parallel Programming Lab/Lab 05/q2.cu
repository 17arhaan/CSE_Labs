#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#define THREADS_PER_BLOCK 256
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
int main() {
    int N;
    printf("Enter Size : ");
    scanf("%d", &N);
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);
    printf("Input Vector A:\n");
    for (int i = 0; i < N; i++) {
        scanf("%f", &A[i]);
    }
    printf("Input Vector B:\n");
    for (int i = 0; i < N; i++) {
        scanf("%f", &B[i]);
    }
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Resultant Vector --->\n[ ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("%f ", C[i]);
    }
    printf(" ]\n");
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
