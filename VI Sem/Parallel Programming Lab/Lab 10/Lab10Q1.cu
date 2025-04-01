#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
__global__ void matrixMul(int* A, int* B, int* C, int width) {
    __shared__ int ds_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int ds_B[BLOCK_SIZE][BLOCK_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    int sum = 0;
    for (int t = 0; t < (width - 1) / BLOCK_SIZE + 1; ++t) {
        if (row < width && t * BLOCK_SIZE + tx < width){
            ds_A[ty][tx] = A[row * width + t * BLOCK_SIZE + tx];
        }
        else{
            ds_A[ty][tx] = 0;
        }
        if (t * BLOCK_SIZE + ty < width && col < width){
            ds_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * width + col];
        }
        else{
            ds_B[ty][tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; ++i){
            sum += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width){
        C[row * width + col] = sum;
    }
}
int main() {
    int width;
    printf("Enter the width of the matrices: ");
    scanf("%d", &width);
    size_t size = width * width * sizeof(int);
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);
    printf("Enter the elements of matrix A:\n");
    for (int i = 0; i < width * width; ++i) {
        scanf("%d", &h_A[i]);
    }
    printf("Enter the elements of matrix B:\n");
    for (int i = 0; i < width * width; ++i) {
        scanf("%d", &h_B[i]);
    }
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y, 1);
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Resulting matrix C:\n");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", h_C[i * width + j]);
        }
        printf("\n");
    }
    return 0;
}