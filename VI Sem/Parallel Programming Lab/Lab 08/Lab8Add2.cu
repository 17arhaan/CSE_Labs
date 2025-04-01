#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
__device__ int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
__device__ int sumOfDigits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}
__global__ void transformMatrix(int *a, int *b, int n) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < n && col < n) {
        int idx = row * n + col;
        if (row == col) {
            b[idx] = 0;
        }
        else if (row < col) {
            b[idx] = factorial(a[idx]);
        }
        else {
            b[idx] = sumOfDigits(a[idx]);
        }
    }
}
int main() {
    int n;
    printf("Enter the size of the matrix (n x n): ");
    scanf("%d", &n);
    int size = n * n * sizeof(int);
    int *h_a = (int *)malloc(size);  
    int *h_b = (int *)malloc(size); 
    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        h_a[i] = rand() % 10 + 1;
    }
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    transformMatrix<<<n, n>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    printf("\nMatrix A (Original):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", h_a[i * n + j]);
        }
        printf("\n");
    }
    printf("\nMatrix B (Transformed):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", h_b[i * n + j]);
        }
        printf("\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}
