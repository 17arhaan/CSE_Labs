#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
__global__ void matMulCol(int *a, int *b, int *c, int ha, int wa) {
    int cidB = threadIdx.x;
    int wb = blockDim.x;
    int sum, k;
    for(int ridA = 0; ridA < ha; ridA++) {
        sum = 0;
        for(k = 0; k < wa; k++) {
            sum += (a[ridA * wa + k] * b[k * wb + cidB]);
        }
        c[ridA * wb + cidB] = sum;
    }
}
void initializeMatrix(int *matrix, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i * cols + j] = rand() % 10;
        }
    }
}
void printMatrix(int *matrix, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
int main() {
    int m, n, p;
    printf("Enter dimensions for matrix multiplication:\n");
    printf("Matrix A rows (m): ");
    scanf("%d", &m);
    printf("Matrix A columns / Matrix B rows (n): ");
    scanf("%d", &n);
    printf("Matrix B columns (p): ");
    scanf("%d", &p);
    int size_a = m * n * sizeof(int);
    int size_b = n * p * sizeof(int);
    int size_c = m * p * sizeof(int);
    int *h_a = (int *)malloc(size_a);
    int *h_b = (int *)malloc(size_b);
    int *h_c = (int *)malloc(size_c);
    srand(time(NULL));
    initializeMatrix(h_a, m, n);
    initializeMatrix(h_b, n, p);
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    matMulCol<<<1, p>>>(d_a, d_b, d_c, m, n);
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    printf("\nMatrix A (%d x %d):\n", m, n);
    printMatrix(h_a, m, n);
    printf("\nMatrix B (%d x %d):\n", n, p);
    printMatrix(h_b, n, p);    
    printf("\nResult Matrix C (%d x %d):\n", m, p);
    printMatrix(h_c, m, p);
    return 0;
}