#include <stdio.h>
#include <cuda_runtime.h>
__global__ void transformMatrix(int *a, int *b, int m, int n) {
    int row = blockIdx.x;     
    int col = threadIdx.x;   
    if (row < m && col < n) {
        int idx = row * n + col; 
        int rowSum = 0, colSum = 0;
        for (int i = 0; i < n; i++) {
            rowSum += a[row * n + i];
        }
        for (int j = 0; j < m; j++) {
            colSum += a[j * n + col];
        }
        if (a[idx] % 2 == 0) {
            b[idx] = rowSum;
        } else {
            b[idx] = colSum;
        }
    }
}
int main() {
    int m = 3, n = 3;  
    int size = m * n * sizeof(int);
    
    int h_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};  
    int h_b[m * n];  

    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    transformMatrix<<<m, n>>>(d_a, d_b, m, n);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    printf("Matrix A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", h_a[i * n + j]);
        }
        printf("\n");
    }
    printf("\nMatrix B (Transformed):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", h_b[i * n + j]);
        }
        printf("\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
