#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void matAddRow(int *a, int *b, int *c, int m, int n) {
    int row = threadIdx.x;
    if (row < m) {
        for (int j = 0; j < n; j++) {
            c[row * n + j] = a[row * n + j] + b[row * n + j];
        }
    }
}
int main() {
    int m, n;
    printf("Enter m and n: ");
    scanf("%d %d", &m, &n);
    int *h_a = (int *)malloc(m * n * sizeof(int));
    int *h_b = (int *)malloc(m * n * sizeof(int));
    int *h_c = (int *)malloc(m * n * sizeof(int));
    printf("Enter mat a: ");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &h_a[i * n + j]);
        }
    }
    printf("Enter mat b: ");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            scanf("%d", &h_b[i * n + j]);
        }
    }
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * n * sizeof(int));
    cudaMalloc(&d_b, m * n * sizeof(int));
    cudaMalloc(&d_c, m * n * sizeof(int));
    cudaMemcpy(d_a, h_a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m * n * sizeof(int), cudaMemcpyHostToDevice);
    matAddRow<<<1, m>>>(d_a, d_b, d_c, m, n);
    cudaMemcpy(h_c, d_c, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Resultant Matrix:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", h_c[i * n + j]);
        }
        printf("\n");
    }
    return 0;
}