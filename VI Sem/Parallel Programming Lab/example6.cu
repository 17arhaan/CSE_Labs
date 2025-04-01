#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void transpose(int *a, int *t, int m, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < m && j < n) {
        t[j * m + i] = a[i * n + j];
    }
}

int main(void) {
    int *a, *t;
    int m, n, i, j;
    int *d_a, *d_t;

    printf("Enter the value of m: ");
    scanf("%d", &m);
    printf("Enter the value of n: ");
    scanf("%d", &n);

    int size = sizeof(int) * m * n;
    a = (int*)malloc(size);
    t = (int*)malloc(size);

    printf("Enter input matrix:\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            scanf("%d", &a[i * n + j]);
        }
    }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_t, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose<<<numBlocks, threadsPerBlock>>>(d_a, d_t, m, n);

    cudaMemcpy(t, d_t, size, cudaMemcpyDeviceToHost);

    printf("Result vector is:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%d ", t[i * m + j]);
        }
        printf("\n");
    }

    free(a);
    free(t);
    cudaFree(d_a);
    cudaFree(d_t);

    return 0;
}
