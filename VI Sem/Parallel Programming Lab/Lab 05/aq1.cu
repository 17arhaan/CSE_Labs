#include <stdio.h>
#include <cuda.h>
#define THREADS_PER_BLOCK 256
__global__ void saxpy(float *x, float *y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
int main() {
    int N;
    printf("Enter Size: ");
    scanf("%d", &N);
    float *x, *y, a;
    float *d_x, *d_y;
    size_t size = N * sizeof(float);
    x = (float*)malloc(size);
    y = (float*)malloc(size);
    printf("Enter Scalar a: ");
    scanf("%f", &a);
    printf("Input Vector x :");
    for (int i = 0; i < N; i++) scanf("%f", &x[i]);
    printf("Input Vector y :");
    for (int i = 0; i < N; i++) scanf("%f", &y[i]);
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    saxpy<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_x, d_y, a, N);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    printf("Resultant Vector y --->\n[ ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) printf("%f ", y[i]);
    printf("]\n");
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
