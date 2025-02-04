#include <stdio.h>
#include <cuda.h>
#define THREADS_PER_BLOCK 256
__global__ void oddEvenSort(float *arr, int n, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2) {
        int index = (phase % 2 == 0) ? (2 * i) : (2 * i + 1);
        if (index + 1 < n && arr[index] > arr[index + 1]) {
            float temp = arr[index];
            arr[index] = arr[index + 1];
            arr[index + 1] = temp;
        }
    }
}
int main() {
    int N;
    printf("Enter Size : ");
    scanf("%d", &N);
    float *arr, *d_arr;
    size_t size = N * sizeof(float);
    arr = (float*)malloc(size);
    printf("Enter Elements : ");
    for (int i = 0; i < N; i++) scanf("%f", &arr[i]);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int phase = 0; phase < N; phase++) {
        oddEvenSort<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_arr, N, phase);
    }
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    printf("Sorted Array --->\n[ ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) printf("%f ", arr[i]);
    printf("]\n");
    free(arr);
    cudaFree(d_arr);
    return 0;
}
