#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void inclusiveScan(int* d_in, int* d_out, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;
    temp[2 * tid] = d_in[2 * tid];
    temp[2 * tid + 1] = d_in[2 * tid + 1];
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (tid == 0) {
        temp[n - 1] = 0;
    }
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    d_out[2 * tid] = temp[2 * tid];
    d_out[2 * tid + 1] = temp[2 * tid + 1];
}
int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);
    int size = n * sizeof(int);
    int* h_in = (int*)malloc(size);
    int* h_out = (int*)malloc(size);
    printf("Enter the elements of the input array:\n");
    for (int i = 0; i < n; ++i) {
        scanf("%d", &h_in[i]);
    }
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    int blockSize = n / 2;
    int numBlocks = 1;
    inclusiveScan<<<numBlocks, blockSize, 2 * blockSize * sizeof(int)>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    printf("Resulting array after inclusive scan:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");
    return 0;
}