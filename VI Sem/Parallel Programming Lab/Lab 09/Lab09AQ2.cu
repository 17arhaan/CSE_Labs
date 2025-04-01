#include <stdio.h>
#include <cuda_runtime.h>
 
__global__ void repeat_chars(char *A, int *B, char *output, int *offsets, int total_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (idx < total_len) {
        int start = (idx == 0) ? 0 : offsets[idx - 1];
        char ch = A[idx];
        int reps = B[idx];
        for (int j = 0; j < reps; ++j)
            output[start + j] = ch;
    }
}
 
int main() {
    const int rows = 2, cols = 3;
    char h_A[] = {'p', 'C', 'a', 'e', 'X', 'M'};
    int h_B[] = {1, 2, 4, 2, 4, 3};
    int h_offsets[rows * cols];
    int total_len = 0;
 
    for (int i = 0; i < rows * cols; ++i) {
        h_offsets[i] = total_len;
        total_len += h_B[i];
    }
 
    char *d_A, *d_output;
    int *d_B, *d_offsets;
 
    cudaMalloc(&d_A, sizeof(char) * rows * cols);
    cudaMalloc(&d_B, sizeof(int) * rows * cols);
    cudaMalloc(&d_offsets, sizeof(int) * rows * cols);
    cudaMalloc(&d_output, sizeof(char) * total_len);
 
    cudaMemcpy(d_A, h_A, sizeof(char) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
 
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    repeat_chars<<<blocks, threads>>>(d_A, d_B, d_output, d_offsets, total_len);
 
    char *h_output = (char *)malloc(total_len * sizeof(char));
    cudaMemcpy(h_output, d_output, sizeof(char) * total_len, cudaMemcpyDeviceToHost);
 
    printf("Output string: ");
    for (int i = 0; i < total_len; ++i)
        printf("%c", h_output[i]);
    printf("\n");
 
    free(h_output);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_offsets);
    cudaFree(d_output);
 
    return 0;
}