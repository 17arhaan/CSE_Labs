#include <stdio.h>
#include <cuda_runtime.h>
__global__ void ones_complement_inner(int *i, int *o, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1)
            o[idx] = i[idx];
        else
            o[idx] = ~i[idx];
    }
}
int main() {
    const int rows = 4, cols = 4;
    int h_i[] = {
        1, 2, 3, 4,
        6, 5, 8, 3,
        2, 4, 10, 1,
        9, 1, 2, 5
    };
    int h_o[rows * cols];
    int *d_i, *d_o;
    cudaMalloc(&d_i, sizeof(int) * rows * cols);
    cudaMalloc(&d_o, sizeof(int) * rows * cols);
    cudaMemcpy(d_i, h_i, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    ones_complement_inner<<<blocks, threads>>>(d_i, d_o, rows, cols);
    cudaMemcpy(h_o, d_o, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%d ", h_o[i * cols + j]);
        printf("\n");
    }
    cudaFree(d_i);
    cudaFree(d_o);
    return 0;
}