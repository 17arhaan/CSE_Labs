#include <stdio.h>
#include <cuda_runtime.h>

__global__ void compute_matrix(int *A, int *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int row_sum = 0, col_sum = 0;
        for (int i = 0; i < cols; ++i)
            row_sum += A[row * cols + i];
        for (int j = 0; j < rows; ++j)
            col_sum += A[j * cols + col];
        B[row * cols + col] = row_sum + col_sum;
    }
}

int main() {
    const int rows = 2, cols = 3;
    int h_A[] = {1, 2, 3, 4, 5, 6};
    int h_B[rows * cols];

    int *d_A, *d_B;
    cudaMalloc(&d_A, sizeof(int) * rows * cols);
    cudaMalloc(&d_B, sizeof(int) * rows * cols);

    cudaMemcpy(d_A, h_A, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    compute_matrix<<<blocks, threads>>>(d_A, d_B, rows, cols);

    cudaMemcpy(h_B, d_B, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%d ", h_B[i * cols + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
