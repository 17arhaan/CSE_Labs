#include <stdio.h>
#include <cuda_runtime.h>
__global__ void row_power(int *matrix, int rows, int cols) {
    int row = blockIdx.y;
    int col = threadIdx.x;
    if (col < cols) {
        int index = row * cols + col;
        int val = matrix[index];
        int result = 1;
        for (int i = 0; i < row + 1; ++i)
            result *= val;
        matrix[index] = result;
    }
}
int main() {
    const int rows = 4, cols = 4;
    int h_matrix[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        2, 3, 4, 5,
        1, 1, 2, 2
    };
    int *d_matrix;
    cudaMalloc(&d_matrix, sizeof(int) * rows * cols);
    cudaMemcpy(d_matrix, h_matrix, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);
    dim3 block(cols);
    dim3 grid(1, rows);
    row_power<<<grid, block>>>(d_matrix, rows, cols);
    cudaMemcpy(h_matrix, d_matrix, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%d ", h_matrix[i * cols + j]);
        printf("\n");
    }
    cudaFree(d_matrix);
    return 0;
}