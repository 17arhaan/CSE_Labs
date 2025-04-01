#include <stdio.h>
#include <cuda_runtime.h>

__global__ void spmv_csr_kernel(int *row_ptr, int *col_idx, float *vals, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int jj = row_start; jj < row_end; jj++)
            dot += vals[jj] * x[col_idx[jj]];
        y[row] = dot;
    }
}

int main() {
    const int num_rows = 4;
    const int num_cols = 4;
    const int num_vals = 9;

    int h_row_ptr[] = {0, 2, 4, 7, 9};
    int h_col_idx[] = {0, 1, 1, 2, 0, 2, 3, 2, 3};
    float h_vals[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float h_x[] = {1, 2, 3, 4};
    float h_y[num_rows];

    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, sizeof(int) * (num_rows + 1));
    cudaMalloc(&d_col_idx, sizeof(int) * num_vals);
    cudaMalloc(&d_vals, sizeof(float) * num_vals);
    cudaMalloc(&d_x, sizeof(float) * num_cols);
    cudaMalloc(&d_y, sizeof(float) * num_rows);

    cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, sizeof(int) * num_vals, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals, sizeof(float) * num_vals, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float) * num_cols, cudaMemcpyHostToDevice);

    spmv_csr_kernel<<<1, num_rows>>>(d_row_ptr, d_col_idx, d_vals, d_x, d_y, num_rows);

    cudaMemcpy(h_y, d_y, sizeof(float) * num_rows, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_rows; ++i)
        printf("%f\n", h_y[i]);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_vals);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
