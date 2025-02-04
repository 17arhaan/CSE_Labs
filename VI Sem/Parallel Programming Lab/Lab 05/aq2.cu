#include <stdio.h>
#include <cuda.h>
#define THREADS_PER_BLOCK 256
__global__ void selectionSortRows(float *matrix, int cols, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int i = 0; i < cols - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < cols; j++) {
                if (matrix[row * cols + j] < matrix[row * cols + minIdx]) {
                    minIdx = j;
                }
            }
            float temp = matrix[row * cols + i];
            matrix[row * cols + i] = matrix[row * cols + minIdx];
            matrix[row * cols + minIdx] = temp;
        }
    }
}
int main() {
    int rows, cols;
    printf("Enter Dimensions : ");
    scanf("%d %d", &rows, &cols);

    float *matrix, *d_matrix;
    size_t size = rows * cols * sizeof(float);
    matrix = (float*)malloc(size);

    printf("Enter Elements :\n");
    for (int i = 0; i < rows * cols; i++) scanf("%f", &matrix[i]);

    cudaMalloc((void**)&d_matrix, size);
    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

    int blocksPerGrid = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    selectionSortRows<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_matrix, cols, rows);

    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    printf("Sorted Matrix --->\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    free(matrix);
    cudaFree(d_matrix);
    return 0;
}
