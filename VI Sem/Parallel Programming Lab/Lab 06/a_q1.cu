#include <stdio.h>
#include <cuda.h>

__device__ int to_octal(int num) {
    int octal = 0, place = 1;
    while (num > 0) {
        octal += (num % 8) * place;
        num /= 8;
        place *= 10;
    }
    return octal;
}

__global__ void convert_to_octal(int *input, int *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = to_octal(input[i]);
    }
}

int main() {
    int N;
    printf("Enter the number of elements: ");
    scanf("%d", &N);

    int h_input[N], h_output[N];

    printf("Enter %d decimal numbers:\n", N);
    for (int i = 0; i < N; i++) {
        scanf("%d", &h_input[i]);
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    convert_to_octal<<<gridSize, blockSize>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Octal Values:\n");
    for (int i = 0; i < N; i++) {
        printf("%d -> %d\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
