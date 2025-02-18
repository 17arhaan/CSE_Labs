#include <stdio.h>
#include <cuda.h>

__device__ int ones_complement(int num) {
    int bits = sizeof(int) * 8; // Find number of bits in an int
    int mask = (1 << bits) - 1; // Create a mask of all 1s
    return (~num) & mask;  // Apply bitwise NOT and mask to avoid sign extension
}

__global__ void compute_ones_complement(int *input, int *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = ones_complement(input[i]);
    }
}

void print_binary(int num) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (num >> i) & 1);
    }
}

int main() {
    int N;
    
    printf("Enter the number of binary numbers: ");
    scanf("%d", &N);

    int h_input[N], h_output[N];

    printf("Enter %d binary numbers (as decimal integers, e.g., 10 for 0b1010):\n", N);
    for (int i = 0; i < N; i++) {
        scanf("%d", &h_input[i]);
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    compute_ones_complement<<<gridSize, blockSize>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nOne's Complement Results:\n");
    for (int i = 0; i < N; i++) {
        printf("Original: ");
        print_binary(h_input[i]);
        printf(" (%d) -> One's Complement: ", h_input[i]);
        print_binary(h_output[i]);
        printf(" (%d)\n", h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
