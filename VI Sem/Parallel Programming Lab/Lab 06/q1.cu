#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 16  

__global__ void convolution_1D(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}

int main() {
    int Width, Mask_Width;

    // Get input from the user
    printf("Enter Width: ");
    scanf("%d", &Width);
    
    printf("Enter Mask_Width: ");
    scanf("%d", &Mask_Width);

    float h_N[Width], h_M[Mask_Width], h_P[Width];

    printf("Enter %d Elements:\n", Width);
    for (int i = 0; i < Width; i++) {
        scanf("%f", &h_N[i]);
    }

    printf("Enter %d Elements for the mask array:\n", Mask_Width);
    for (int i = 0; i < Mask_Width; i++) {
        scanf("%f", &h_M[i]);
    }

    float *d_N, *d_M, *d_P;

    cudaMalloc(&d_N, Width * sizeof(float));
    cudaMalloc(&d_M, Mask_Width * sizeof(float));
    cudaMalloc(&d_P, Width * sizeof(float));

    cudaMemcpy(d_N, h_N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (Width + blockSize - 1) / blockSize;

    convolution_1D<<<gridSize, blockSize>>>(d_N, d_M, d_P, Mask_Width, Width);

    cudaMemcpy(h_P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultant Array---> ");
    for (int i = 0; i < Width; i++) {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
