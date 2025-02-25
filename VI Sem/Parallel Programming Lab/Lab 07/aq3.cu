#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  

__global__ void generateT(char *d_Sin, char *d_T, int Sin_length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < Sin_length) {
        int repeat = i + 1;
        for (int j = 0; j < repeat; j++) {
            d_T[i * repeat + j] = d_Sin[i];
        }
    }
}

int main() {
    char h_Sin[] = "Hai";
    char h_T[N];
    char *d_Sin, *d_T;

    int Sin_length = strlen(h_Sin);
    int T_length = (Sin_length * (Sin_length + 1)) / 2;

    cudaMalloc((void**)&d_Sin, Sin_length * sizeof(char));
    cudaMalloc((void**)&d_T, T_length * sizeof(char));

    cudaMemcpy(d_Sin, h_Sin, Sin_length * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (Sin_length + blockSize - 1) / blockSize;
    
    generateT<<<numBlocks, blockSize>>>(d_Sin, d_T, Sin_length);
    
    cudaMemcpy(h_T, d_T, T_length * sizeof(char), cudaMemcpyDeviceToHost);

    h_T[T_length] = '\0';

    printf("Output string T: %s\n", h_T);

    cudaFree(d_Sin);
    cudaFree(d_T);

    return 0;
}
