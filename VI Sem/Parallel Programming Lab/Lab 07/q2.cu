#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  

__global__ void generateRS(char *d_S, char *d_RS, int S_length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < S_length) {
        d_RS[2 * i] = d_S[i];
        d_RS[2 * i + 1] = d_S[i];
    }
}

int main() {
    char h_S[] = "PCAP";
    char h_RS[2 * N];
    char *d_S, *d_RS;

    int S_length = strlen(h_S);
    int RS_length = 2 * S_length;

    cudaMalloc((void**)&d_S, S_length * sizeof(char));
    cudaMalloc((void**)&d_RS, RS_length * sizeof(char));

    cudaMemcpy(d_S, h_S, S_length * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (S_length + blockSize - 1) / blockSize;
    
    generateRS<<<numBlocks, blockSize>>>(d_S, d_RS, S_length);
    
    cudaMemcpy(h_RS, d_RS, RS_length * sizeof(char), cudaMemcpyDeviceToHost);

    h_RS[RS_length] = '\0';

    printf("Output string RS: %s\n", h_RS);

    cudaFree(d_S);
    cudaFree(d_RS);

    return 0;
}
