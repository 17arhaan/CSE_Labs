#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  

__global__ void repeatString(char *d_S, char *d_Sout, int S_length, int repeatCount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < S_length * repeatCount) {
        d_Sout[i] = d_S[i % S_length];
    }
}

int main() {
    char h_S[] = "Hello";
    int repeatCount = 3;
    char h_Sout[N];
    char *d_S, *d_Sout;

    int S_length = strlen(h_S);
    int Sout_length = S_length * repeatCount;

    cudaMalloc((void**)&d_S, S_length * sizeof(char));
    cudaMalloc((void**)&d_Sout, Sout_length * sizeof(char));

    cudaMemcpy(d_S, h_S, S_length * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (Sout_length + blockSize - 1) / blockSize;
    
    repeatString<<<numBlocks, blockSize>>>(d_S, d_Sout, S_length, repeatCount);
    
    cudaMemcpy(h_Sout, d_Sout, Sout_length * sizeof(char), cudaMemcpyDeviceToHost);

    h_Sout[Sout_length] = '\0';

    printf("Output string Sout: %s\n", h_Sout);

    cudaFree(d_S);
    cudaFree(d_Sout);

    return 0;
}
