#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define N 1024  

__global__ void reverseWords(char *sentence, char *result, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < len) {
        int start = i, end = i;
        
        while (start > 0 && sentence[start - 1] != ' ') start--;
        while (end < len && sentence[end] != ' ') end++;
        
        result[i] = sentence[start + (end - start - 1) - (i - start)];
    }
}

int main() {
    char h_sentence[] = "CUDA programming is fun";
    char h_result[N];
    char *d_sentence, *d_result;
    int len = strlen(h_sentence);

    cudaMalloc((void**)&d_sentence, len * sizeof(char));
    cudaMalloc((void**)&d_result, len * sizeof(char));

    cudaMemcpy(d_sentence, h_sentence, len * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (len + blockSize - 1) / blockSize;
    
    reverseWords<<<numBlocks, blockSize>>>(d_sentence, d_result, len);
    
    cudaMemcpy(h_result, d_result, len * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Reversed Words: %s\n", h_result);

    cudaFree(d_sentence);
    cudaFree(d_result);

    return 0;
}
