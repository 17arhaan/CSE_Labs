#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  
#define WORD_LEN 32 

__global__ void countWordOccurrences(char *sentence, char *word, int *count, int sentenceLength, int wordLength) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i <= sentenceLength - wordLength) {
        int match = 1;
        for (int j = 0; j < wordLength; j++) {
            if (sentence[i + j] != word[j]) {
                match = 0;
                break;
            }
        }
        if (match) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    char sentence[] = "hello world hello hello CUDA hello";
    char word[] = "hello";
    int h_count = 0, *d_count;
    
    int sentenceLength = strlen(sentence);
    int wordLength = strlen(word);

    char *d_sentence, *d_word;
    
    cudaMalloc((void**)&d_sentence, sentenceLength * sizeof(char));
    cudaMalloc((void**)&d_word, wordLength * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, sentence, sentenceLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (sentenceLength + blockSize - 1) / blockSize;
    
    countWordOccurrences<<<numBlocks, blockSize>>>(d_sentence, d_word, d_count, sentenceLength, wordLength);
    
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the given sentence.\n", word, h_count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
