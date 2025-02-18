#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 8  

__global__ void odd_even_sort(int *arr, int n) {
    int tid = threadIdx.x;

    for (int i = 0; i < n; i++) {
        if (i % 2 == 0 && tid < n / 2) {
            int idx = 2 * tid;
            if (idx + 1 < n && arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
        __syncthreads();

        if (i % 2 == 1 && tid < (n - 1) / 2) {
            int idx = 2 * tid + 1;
            if (idx + 1 < n && arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
        __syncthreads();
    }
}

int main() {
    int N;
    
    printf("Enter the number of elements in the array: ");
    scanf("%d", &N);

    int h_arr[N];
    
    printf("Enter %d elements:\n", N);
    for (int i = 0; i < N; i++) {
        scanf("%d", &h_arr[i]);
    }

    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = (N + 1) / 2;  
    odd_even_sort<<<1, blockSize>>>(d_arr, N);

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
