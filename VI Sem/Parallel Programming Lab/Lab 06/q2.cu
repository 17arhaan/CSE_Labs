#include <stdio.h>
#include <cuda.h>

__global__ void selection_sort(int *arr, int n) {
    int tid = threadIdx.x;
    
    if (tid == 0) { 
        for (int i = 0; i < n - 1; i++) {
            int min_idx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[min_idx]) {
                    min_idx = j;
                }
            }
            int temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
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

    selection_sort<<<1, 1>>>(d_arr, N);

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
