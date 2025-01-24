#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    int rank, size, n;
    int *array;
    int *results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
        printf("Enter the number of elements: ");
        scanf("%d", &n);

        array = (int*)malloc(n * sizeof(int));
        printf("Enter %d elements: ", n);
        for (int i = 0; i < n; i++) {
            scanf("%d", &array[i]);
        }

        results = (int*)malloc(n * sizeof(int));
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    array = (int*)malloc(n * sizeof(int));
    MPI_Bcast(array, n, MPI_INT, 0, MPI_COMM_WORLD);

    int elements_per_process = n / size;
    int remainder = n % size;
    int start = rank * elements_per_process + (rank < remainder ? rank : remainder);
    int end = start + elements_per_process + (rank < remainder ? 1 : 0);

    for (int i = start; i < end; i++) {
        results[i] = is_prime(array[i]) ? 1 : 0;
    }

    MPI_Gather(&results[start], end - start, MPI_INT, results, elements_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Results:\n");
        for (int i = 0; i < n; i++) {
            if (results[i] == 1) {
                printf("%d is prime\n", array[i]);
            } else {
                printf("%d is not prime\n", array[i]);
            }
        }

        free(array);
        free(results);
    }

    MPI_Finalize();
    return 0;
}
