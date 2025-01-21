#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int factorial(int n) {
    int fact = 1;
    for (int i = 1; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char *argv[]) {
    int rank, size, N, value, fact, sum = 0, i;
    int *A = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the number of values: ");
        fflush(stdout);
        scanf("%d", &N);
        A = (int*)malloc(N * sizeof(int));

        printf("Enter %d values: \n", N);
        for (i = 0; i < N; i++) {
            scanf("%d", &A[i]);
        }
    }

    MPI_Scatter(A, 1, MPI_INT, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    fact = factorial(value);

    MPI_Gather(&fact, 1, MPI_INT, A, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (i = 0; i < N; i++) {
            sum += A[i];
        }

        printf("The sum of factorials is: %d\n", sum);

        free(A);
    }

    MPI_Finalize();
    return 0;
}
