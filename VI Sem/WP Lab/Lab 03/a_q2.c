#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size, N, *a, *local_a, local_even_count = 0, local_odd_count = 0, global_even_count = 0, global_odd_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the size of the array N: ");
        scanf("%d", &N);

        a = (int*)malloc(N * sizeof(int));

        printf("Enter the elements of the array:\n");
        for (int i = 0; i < N; ++i) {
            scanf("%d", &a[i]);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_a = (int*)malloc(N / size * sizeof(int));

    MPI_Scatter(a, N / size, MPI_INT, local_a, N / size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N / size; ++i) {
        if (local_a[i] % 2 == 0) {
            local_a[i] = 1;
            local_even_count++;
        } else {
            local_a[i] = 0;
            local_odd_count++;
        }
    }

    MPI_Gather(local_a, N / size, MPI_INT, a, N / size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_even_count, &global_even_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_odd_count, &global_odd_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultant array:\n");
        for (int i = 0; i < N; ++i) {
            printf("%d ", a[i]);
        }
        printf("\n");

        printf("Count of even numbers: %d\n", global_even_count);
        printf("Count of odd numbers: %d\n", global_odd_count);

        free(a);
    }

    free(local_a);

    MPI_Finalize();
    return 0;
}
