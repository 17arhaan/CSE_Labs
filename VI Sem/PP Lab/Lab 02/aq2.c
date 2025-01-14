#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

long long factorial(int num) {
    long long result = 1;
    for (int i = 2; i <= num; i++) {
        result *= i;
    }
    return result;
}

long long sum_of_integers(int num) {
    return (num * (num + 1)) / 2;
}

int main(int argc, char* argv[]) {
    int rank, size, n;
    long long local_sum = 0, total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the value of n: ");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = rank + 1; i <= n; i += size) {
        if (i % 2 == 1) {
            local_sum += factorial(i);
        } else {
            local_sum += sum_of_integers(i);
        }
    }

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The total sum is: %lld\n", total_sum);
    }

    MPI_Finalize();
    return 0;
}
