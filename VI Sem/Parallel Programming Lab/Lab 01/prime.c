#include "mpi.h"
#include <stdio.h>
#include <stdbool.h>

bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int primes[100];
    int start = 2, end = 100;
    int local_start, local_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int range_size = (end - start + 1) / 2;

    if (rank == 0) {
        local_start = start;
        local_end = start + range_size - 1;
    } else {
        local_start = start + range_size;
        local_end = end;
    }

    int local_primes[range_size];
    int local_count = 0;

    for (int i = local_start; i <= local_end; i++) {
        if (is_prime(i)) {
            local_primes[local_count++] = i;
        }
    }

    int total_primes[100];
    MPI_Gather(&local_count, 1, MPI_INT, total_primes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Prime numbers between 1 and 100: ");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < total_primes[i]; j++) {
                printf("%d ", local_primes[j]);
            }
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
