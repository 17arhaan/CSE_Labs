#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int data = 100;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("This program requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Process 0: Before sending data to process 1\n");
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);  // Standard send
        printf("Process 0: Sent data to process 1\n");
    } else if (rank == 1) {
        printf("Process 1: Before sending data to process 0\n");
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // Standard send
        printf("Process 1: Sent data to process 0\n");
    }

    MPI_Finalize();
    return 0;
}
