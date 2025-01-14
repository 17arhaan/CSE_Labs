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
        MPI_Recv(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive data from process 1
        printf("Process 0: Received data %d from process 1\n", data);
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);  // Send data to process 1
        printf("Process 0: Sent data to process 1\n");
    } else if (rank == 1) {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive data from process 0
        printf("Process 1: Received data %d from process 0\n", data);
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // Send data to process 0
        printf("Process 1: Sent data to process 0\n");
    }

    MPI_Finalize();
    return 0;
}
