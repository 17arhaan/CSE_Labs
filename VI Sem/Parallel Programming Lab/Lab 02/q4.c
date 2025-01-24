#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size, value;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
        printf("Enter an integer value: ");
        scanf("%d", &value);
        
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent value %d to process 1\n", rank, value);
        
        MPI_Recv(&value, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, &status);
        printf("Process %d received value %d from process %d\n", rank, value, size-1);
    } else {
        MPI_Recv(&value, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
        value++;
        printf("Process %d received value %d \n", rank, status.MPI_SOURCE);
        
        if (rank != size - 1) {
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            printf("Process %d sent value %d to process %d\n", rank, value, rank + 1);
        } else {
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            printf("Process %d sent value %d back to process 0\n", rank, value);
        }
    }

    MPI_Finalize();
    return 0;
}
