#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int *arr;
    int value;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    
    if (rank == 0) {
        printf("Enter Elements :-\n", size);
        arr = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            scanf("%d", &arr[i]);
        }
    }

    MPI_Buffer_attach((void*)malloc(1024), 1024); 

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Bsend(&arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (rank % 2 == 0) {
            printf("Process %d: Square of %d is %d\n", rank, value, value * value);
        } else {
            printf("Process %d: Cube of %d is %d\n", rank, value, value * value * value);
        }
    }

    MPI_Buffer_detach(NULL, NULL); 
    MPI_Finalize();
    return 0;
}
