#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int x = 5;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    double result = pow(x, rank);
    
    printf("Process %d out of %d processes: pow(%d, %d) = %.2f\n", rank, size, x, rank, result);
    
    MPI_Finalize();
    return 0;
}
