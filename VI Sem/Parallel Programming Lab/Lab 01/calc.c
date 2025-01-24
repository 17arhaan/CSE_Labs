#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int num1, num2, result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) {
            printf("TRequirement Atleast 4 Processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter Number 1: ");
        scanf("%d",&num1);
        printf("Enter Number 2: ");
        scanf("%d",&num2);
    }
    
    MPI_Bcast(&num1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        result = num1 + num2;
        printf("Process %d: Addition result ( %d + %d ) = %d\n", rank, num1, num2, result);
    } else if (rank == 1) {
        result = num1 - num2;
        printf("Process %d: Subtraction result ( %d - %d ) = %d\n", rank, num1, num2, result);
    } else if (rank == 2) {
        result = num1 * num2;
        printf("Process %d: Multiplication result ( %d * %d ) = %d\n", rank, num1, num2, result);
    } else if (rank == 3) {
        if (num2 != 0) {
            result = num1 / num2;
            printf("Process %d: Division result ( %d / %d ) = %d\n", rank, num1, num2, result);
        } else {
            printf("Process %d: Division by zero error.\n", rank);
        }
    }
    MPI_Finalize();
    return 0;
}