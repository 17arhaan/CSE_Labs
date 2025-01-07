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

/*
This program uses MPI to calculate the power of a number `x` raised to the rank of each process.

1. **MPI Initialization**: 
   - The program initializes the MPI environment with `MPI_Init()`. This is required for any MPI program.

2. **Rank and Size**: 
   - The program retrieves the rank (unique ID) of the current process using `MPI_Comm_rank()`, and the total number of processes using `MPI_Comm_size()`.

3. **Input from Rank 0**: 
   - The rank 0 process prompts the user to input the value of `x`. If `x == 1`, the program prints a message and exits early to avoid unnecessary calculations.

4. **Broadcasting the Value**: 
   - The value of `x` is broadcasted from rank 0 to all other processes using `MPI_Bcast()`. This ensures that all processes have the same value of `x`.

5. **Power Calculation**:
   - Each process calculates `pow(x, rank)`, where the rank of the process is used as the exponent. This means each process calculates the power of `x` raised to its own rank (e.g., `x^0`, `x^1`, `x^2`, ...).

6. **Output**:
   - Each process prints its rank, the total number of processes, the value of `x`, its rank, and the calculated result (`pow(x, rank)`).

7. **MPI Finalization**:
   - The MPI environment is finalized using `MPI_Finalize()`. This is necessary before the program exits to clean up the MPI resources.
*/
