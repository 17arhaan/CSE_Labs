#include "mpi.h"
#include <stdio.h>
#define SIZE 9
int reverse_number(int num) {
    int reversed = 0;
    while (num != 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }
    return reversed;
}
int main(int argc, char* argv[]) {
    int rank, size;
    int input_array[SIZE] = {18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int reversed_array[SIZE];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int reversed_value = reverse_number(input_array[rank]);
    MPI_Gather(&reversed_value, 1, MPI_INT, reversed_array, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reversed array: ");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", reversed_array[i]);
        }
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
