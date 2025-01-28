#include <stdio.h>
#include <mpi.h>
#define N 3
int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[N][N];
    int searchElement, localCount = 0, totalCount = 0;
    int i, j;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 3) {
        if (rank == 0) {
            printf("This program requires exactly 3 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Enter the 3x3 matrix:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter the element to search: ");
        scanf("%d", &searchElement);
    }
    MPI_Bcast(&searchElement, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    int startRow = rank;
    for (i = startRow; i < startRow + 1; i++) {
        for (j = 0; j < N; j++) {
            if (matrix[i][j] == searchElement) {
                localCount++;
            }
        }
    }
    MPI_Reduce(&localCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("The element %d occurred %d times in the matrix.\n", searchElement, totalCount);
    }
    MPI_Finalize();
    return 0;
}