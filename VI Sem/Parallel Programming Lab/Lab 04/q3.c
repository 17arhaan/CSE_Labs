#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
    int r, s;
    int arr[4][4], local_row[4], result_row[4], cumulative_row[4];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    if(r == 0){
        printf("Enter a 4x4 Matrix:\n");
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++) {
                scanf("%d", &arr[i][j]);
            }
        }
    }
    MPI_Scatter(arr, 4, MPI_INT, local_row, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scan(local_row, cumulative_row, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Gather(cumulative_row, 4, MPI_INT, arr, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if(r == 0){
        printf("Resultant arr:\n");
        for (int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                printf("%d ", arr[i][j]);
            }
            printf("\n");
        }
    }
    MPI_Finalize();
    return 0;
}