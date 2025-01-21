#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
int main(int argc, char **argv){
    int rank, size, *ori, *store, m;
    float avg = 0.0, ans;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0){
        printf("Enter Value of M : ");
        scanf("%d", &m);
        ori = (int *)malloc((m * size) * sizeof(int));
        printf("Enter the elements : \n");
        for(int i = 0; i < m * size; ++ i){
            scanf("%d", &ori[i]);
        }
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    store = (int *)malloc(m * sizeof(int));
    MPI_Scatter(ori, m, MPI_INT, store, m, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < m; ++i){
        avg += store[i];
    }
    avg /= m;
    printf("%f from rank %d\n", avg, rank);
    MPI_Reduce(&avg, &ans, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("%f is the answer\n", ans);
    }
    MPI_Finalize();
}