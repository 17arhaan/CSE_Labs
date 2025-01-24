#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv){
    int rank, size, l_size;
    char *s1, *s2, *local_s1, *local_s2, *ans;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0){
        s1 = (char *)malloc(100 * sizeof(char));
        s2 = (char *)malloc(100 * sizeof(char));
        printf("Enter String 1 : ");
        scanf("%s", s1);
        printf("Enter String 2 : ");
        scanf("%s", s2);
        l_size = (strlen(s1) / size);
        ans = (char *)malloc(((strlen(s1) * 2) + 1) * sizeof(char));
        ans[strlen(s1) * 2] = '\0';
    }
    MPI_Bcast(&l_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_s1 = (char *)malloc(l_size * sizeof(char));
    local_s2 = (char *)malloc(l_size * sizeof(char));
    MPI_Scatter(s1, l_size, MPI_CHAR, local_s1, l_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, l_size, MPI_CHAR, local_s2, l_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    char *store = (char *)malloc((l_size + l_size + 1) * sizeof(char));
    int check = 1, i = 0, j = 0, k = 0;
    while (i < l_size || j < l_size){
        if (check){
            store[k++] = local_s1[i++];
            check = 0;
        }
        else{
            store[k++] = local_s2[j++];
            check = 1;
        }
    }
    store[l_size * 2] = '\0';
    MPI_Gather(store, l_size * 2, MPI_CHAR, ans, l_size * 2, MPI_CHAR, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("Modified String = %s\n", ans);
    }
    MPI_Finalize();
}