#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv){
    int r, s, store_size, vow = 0, ans;
    char *s1 = (char *)malloc(1000 * sizeof(char));
    char *store;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    if (r == 0)
    {
        printf("Enter the Word : ");
        scanf("%s", s1);
        store_size = strlen(s1) / s;
    }
    MPI_Bcast(&store_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    store = (char *)malloc(store_size * sizeof(char));
    MPI_Scatter(s1, store_size, MPI_CHAR, store, store_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    for (int i = 0; i < store_size; ++i)
    {
        if (store[i] == 'a' || store[i] == 'e' || store[i] == 'i' || store[i] == 'o' || store[i] == 'u' || store[i] == 'A' || store[i] == 'E' || store[i] == 'I' || store[i] == 'O' || store[i] == 'U')
        {
            vow += 1;
        }
    }
    printf("%d vowels from rank %d\n", vow, r);
    MPI_Reduce(&vow, &ans, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (r == 0){
        printf("%d is the answer\n", ans);
    }
    MPI_Finalize();
}