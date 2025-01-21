#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
    int rank, size, *ori, *store, m, n;
    float *results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Enter the value of M: ");
        scanf("%d", &m);
        n = size;

        ori = (int *)malloc((m * n) * sizeof(int));

        printf("Enter %d elements: \n", m * n);
        for (int i = 0; i < m * n; ++i)
        {
            scanf("%d", &ori[i]);
        }
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    store = (int *)malloc(m * sizeof(int));

    MPI_Scatter(ori, m, MPI_INT, store, m, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < m; ++i)
    {
        if (rank % 2 == 0)
        {
            store[i] = (int)pow(store[i], 2);
        }
        else
        {
            store[i] = (int)pow(store[i], 3);
        }
    }

    if (rank == 0)
    {
        results = (float *)malloc(n * m * sizeof(float));
    }

    MPI_Gather(store, m, MPI_INT, results, m, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Results:\n");
        for (int i = 0; i < n * m; i++)
        {
            printf("%f ", (float)results[i]);
            if ((i + 1) % m == 0)
            {
                printf("\n");
            }
        }
        free(ori);
        free(results);
    }

    free(store);

    MPI_Finalize();
    return 0;
}
