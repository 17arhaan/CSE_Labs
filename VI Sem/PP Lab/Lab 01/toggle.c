#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#define MAX 99

int main(int argc, char* argv[]) {
    int rank, size;
    char str[MAX];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("Enter a string: ");
        fgets(str, sizeof(str), stdin);
        str[strcspn(str, "\n")] = '\0';
    }
    
    MPI_Bcast(str, MAX, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank < strlen(str)) {
        printf("Process %d: Toggled %c", rank, str[rank]);
        
        if (isupper(str[rank])) {
            str[rank] = tolower(str[rank]);
        } else if (islower(str[rank])) {
            str[rank] = toupper(str[rank]);
        }

        printf(" -> %c\n", str[rank]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Modified String  ---> %s\n", str);
    }

    MPI_Finalize();
    return 0;
}
