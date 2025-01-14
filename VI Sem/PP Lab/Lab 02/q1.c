#include "mpi.h"
#include <stdio.h>
#include <ctype.h>

#define MAX 99

void toggle_case(char* msg) {
    for (int i = 0; msg[i] != '\0'; i++) {
        if (isupper(msg[i])) {
            msg[i] = tolower(msg[i]);
        } else if (islower(msg[i])) {
            msg[i] = toupper(msg[i]);
        }
    }
}

int main(int argc , char*argv[]) {
    int rank, size;
    char msg[MAX];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
        printf("Enter Message: ");
        fgets(msg, MAX, stdin);
        MPI_Ssend(msg, MAX, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        printf("Process 0 sent: %s\n", msg);

        MPI_Recv(msg, MAX, MPI_CHAR, 1, 2, MPI_COMM_WORLD, &status);
        printf("Process 0 received back: %s\n", msg);
    } 
    else if (rank == 1) {
        MPI_Recv(msg, MAX, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);
        printf("Process 1 received: %s\n", msg);
        toggle_case(msg);
        MPI_Ssend(msg, MAX, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
        printf("Process 1 sent modified message: %s\n", msg);
    }

    MPI_Finalize();
    return 0;
}
