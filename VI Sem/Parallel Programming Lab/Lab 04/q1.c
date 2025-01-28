#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

long long factorial(int n) {
    long long facct = 1;
    for (int i = 1; i <= n; i++) {
        facct *= i;
    }
    return facct;
}
int main(int argc, char* argv[]) {
    int rank, size, N, tag = 0;
    long long local_sum = 0, total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    // Check for Negative Value Tag Error
    if (rank == 0) {
        printf("Enter N: ");
        scanf("%d", &N);

        if (N < tag) {
            printf("Error: N must be a positive integer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Check for MPI_COMM_NULL Error
    MPI_Comm null_comm = MPI_COMM_WORLD;
    int comm_rank, comm_size;
    int error_code = MPI_Comm_rank(null_comm, &comm_rank);
    if (error_code != MPI_SUCCESS) {
        char error_string[256];
        int length_of_error_string;
        MPI_Error_string(error_code, error_string, &length_of_error_string);
        printf("Null Communicator Error !! %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
    int start = (rank * N) / size + 1;
    int end = ((rank + 1) * N) / size;
    for (int i = start; i <= end; i++) {
        local_sum += factorial(i);
    }
    // // Error handling with send/receive
    // if (rank == 0) {
    //     long long buffer = local_sum; 
    //     int send_error = MPI_Send(&buffer, 1, MPI_LONG_LONG, 1, tag, MPI_COMM_WORLD);

    //     if (send_error != MPI_SUCCESS) {
    //         printf("Error in MPI_Send: Error code %d\n", send_error);
    //         MPI_Abort(MPI_COMM_WORLD, 1);
    //     }
    // }

    // long long received_sum = 0;
    // if (rank == 1) {
    //     int recv_error = MPI_Recv(&received_sum, 1, MPI_LONG_LONG, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     if (recv_error != MPI_SUCCESS) {
    //         printf("Error in MPI_Recv: Error code %d\n", recv_error);
    //         MPI_Abort(MPI_COMM_WORLD, 1);
    //     }

    //     printf("Received sum: %lld\n", received_sum);
    // }
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total sum: %lld\n", total_sum);
    }
    MPI_Finalize();
    return 0;
}