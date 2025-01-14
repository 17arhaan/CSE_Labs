#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc , char* argv[]){
int rank , size;
char str[5] = "HELLO";
MPI_Init(&argc , &argv);
MPI_Comm_rank(MPI_COMM_WORLD , &rank);
MPI_Comm_size(MPI_COMM_WORLD , &size);
printf("Process rank %d : toggle %c\n",rank , str[rank]);
str[rank] = str[rank] + 32;
for(int i = 0;i<5;i++){
	printf("%c",str[i]);
}
printf("\n");
MPI_Finalize();
return 0;
}
