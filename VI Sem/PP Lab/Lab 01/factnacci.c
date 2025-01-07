#include <stdio.h>
#include "mpi.h"

int fib(int n) {
	int res;
    if (n == 0) {
        return 0;
    }
    else if (n == 1) {
        return 1;
    }
    else {
         res = fib(n-1) + fib(n-2);
    }
    return res;
}

int main(int argc , char* argv[]){
int rank , size;
MPI_Init(&argc , &argv);
MPI_Comm_rank(MPI_COMM_WORLD , &rank);
MPI_Comm_size(MPI_COMM_WORLD , &size);
int res;
if(rank % 2 == 0){
	res = 1;
	for(int i = 1;i<=rank ;i++){
		res = res*i;
	}
	printf("Process rank %d : Factorial of rank : %d\n", rank , res);
}
else{
	printf("Process rank %d : Fibonacci of rank : %d\n", rank , fib(rank));
}
MPI_Finalize();
return 0;
}
