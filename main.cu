#include<stdio.h>
#include<stdlib.h>


# define NUM_ITERS 10
# define FILL_PERCENT 10
# define SIZE 500

//generate spmv



//make csr



//make ecsr



//normal multiplication



//csr multiplication


//ecsr multiplication



//cost function


//comparison


int main(){
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//variable declarations
	int i,j,k,iter;
	int **M,*V;

	for(iter=0;iter<NUM_ITERS;iter++){
		//allocating SpMV

		printf("Generating Sparse Matrix ...");
		M = (int **)malloc(SIZE * sizeof(int *));
		for(i=0;i<SIZE;i++)
			M[i] = (int_*)malloc(SIZE * sizeof(int));

		for(i=0;i<SIZE;i++)
			for(j=0;j<SIZE;j++)
				M[i][j] = 0;

		int non_zero_ct = (int)(FILL_PERCENT * SIZE/100);
		for(i=0;i<non_zero_ct;i++){
			long long n = (long long)(rand()/RAND_MAX) * (long long)(SIZE)*(long long)(SIZE);
			long c = n % SIZE;
			long r = (int)(n / SIZE);
			M[r][c] = (rand() % 100) + 1;
		}

		printf("Done\n");

		printf("Generating Dense Vector...");
		V = (int *)malloc(SIZE * sizeof(int));

		for(i=0;i<SIZE;i++)
			v[i] = (rand() % 100) + 1;

		printf("Done\n");



	}
}