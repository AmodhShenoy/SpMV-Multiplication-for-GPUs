#include<stdio.h>
#include<stdlib.h>
#include<time.h>


# define NUM_ITERS 10
# define FILL_PERCENT 10
# define SIZE 500
# define BLOCK_SIZE 32


__global__ void spmvNormal( int *M, int *V, int *res){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int temp,i;
    if (idx < SIZE) {
    	temp = 0;
        //dot product for one row
        for (i = 0; i < SIZE; i++){
            temp += M[idx * SIZE + i] * V[i];
        }
         res[idx] = temp;
    } 
    __syncthreads();
}

__global__ void spmvCSR(int *ro, int *ci,int *val, int *V, int* res_csr){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	int start, end;
	int dot;
	if(idx < SIZE){
		dot = 0;
		start = ro[row];
		end = ro[row + 1];
		for(i = start; i < end; i++){
			dot+= val[i] * V[ci[i]];
		}
	}
	res_csr[idx] = dot;
}


__global__ void spmvECSR(int *ro, int *dd, int *val, int *V, int *res_ecsr){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i,j;
	int start, end;
	int dot;
	if(idx < SIZE){
		dot = 0;
		start = ro[row];
		end = ro[row + 1];
		j = dd[start];
		res_ecsr[idx] = val[start] * V[j];
		for(i = start+1; i < end; i++){
			dot+= val[i] * V[j+dd[i]];
		}
	}
	res_csr[idx] = dot;
}

int main(){
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//variable declarations
	int i,j,k,iter;
	int **M,*V;
	int *ro,*ci,*val,*dd;
	int *ro_gpu,*ci_gpu,*val_gpu,*dd_gpu,*V_gpu,*M_gpu;
	int *res_csr,*res_ecsr,*res;
	int *res_csr_gpu, *res_ecsr_gpu,*res_gpu;
	double total_csr=0,total_ecsr=0,total_normal=0;

	// Define CudaError
    cudaError_t err;

	for(iter=0;iter<NUM_ITERS;iter++){
		printf("==================================TRIAL NO %d==================================\n",iter+1);
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


		//Building CSR and ECSR rep of SpM
		printf("Building CSR vectors and Distance Difference vector...");
		int cct = 0;
		int prev = 0;
		ro = (int *)malloc((SIZE + 1)*sizeof(int));
		ci = (int *)malloc(non_zero_ct *1.5* sizeof(int));
		val = (int *)malloc(non_zero_ct *1.5* sizeof(int));
		ro[0] = 0;

		dd = (int *)malloc(non_zero_ct *1.5* sizeof(int)/2);

		for(i=0;i<SIZE;i++){
			int flag = 0;
			for(j=0;j<SIZE;j++){
				if(M[i][j]!=0){					
					while(j-prev<255){
						ci[cct] = prev + 255;
						val[cct] = 0;
						dd[cct] = 255;
						prev = prev + 255;
						cct++;
					}

					ci[cct] = j;
					val[cct] = M[i][j];
					if(flag==0){
						dd[cct] = j;
						flag++;
					}
					else
						dd[cct] = j - prev;
					prev = j;
					cct++;
				}
			}
			ro[i+1] = cct;
		}
		printf("Done\n");

		//Setup memory on GPU
		cudaMalloc((void **)&M_gpu,(SIZE * sizeof(int))*(SIZE));
		cudaMalloc((void **)&ro_gpu, (SIZE + 1)*sizeof(int));
		cudaMalloc((void **)&ci_gpu, (non_zero_ct * 1.5 * sizeof(int)));
		cudaMalloc((void **)&val_gpu, (non_zero_ct * 1.5 * sizeof(int)));
		cudaMalloc((void **)&dd_gpu, (non_zero_ct * 1.5 * sizeof(int))/2);
		cudaMalloc((void **)&V_gpu, (SIZE * sizeof(int)));
		cudaMalloc((void **)&res_csr_gpu, (SIZE * sizeof(int)));
		cudaMalloc((void **)&res_ecsr_gpu, (SIZE * sizeof(int)));

		//transfer to device
		cudaMemcpy(M_gpu, M, (SIZE * SIZE * sizeof(int)), cudaMemcpyHostToDevice);
		cudaMemcpy(ro_gpu, ro, (SIZE +1)*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(ci_gpu, ci , (non_zero_ct * 1.5 * sizeof(int)), cudaMemcpyHostToDevice);
		cudaMemcpy(val_gpu, val, (non_zero_ct * 1.5 * sizeof(int)),cudaMemcpyHostToDevice);
		cudaMemcpy(dd_gpu, dd, (non_zero_ct * 1.5 * sizeof(int)/2), cudaMemcpyHostToDevice);
		cudaMemcpy(V_gpu, V, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);

		//setting CUDA parameters
		int nb = ceil(SIZE/BLOCK_SIZE);
		int nt = BLOCK_SIZE;

		//Starting Normal Multiplication
		printf("\n\nStarting Normal Multiplication...");
		clock_t start,end;
		start = clock();

		spmvNormal<<< nb,nt >>>(M_gpu,V_gpu,res_gpu);

		end = clock();
		total_normal += end - start;

		//Checking for CUDA errors
		err = cudaGetLastError();
		if(err!=cudaSuccess){
			printf("ERROR: %s\n",cudaGetErrorString(err));
			exit(0);
		}
		printf("Done\n");

		//Transfer result back to memory
		cudaMemcpy(res, res_gpu, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);


		//Starting CSR Multiplication
		printf("\n\nStarting CSR Multiplication...");
		clock_t start_csr,end_csr;
		start_csr = clock();

		spmvCSR<<< nb,nt>>>(ro_gpu,ci_gpu,val_gpu,V_gpu,res_csr_gpu);

		end_csr = clock();
		total_csr += end_csr - start_csr;

		//Checking for CUDA errors
		err = cudaGetLastError();
		if(err!=cudaSuccess){
			printf("ERROR: %s\n",cudaGetErrorString(err));
			exit(0);
		}
		printf("Done\n");

		//Transfer result back to memory
		cudaMemcpy(res_csr, res_csr_gpu, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);


		//Starting ECSR Multiplication
		printf("\n\nStarting ECSR Multiplication...");
		clock_t start_ecsr,end_ecsr;
		start_ecsr = clock();

		spmvECSR<<< nb,nt>>>(ro_gpu,dd_gpu,val_gpu,V_gpu,res_ecsr_gpu);

		end_ecsr = clock();
		total_ecsr += end_ecsr - start_ecsr;

		//Checking for CUDA errors
		err = cudaGetLastError();
		if(err!=cudaSuccess){
			printf("ERROR: %s\n",cudaGetErrorString(err));
			exit(0);
		}
		printf("Done\n");

		//Transfer result back to memory
		cudaMemcpy(res_csr, res_csr_gpu, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);		

		//free memory
		for(i=0;i<SIZE;i++)
			free(M[i]);
		free(M);
		free(V);
		free(ro);
		free(ci);
		free(val);
		free(dd);
		free(res_csr);
		free(res_ecsr);
		cudaFree(ro_gpu);
		cudaFree(ci_gpu);
		cudaFree(val_gpu);
		cudaFree(dd_gpu);
		cudaFree(V_gpu);
		cudaFree(res_csr_gpu);
		cudaFree(res_esct_gpu);
		printf("===============================================================================\n");
	}

	double avg = total_normal/NUM_ITERS;
	double avg_csr = total_csr/NUM_ITERS;
	double avg_ecsr = total_ecsr/NUM_ITERS;
	printf("Average time taken for normal multiplication:%lf\n",avg);
	printf("---------------------------------------------------------------");
	printf("Average time taken for CSR multiplication:%lf\n",avg_csr);
	printf("CSR multiplication runs %lf times faster than normal multiplication\n",avg/avg_csr);
	printf("---------------------------------------------------------------");
	printf("Average time taken for ECSR multiplication:%lf\n",avg_ecsr);
	printf("ECSR multiplication runs %lf times faster than normal multiplication\n",avg/avg_ecsr);
	printf("---------------------------------------------------------------");
	printf("It is seen that time taken for CSR multiplication is %lf times that for ECSR multiplication\n",avg_csr/avg_ecsr);

	return 0;
}