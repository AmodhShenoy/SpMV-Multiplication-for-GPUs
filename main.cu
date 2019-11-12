#include<stdio.h>
#include<stdlib.h>
#include<time.h>

# define FILL_PERCENT 10
# define SIZE 250
# define BLOCK_SIZE 32


__global__ void spmvNormal( int *M, int *V, int *res){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int temp,i;
    if (idx < SIZE) {
    	temp = 0;
        //dot product for one row
        for (i = 0; i < SIZE; i++){
            temp += M[idx * SIZE + i] * V[i];
        }
         res[idx] = temp;
    } 
}

__global__ void spmvCSR(int *ro, int *ci, int *val, int *V, int *res_csr){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	int start, end;
	int dot;
	if(idx < SIZE){
		dot = 0;
		start = ro[idx];
		end = ro[idx + 1];
		for(i = start; i < end; i++){
			dot+= val[i] * V[ci[i]];
		}
	}
	res_csr[idx] = dot;
}



__global__ void spmvECSR(int *ro, int *dd,int *val, int *V, int* res_ecsr){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i,j;
	int start, end;
	int dot;
	if(idx < SIZE){
		start = ro[idx];
		end = ro[idx + 1];
		j=0;
		for(i = 0;i<=start;i++)
			j += dd[i];
		dot = val[start] * V[j];
		for(i = start+1; i < end; i++){
			dot += val[i] * V[j+dd[i]];
		}
	}
	res_ecsr[idx] = dot;
}


__global__ void spmvECSR_mod(int *ro, int *dd, int *val, int *V, int *res_ecsr){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i,j;
	int start, end;
	int dot;
	if(idx < SIZE){
		start = ro[idx];
		end = ro[idx + 1];
		j = dd[start];
		dot = val[start] * V[j];
		for(i = start+1; i < end; i++){
			dot+= val[i] * V[j+dd[i]];
		}
	}
	res_ecsr[idx] = dot;
}

int **M,*V;
int *ro,*ci,*val,*dd;
int *ro_gpu,*ci_gpu,*val_gpu,*dd_gpu,*V_gpu,*M_gpu;
int *res_csr,*res_ecsr,*res,*res_ecsr_mod;
int *res_csr_gpu, *res_ecsr_gpu,*res_gpu,*res_ecsr_mod_gpu;

int main(){
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//variable declarations
	int i,j;
	cudaEvent_t start,stop,start_csr,stop_csr,start_ecsr,stop_ecsr,start_ecsr_mod,stop_ecsr_mod;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_csr);
	cudaEventCreate(&stop_csr);
	cudaEventCreate(&start_ecsr);
	cudaEventCreate(&stop_ecsr);
	cudaEventCreate(&start_ecsr_mod);
	cudaEventCreate(&stop_ecsr_mod);
	
	float time_csr=0,time_ecsr=0,time_normal=0,time_ecsr_mod=0;

	// Define CudaError
	cudaError_t err;
	
	//initiallizing result vectors
	res = (int *)malloc(SIZE*sizeof(int));
	res_csr = (int * )malloc(SIZE * sizeof(int));
	res_ecsr = (int *)malloc(SIZE * sizeof(int));
	res_ecsr_mod = (int *)malloc(SIZE * sizeof(int));

	//allocating SpMV
	printf("Generating Sparse Matrix ...");
	M = (int **)malloc(SIZE * sizeof(int *));
	for(i=0;i<SIZE;i++)
		M[i] = (int *)malloc(SIZE * sizeof(int));

	for(i=0;i<SIZE;i++)
		for(j=0;j<SIZE;j++)
			M[i][j] = 0;

	int non_zero_ct = (int)(FILL_PERCENT * SIZE/100);
	// printf("%d\n",non_zero_ct);
	// int non_zero_ct = 2;
	for(i=0;i<non_zero_ct;i++){
		long long n = (long long)((rand() % 100) * SIZE* SIZE)/100;
		long c = n % SIZE;
		long r = (int)(n / SIZE);
		M[r][c] = (rand() % 100) + 1;
	}

	printf("Done\n");

	printf("Generating Dense Vector...");
	V = (int *)malloc(SIZE * sizeof(int));
	for(i=0;i<SIZE;i++)
		V[i] = (rand() % 100) + 1;

	printf("Done\n");


	//Building CSR and ECSR rep of SpM
	printf("Building CSR vectors and Distance Difference vector...");
	int cct = 0;
	int prev = 0;
	ro = (int *)malloc((SIZE + 1)*sizeof(int));
	ci = (int *)malloc(non_zero_ct *2* sizeof(int));
	val = (int *)malloc(non_zero_ct *2* sizeof(int));
	ro[0] = 0;



	dd = (int *)malloc(non_zero_ct * 2 * sizeof(int)/2);

	/*for(i=0;i<SIZE;i++)
		for(j=0;j<SIZE;j++)
			printf("%d ",M[i][j]);
		printf("\n");*/
	for(i=0;i<SIZE;i++){
		int flag = 0;
		for(j=0;j<SIZE;j++){
			//printf("%d ",M[i][j]);
			if(M[i][j]!=0){					
				while(j-prev>255){
					printf("abc ");
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
		//printf("\n");
		ro[i+1] = cct;
	}
	printf("Done\n");

	// for(i=0;i<ro[SIZE];i++){
	// 	printf("%d %d\n",ci[i],val[i]);
	// }
	//Setup memory on GPU
	cudaMalloc((void **)&M_gpu,(SIZE * sizeof(int))*(SIZE));
	cudaMalloc((void **)&ro_gpu, (SIZE + 1)*sizeof(int));
	cudaMalloc((void **)&ci_gpu, (non_zero_ct * 2 * sizeof(int)));
	cudaMalloc((void **)&val_gpu, (non_zero_ct * 2 * sizeof(int)));
	cudaMalloc((void **)&dd_gpu, (non_zero_ct * 2 * sizeof(int))/2);
	cudaMalloc((void **)&V_gpu, (SIZE * sizeof(int)));
	cudaMalloc((void **)&res_gpu, (SIZE * sizeof(int)));
	cudaMalloc((void **)&res_csr_gpu, (SIZE * sizeof(int)));
	cudaMalloc((void **)&res_ecsr_gpu, (SIZE * sizeof(int)));
	cudaMalloc((void **)&res_ecsr_mod_gpu, (SIZE * sizeof(int)));

	//printf("Done cuda malloc\n");

	//transfer to device
	cudaMemcpy(M_gpu, M, (SIZE * SIZE * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(ro_gpu, ro, (SIZE +1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ci_gpu, ci , (non_zero_ct * 2 * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(val_gpu, val, (non_zero_ct * 2 * sizeof(int)),cudaMemcpyHostToDevice);
	cudaMemcpy(dd_gpu, dd, (non_zero_ct * 2 * sizeof(int)/2), cudaMemcpyHostToDevice);
	cudaMemcpy(V_gpu, V, (SIZE * sizeof(int)), cudaMemcpyHostToDevice);

	//printf("Done transferring to device\n");

	//setting CUDA parameters
	int nb = ceil(SIZE/BLOCK_SIZE);
	int nt = BLOCK_SIZE;
	// dim3 GridDim,BlockDim;
	// BlockDim.x = nb;
	// BlockDim.y=1;
	// GridDim.x = BLOCK_SIZE;
	// GridDim.y = BLOCK_SIZE;

	//Starting Normal Multiplication
	printf("\n\nStarting Normal Multiplication...");
	//clock_t start,end;
	//start = clock();

	cudaEventRecord(start);
	spmvNormal<<< nb,nt >>>(M_gpu,V_gpu,res_gpu);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//end = clock();
	//time_normal += end - start;
	cudaEventElapsedTime(&time_normal, start, stop);

	//Checking for CUDA errors
	err = cudaGetLastError();
	if(err!=cudaSuccess){
		printf("ERROR: %s\n",cudaGetErrorString(err));
	}
	printf("Done\n");

	//Transfer result back to memory
	cudaMemcpy(res, res_gpu, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);
		



	//Starting CSR Multiplication
	printf("\n\nStarting CSR Multiplication...");
	// clock_t start_csr,end_csr;
	// start_csr = clock();

	cudaEventRecord(start_csr);
	spmvCSR<<< nb,nt>>>(ro_gpu,dd_gpu,val_gpu,V_gpu,res_csr_gpu);
	cudaEventRecord(stop_csr);
	cudaEventSynchronize(stop_csr);
	cudaEventElapsedTime(&time_csr,start_csr,stop_csr);
	//end_csr = clock();
	// time_csr += end_csr - start_csr;

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
	// clock_t start_ecsr,end_ecsr;
	// start_ecsr = clock();

	cudaEventRecord(start_ecsr);
	spmvECSR<<< nb,nt>>>(ro_gpu,dd_gpu,val_gpu,V_gpu,res_ecsr_gpu);
	cudaEventRecord(stop_ecsr);
	cudaEventSynchronize(stop_ecsr);
	cudaEventElapsedTime(&time_ecsr, start_ecsr,stop_ecsr);
	// end_ecsr = clock();
	// time_ecsr += end_ecsr - start_ecsr;

	//Checking for CUDA errors
	err = cudaGetLastError();
	if(err!=cudaSuccess){
		printf("ERROR: %s\n",cudaGetErrorString(err));
		exit(0);
	}
	printf("Done\n");

	//Transfer result back to memory
	cudaMemcpy(res_ecsr, res_ecsr_gpu, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);		



	printf("\n\nStarting ECSR(modified) Multiplication...");
	// clock_t start_ecsr,end_ecsr;
	// start_ecsr = clock();

	cudaEventRecord(start_ecsr_mod);
	spmvECSR_mod<<< nb,nt>>>(ro_gpu,dd_gpu,val_gpu,V_gpu,res_ecsr_mod_gpu);
	cudaEventRecord(stop_ecsr_mod);
	cudaEventSynchronize(stop_ecsr_mod);
	cudaEventElapsedTime(&time_ecsr_mod, start_ecsr_mod,stop_ecsr_mod);
	// end_ecsr = clock();
	// time_ecsr += end_ecsr - start_ecsr;

	//Checking for CUDA errors
	err = cudaGetLastError();
	if(err!=cudaSuccess){
		printf("ERROR: %s\n",cudaGetErrorString(err));
		exit(0);
	}
	printf("Done\n\n");

	//Transfer result back to memory
	cudaMemcpy(res_ecsr_mod, res_ecsr_mod_gpu, (SIZE * sizeof(int)), cudaMemcpyDeviceToHost);	


	//free memory
	for(i=0;i<SIZE;i++)
		free(M[i]);

	free(M);
	free(V);
	free(ro);
	free(ci);
	free(val);
	free(dd);
	free(res);
	free(res_csr);
	free(res_ecsr);
	cudaFree(ro_gpu);
	cudaFree(ci_gpu);
	cudaFree(val_gpu);	
	cudaFree(dd_gpu);
	cudaFree(M_gpu);
	cudaFree(V_gpu);
	cudaFree(res_gpu);
	cudaFree(res_csr_gpu);
	cudaFree(res_ecsr_gpu);
	printf("===============================================================================\n");

	printf("Average time taken for normal multiplication:%lf\n",time_normal);
	printf("---------------------------------------------------------------\n");
	printf("Average time taken for CSR multiplication:%lf\n",time_csr);
	printf("CSR multiplication runs %lf times faster than normal multiplication\n",time_normal/time_csr);
	printf("---------------------------------------------------------------\n");
	printf("Average time taken for ECSR multiplication:%lf\n",time_ecsr);
	printf("ECSR multiplication runs %lf times faster than normal multiplication\n",time_normal/time_ecsr);
	printf("---------------------------------------------------------------\n");
	printf("Average time taken for ECSR(modified) multiplication:%lf\n",time_ecsr_mod);
	printf("ECSR(modfied) multiplication runs %lf times faster than normal multiplication\n",time_normal/time_ecsr_mod);
	printf("---------------------------------------------------------------\n");
	printf("It is seen that time taken for ECSR multiplication is %lf times that for ECSR(modified) multiplication\n",time_ecsr/time_ecsr_mod);

	return 0;
}
