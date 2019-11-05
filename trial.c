#include<stdio.h>
#include<stdlib.h>


	// void allocate(int **a){
	// 	int i,j;
	// 	a = (int **)malloc(size * sizeof(int *));
	// 	for(i=0;i<size;i++)
	// 		a[i] = (int *)malloc(size * sizeof(int));
	// 	for(i=0;i<size;i++)
	// 		for(j=0;j<size;j++)
	// 			a[i][j]=i+j;
	// }

int main(){
	int i,j;
	int size = 5;
	int **a;
	a= (int **)malloc(size * sizeof(int *));
	for(i=0;i<size;i++)
		a[i] = (int *)malloc(size * sizeof(int));
	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
			a[i][j]=i+j;
	

	printf("Done allocating");
	for(i=0;i<size;i++){
		for(j=0;j<size;j++)
			printf("%d ",a[i][j]);
		printf("\n");
	}
}