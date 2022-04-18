#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

#define blocksize 8

// normalize elements not in diagonal
__global__ void nodiag_normalize(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		if (x == i && x!=y) {
			I[x*n + y] /= A[i*n + i];
			A[x*n + y] /= A[i*n + i];
		}
	}
}

// normalize elements in diagonal
__global__ void diag_normalize(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		if (x == y && x == i) {
			I[x*n + y] /= A[i*n + i];
			A[x*n + y] /= A[i*n + i];
		}
	}

}

// realization of parallel matrix inverse 
__global__ void gaussjordan(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		if (x != i) {
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i) {
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

// set first line in every loop to 0
__global__ void set_zero(double *A, double *I, int n, int i) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n) {
		if (x != i) {
			if (y == i) {
				A[x*n + y] = 0;
			}
		}
	}
}


/** start of local helper functions **/
void matrix_read(char filename[], double *L, int dimension) {
	FILE *fp;
	int row, col;

	fp = fopen(filename, "r");//open output file
	if (fp == NULL)//open failed
		return;

	for (row = 0; row < dimension; row++) {
		for (col = 0; col < dimension; col++)
		if (fscanf(fp, "%f,", &L[row * dimension + col]) == EOF) break;//read data

		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file
}

void savetofile(double *A, string s, int n, int h) {
	std::ofstream plik;
	plik.open(s);

	for (int j = 0; j<h; j++) {
		for (int i = 0; i<h; i++) {
			plik << A[j*n + i] << "\t";
		}
		plik << endl;
	}
	plik.close();
}
/** end of local helper functions **/

void invert(char filename[], int n) {
	// creating input
	double *iL = new double[n*n];
	double *L = new double[n*n];
	matrix_read(filename, L, n);

	// initialization 
	double *d_A, *I, *dI;
	float time;
	int ddsize = n*n*sizeof(double);

	// memory allocation    
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);

	cudaMalloc((void**)&d_A, ddsize);
	cudaMalloc((void**)&dI, ddsize);
	
	// identity matrix
	I = new double[n*n];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<n; j++) {
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}

	//copy data from CPU to GPU
	cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);

	//timer start
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// L^(-1)    
	for (int i = 0; i<n; i++) {
		nodiag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
		diag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
		gaussjordan <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
		set_zero <<< numBlocks, threadsPerBlock >>>(d_A, dI, n, i);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy data from GPU to CPU
	cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
	
	
	savetofile(iL, "outputs/inv.txt", n, n);

	// double *c = new double[n*n];
	// for (int i = 0; i<n; i++)  {}
	// 	for (int j = 0; j<n; j++) {
	// 		c[i*n+j] = 0;  //put the initial value to zero

	// 		for (int x = 0; x<n; x++) {
	// 			c[i*n + j] = c[i*n + j] + L[i*n+x] * iL[x*n + j];  //matrix multiplication
	// 		}
	// 	}
	// }
	// savetofile(c, "outputs/c.txt", n, n);

	// free variables 
	cudaFree(d_A);
	cudaFree(dI);
	delete[]I;
	delete[]L;
	delete[]iL;
}

int main(int argc, char *argv[]) {
	char* file;
	int n;

	file = argv[1];
	n = stoi(argv[2]);

	invert(file, n);
    return 0;
}
