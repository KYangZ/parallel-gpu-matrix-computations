#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include<sys/time.h>
#include "io.h"
#include "seq.cpp"

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_matrix(float* mat, int N){
    printf("[");
    for(int i=0;i< N*N;i++){
        printf("%.2f\t",mat[i]);
        if((i+1)%N==0){
            printf("\n");
        }
    }
    printf("]\n");
}

void delete_matrix(float* mat, int N){
    free(mat);
}
void gen_matrix(float* mat, int N){
    for (int i=0;i<N*N;i++){
		mat[i] = ((rand()%10)+1);
	}
}

__global__ void scale(float *g_odata, float *mat, int size, int row_index){
   int thid = threadIdx.x + blockIdx.x * blockDim.x;
   if (thid<size-row_index){
      // int start = row_index*size+row_index;
      // int end = (row_index+1)*size
      g_odata[row_index*size+(thid)*size+row_index] = 
      mat[row_index*size+(thid)*size+row_index]/mat[row_index*size+row_index];
   }
}
__global__ void reduce(float *g_idata, float *mat, int size, int row_index){
   int thid = threadIdx.x + blockIdx.x * blockDim.x;
   if (thid<(size - row_index-1) * (size - row_index)){
      int row = thid/(size - row_index);
      int col = thid%(size - row_index);
      int i = row_index*size + ((row+1)*size)+(col+row_index);
      mat[i] = mat[i] - g_idata[row_index*size + 
      ((row+1)*size)+row_index]*mat[(row_index*size)+col+row_index];
   }
}
//Given square matrix in flattened format with dim, compute LU decomp in parallel
void lud_parallel(float* l, float* u, float* matrix, int N){
    float *dev_matrix, *dev_l;

    HANDLE_ERROR(cudaMalloc((void**)&dev_matrix, N*N*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_l, N*N*sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(dev_matrix, matrix, N*N*sizeof(float), cudaMemcpyHostToDevice));


    for(int i=0;i<N;i++){
        int n = N-i;
        scale<<<(n+127)/128, 128>>>(dev_l, dev_matrix, N, i);
        reduce<<<((n*n)+127)/128, 128>>>(dev_l, dev_matrix, N, i);
    }
    

    HANDLE_ERROR(cudaMemcpy(l, dev_l, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(u, dev_matrix, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dev_l);
    cudaFree(dev_matrix);
}

void simple_verification(int N, float* mat){
    print_matrix(mat, N);

    float *l = new float[N*N];
    float *u = new float[N*N];

    lud_parallel(l, u, mat, N);

    print_matrix(l, N);
    print_matrix(u, N);

    free(l);
    free(u);
    free(mat);
}
void copy_matrix(float* mat1, float* mat2, int N){
    for(int i=0;i<N*N;i++){
        mat1[i] = mat2[i];
    }
}

int main(int argc, char** argv){

    int N = atoi(argv[1]);
    srand(42);
    float *mat = new float[N*N];
    float *mat_copy = new float[N*N];
    gen_matrix(mat, N);
    copy_matrix(mat_copy, mat, N);
    simple_verification(N, mat);
    simple_verification_seq(N, mat_copy);
    return 0;
}

