#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 2147483647
#define DELIM " \t\n"
#define THREADS_PER_BLOCK 1024

double* readMatrix(FILE* fp, int* nrows, int* ncols);
void printMatrix(double* m, int rows, int cols);
void writeResults(double* m, int rows, int cols, float time);

__global__ void cuda_swap_rows(double* M, int i, int j, int M_cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    // swap element col in rows i and j
    if (col < M_cols) {
        double temp = M[i * M_cols + col];
        M[i * M_cols + col] = M[j * M_cols + col];
        M[j * M_cols + col] = temp;
    }

}

__global__ void cuda_row_subtract(double* M, int i, int j, double r, int M_cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    // for all k, m[j * cols + k] -= r * m[i * cols + k];
    if (col < M_cols) {
        M[j * M_cols + col] -= r * M[i * M_cols + col];
    }
}

__global__ void cuda_gaussian_elimination(double* M, int i, int M_rows, int M_cols) {
    int j = threadIdx.x + blockIdx.x * blockDim.x + i + 1;
    if (j > i && j < M_rows && M[i * M_cols + i] != 0) {
        double r = M[j * M_cols + i] / M[i * M_cols + i];
        cuda_row_subtract<<<(M_cols + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(M, i, j, r, M_cols);
    }
}

/*
__global__ void forward_elimination(double* M, int i, int M_rows, int M_cols) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(j>= (M_rows - (1+i))*(M_cols-i))return;

    int coltoelim = i +j%(M_cols-i);
    int rowtoelim = 1+i+j/(M_cols-i);

    int elim = M_cols*rowtoelim + coltoelim;

    double aij = M[i+rowtoelim*M_cols]/M[i+M_cols*i];

    M[elim] -= aij*M[i*M_cols+coltoelim];

}
*/

__global__ void reverse_elimination(double* M, int i, int M_rows, int M_cols) {
    int j = threadIdx.x;
    int cols = M_cols - 2 - i;
    int ind = i*M_cols + i + 1;
    int cnt = cols%2;
    for(int k = cols/2; k > 0; k/=2){
        if(j>=k){
            return;
        }
        M[ind + j] += (M[ind+j+k+cnt]);
        M[ind+j+k+cnt] = 0;
        if(cnt == 1){
            k++;
        }
        cnt=k%2;
        __syncthreads();
    }
    int varel = (-1) + M_cols*(i+1);
    int diael = i + i*M_cols;

    if(diael + 1 != varel){
        M[varel] -= M[diael + 1];
        M[diael + 1] = 0.0;
    }

    M[varel] /= M[diael];
    M[diael] = 1.0;

}

__global__ void fix_cols(double* M, int i, int M_rows, int M_cols) {
    int j = threadIdx.x;
    if(fabs(M[i+(M_cols*j)]) != 1.0){
        M[i+(M_cols*j)] *= M[M_cols*(1+i) - 1];
    }
}

__global__ void fix_zeroes(double* M, int i, int M_rows, int M_cols) {
    if(fabs(M[i*M_cols+i]) <= 1e-4){
        int j = i;
        for(; j<M_rows; j++){
            if(fabs(M[j*M_cols+i]) <= 1e-4)break;
        }
        int thread = threadIdx.x + blockIdx.x * blockDim.x;
        if(thread+i>=M_cols)return;
        M[i*M_cols+i+thread] += M[j*M_cols+i+thread];
    }
}

double linearSolver(char file[]) {
    // open and read input matrix
    FILE* f = fopen(file, "r");
    if (!f) {
        printf("Failed to open %s", file);
        exit(-1);
    }

    int rows, cols;
    double* m = readMatrix(f, &rows, &cols);

    double* d_m;
    cudaMalloc(&d_m, rows * cols * sizeof(double));
    cudaMemcpy(d_m, m, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start);

    // gaussian forward elimination
    for (int i = 0; i < rows; i++) {
        // modified partial pivoting: if current pivot element is zero, swap with row such that pivot element is non zero
        if (m[i * cols + i] == 0) {
            for (int j = i + 1; j < rows; j++) {
                if (m[j * cols + i] != 0) {
                    cuda_swap_rows<<<(cols + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, j, cols);
                    break;
                }
            }
        }

        // apply elementary row operations to turn all elements below current pivot element to 0
        cuda_gaussian_elimination<<<(rows - i + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, rows, cols);
    }


    cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    printMatrix(m, rows, cols);

    //Gaussian Backward substitution
    for(int i = rows-1; i >= 0; i--){
        reverse_elimination<<<1, cols>>>(d_m, i, rows, cols);
        printf("reverse step \n");
        cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
        printMatrix(m, rows, cols);

        fix_cols<<<1, rows>>>(d_m, i, rows, cols);
        cudaThreadSynchronize();
        printf(" fix step\n");
        cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
        printMatrix(m, rows, cols);
        
    }

    /*
    reverse_elimination<<<1, cols>>>(d_m, 0, rows, cols);
    printf("reverse step \n");
    cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    printMatrix(m, rows, cols);
    */
    
    // stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // print row echelon form of matrix
    // printMatrix(m, rows, cols);
    writeResults(m, rows, cols, milliseconds);

    // close files and free memory
    fclose(f);
    free(m);
    cudaFree(d_m);

    return 0;
}

double* readMatrix(FILE* fp, int* nrows, int* ncols) {
    // read in a matrix from file input
    char* line = (char*) malloc(MAX_LINE_LEN * sizeof(char));
    char* token = NULL;

    double* m;

    fgets(line, MAX_LINE_LEN, fp);
    token = strtok(line, DELIM);
    *nrows = atoi(token);
    token = strtok(NULL, DELIM);
    *ncols = atoi(token);

    m = (double*) malloc((*nrows) * (*ncols) * sizeof(double));

    for (int r = 0; r < *nrows; r++) {
        if (fgets(line, MAX_LINE_LEN, fp) == NULL) {
            printf("input matrix contains wrong dimensions\n");
        }

        for (int c = 0; c < *ncols; c++) {
            if (c == 0) {
                token = strtok(line, DELIM);
            } else {
                token = strtok(NULL, DELIM);
            }
            m[r * (*ncols) + c] = atof(token);
        }
    }

    return m;
}

void printMatrix(double* m, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        printf("%f ", m[r * cols + c]);
      }
      printf("\n");
    }
}

void writeResults(double* m, int rows, int cols, float time) {
    FILE* fout = fopen("solved_sys.txt", "a");
    if (!fout) {
      printf("Failed to create output file det_out.txt\n");
    }
    fprintf(fout, "%d %d \n", rows, cols);
    fprintf(fout, "Output matrix: \n");
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        fprintf(fout, "%f ", m[r * cols + c]);
      }
      fprintf(fout, "\n");
    }
    
    fprintf(fout, "execution time: %f milliseconds\n", time);
    fclose(fout);
}

int main(int argc, char* argv[]) {
    char* file, output;
    int in_rows, in_cols;

    file = argv[1];
    /*
    output = argv[2];
    in_rows = stoi(argv[3]);
    in_cols = stoi(argv[4]);
    */

    linearSolver(file); /*, output, in_rows, in_cols*/

    return 0;
}