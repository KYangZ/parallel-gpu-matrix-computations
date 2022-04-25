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

__global__ void back_sub(double* M, int i, int j, double r, int M_rows, int M_cols) {
    M[j*M_cols+M_cols-1] -= r*M[i*M_cols+M_cols-1];
}

__global__ void scale_rows(double* M, int i, int j, double r, int M_rows, int M_cols) {
    M[i*M_cols+j] /= r;
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
        cudaThreadSynchronize();

        // apply elementary row operations to turn all elements below current pivot element to 0
        cuda_gaussian_elimination<<<(rows - i + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, rows, cols);
        cudaThreadSynchronize();
    }

    cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    //Scaling done non-parallel becuase it kept producing incorrect solutions
    for(int i = rows-1; i >=0; i--){
        double r = m[i*cols+i];
        for(int j = cols-1; j >= i; j--){
            m[i*cols+j] /= r;
          //  scale_rows<<<(rows - i + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, j, r, rows, cols);
        }
    }


   // printMatrix(m, rows, cols);
   // printf("\n");

    cudaMemcpy(d_m, m, rows * cols * sizeof(double), cudaMemcpyHostToDevice);

    //Gaussian Backward substitution
    for(int i = rows-1; i > 0; i--){
        for(int j = i-1; j >= 0; j-- ){
            double r = m[j * cols + i]; 
          //m[j*cols+cols-1] -= r*m[i*cols+cols-1];
            back_sub<<<(rows - i + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, j, r, rows, cols);
        }
    }

    // stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    double *solutions = (double*)malloc((rows)*sizeof(double));
    for(int i = 0; i < rows; i++){
        solutions[i] = m[i*cols+cols-1];
    }
    // print row echelon form of matrix
    printMatrix(solutions, rows, 1);
    writeResults(solutions, rows, 1, milliseconds);

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
      printf("Failed to create output file solved_sys.txt\n");
    }
    fprintf(fout, "rows: %d cols: %d \n", rows, cols);
    fprintf(fout, "execution time: %f milliseconds\n ", time);
    fprintf(fout, "Output matrix B of size [rows x 1]: \n");
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        fprintf(fout, "%f ", m[r * cols + c]);
      }
      fprintf(fout, "\n");
    }
    
    fprintf(fout, "\n \n \n \n \n ");
    fclose(fout);
}

int main(int argc, char* argv[]) {
    char* file; 
    file = argv[1];

    linearSolver(file);

    return 0;
}