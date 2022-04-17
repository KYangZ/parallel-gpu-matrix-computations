#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 4096
#define DELIM " \t\n"
#define THREADS_PER_BLOCK 128

double* readMatrix(FILE* fp, int* nrows, int* ncols);
void printMatrix(double* m, int rows, int cols);
void writeResults(double* m, int rows, int cols, double det);

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

double determinant(char file[]) {
    // open and read input matrix
    FILE* f = fopen(file, "r");
    if (!f) {
        printf("Failed to open %s", file);
        return 0.0;
    }

    int rows, cols;
    double* m = readMatrix(f, &rows, &cols);

    // prints out matrices (for testing)
    // printMatrix(m, rows, cols);

    // dimension check - determinant is only defined on square matrices
    if (rows != cols) {
        printf("undefined determinant: non-square matrix\n");
        exit(-1);
    }

    // initialization
    double* d_m;
    cudaMalloc(&d_m, rows * cols * sizeof(double));
    double det = 1;

    // gaussian elimination
    for (int i = 0; i < rows - 1; i++) {
        // modified partial pivoting: if current pivot element is zero, swap with row such that pivot element is non zero
        if (m[i * cols + i] == 0) {
            for (int j = i + 1; j < rows; j++) {
                if (m[j * cols + i] != 0) {
                    cudaMemcpy(d_m, m, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
                    cuda_swap_rows<<<(cols + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, j, cols);
                    cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
                    det *= -1;
                    break;
                }
            }
        }

        // apply elementary row operations to turn all elements below current pivot element to 0
        if (m[i * cols + i] != 0) {
            for (int j = i + 1; j < rows; j++) {
                double r = m[j * cols + i] / m[i * cols + i];
                cudaMemcpy(d_m, m, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
                cuda_row_subtract<<<(cols + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_m, i, j, r, cols);
                cudaMemcpy(m, d_m, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
            }
        }

        // multiply elements on main diagonal to obtain determinant
        det *= m[i * cols + i];
    }

    // multiply the bottom right element on the main diagonal (not covered in the above for loop)
    det *= m[(rows - 1) * cols + (rows - 1)];

    // print row echelon form of matrix
    // printMatrix(m, rows, cols);
    writeResults(m, rows, cols, det);

    // close files and free memory
    fclose(f);
    free(m);
    cudaFree(d_m);

    return det;
}

double* readMatrix(FILE* fp, int* nrows, int* ncols) {
    // read in a matrix from file input
    char* line = (char*) malloc(MAX_LINE_LEN * sizeof(char));
    char* token = NULL;
    size_t len = 0;

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
            exit(-1);
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

void writeResults(double* m, int rows, int cols, double det) {
    FILE* fout = fopen("out.txt", "a");
    if (!fout) {
      printf("Failed to create output file out.txt\n");
    }
    fprintf(fout, "%d %d \n", rows, cols);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        fprintf(fout, "%f ", m[r * cols + c]);
      }
      fprintf(fout, "\n");
    }
    fprintf(fout, "determinant: %f\n", det);
    fclose(fout);
}

int main(int argc, char* argv[]) {
    char* file;
    file = argv[1];
    printf("determinant: %f\n", determinant(file));
    return 0;
}
