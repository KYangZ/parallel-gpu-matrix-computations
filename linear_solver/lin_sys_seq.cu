#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LEN 2147483647
#define DELIM " \t\n"

double* readMatrix(FILE* fp, int* nrows, int* ncols);
void printMatrix(double* m, int rows, int cols);
void writeResults(double* m, int rows, int cols, float time);

double linearSolver(char file[]) {
    // open and read input matrix
    FILE* f = fopen(file, "r");
    if (!f) {
        printf("Failed to open %s \n", file);
        return 0.0;
    }

    int rows, cols;
    double* m = readMatrix(f, &rows, &cols);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start);

    
    // prints out matrices (for testing)
    // printMatrix(m, rows, cols);

    // gaussian elimination
    for (int i = 0; i < rows - 1; i++) {
        // modified partial pivoting: if current pivot element is zero, swap with row such that pivot element is non zero
        if (m[i * cols + i] == 0) {
            for (int j = i + 1; j < rows; j++) {
                if (m[j * cols + i] != 0) {
                    for (int k = 0; k < cols; k++) {
                        double temp;
                        temp = m[i * cols + k];
                        m[i * cols + k] = m[j * cols + k];
                        m[j * cols + k] = temp;
                    }
                    break;
                }
            }
        }

        // apply elementary row operations to turn all elements below current pivot element to 0
        if (m[i * cols + i] != 0) {
            for (int j = i + 1; j < rows; j++) {
                double r = m[j * cols + i] / m[i * cols + i];
                for (int k = 0; k < cols; k++) {
                    m[j * cols + k] -= r * m[i * cols + k];
                }
            }
        }

    }
   
    //printMatrix(m, rows, cols);

    for(int i = rows-1; i >=0; i--){
        double r = m[i*cols+i];
        for(int j = cols-1; j >= i; j--){
            m[i*cols+j] /= r;
        }
    }

    //Gaussian Backward substitution
    for(int i = rows-1; i > 0; i--){
        for(int j = i-1; j >= 0; j-- ){
            double r = m[j * cols + i]; 
            m[j*cols+cols-1] -= r*m[i*cols+cols-1];
        }
    }

    // stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double *solutions = (double*)malloc((rows)*sizeof(double));
    for(int i = 0; i < rows; i++){
        solutions[i] = m[i*cols+cols-1];
    }

    // printMatrix(m, rows, cols);
    writeResults(solutions, rows, 1, milliseconds);

    // close files
    fclose(f);
    free(m);

    return 0;
}

double* readMatrix(FILE* fp, int* nrows, int* ncols) {
    // read in a matrix from file input
    char* line = (char*) malloc(MAX_LINE_LEN * sizeof(char));
    char* token = NULL;
    // size_t len = 0;

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

void writeResults(double* m, int rows, int cols, float time) {
    FILE* fout = fopen("solved_sys_seq.txt", "a");
    if (!fout) {
      printf("Failed to create output file solved_sys.txt\n");
    }
    fprintf(fout, "rows: %d cols: %d \n", rows, cols);
    fprintf(fout, "execution time: %f milliseconds\n ", time);
    
    fprintf(fout, "\n \n \n \n \n ");
    fclose(fout);
}

int main(int argc, char* argv[]) {
    char* file; 
    file = argv[1];

    linearSolver(file);

    return 0;
}
