#include <stdio.h>
#include <stdlib.h>

void writeResults(double* m, int rows, int cols);

void writeResults(double* m, int rows, int cols) {
    FILE* fout = fopen("5kx5k.txt", "a");
    if (!fout) {
      printf("Failed to create output file solved_sys.txt\n");
    }
    fprintf(fout, "%d %d \n", rows, cols);
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
    int rows = 5000;
    int cols = 5000;
    srand ( 123 );
    double *m = (double*)malloc(rows*cols*sizeof(double));
    for(int i = 0; i < rows-1; i++){
        for(int j = 0; j < cols; j++){
            m[i*cols+j] = rand();
        }
    }
    writeResults(m, rows, cols);
    return 0;
}