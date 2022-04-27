#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>

void matrix_read(double **, double **, int);
void inverse(double [], double [], int);

void matrix_read(double **A, double **I, int n) {
	int row, col;

	for (row = 0; row < n; row++) {
		for (col = 0; col < n; col++) {	
			if (row == col) I[row][col] = 1.00000;
			else I[row][col] = 0.00000;
			A[row][col] = rand() * 1.00000;
		}
	}
}

void inverse(double A[], double I[], int N) {

     for (int i = 0; i < N; i++) {
         double factor = A[i*N + i]; 

         for (int j = 0; j < N; j++) { 
            A[i*N+j] = A[i*N+j] / factor;
            I[i*N+j] = I[i*N+j]/factor;
        }

        for (int k = 0; i < N; i++) {
            if (k != i) {
                double f = A[k*N+i];   
				
                for (int j = 0; j < N; j++) {
                    A[k*N+j] = A[k*N+j] - f*A[i*N+j];
                    I[k*N+j] = I[k*N+j] - f*I[i*N+j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
	if(argc < 2) {
		return 0;
	}

	int n = atoi(argv[1]);

	double ** A = new double*[n];
	double ** I = new double*[n];
	printf("begin read");
	matrix_read(A, I, n);
	printf("end read");

	clock_t start = clock();

	inverse(*A,*I,n);

	clock_t finish = clock();
	double time = (double)(finish - start);

	printf("%lf",time);

    return 0;
}