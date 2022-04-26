#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>

using namespace std;

void matrix_read(char inFile[], double *A[], double *I[], int n) {
	FILE *fp = fopen(inFile, "r");
	if (fp == NULL)
		return;

	for (int row = 0; row < n; row++) {
		A[row] = new double[n];

		for (int col = 0; col < n; col++) {	
			if (row == col) I[row][col] = 1;
			else I[row][col] = 0;

			if (fscanf(fp, "%lf,", A[row * n + col]) == EOF) break;
		}

		if (feof(fp)) break;
	}

	fclose(fp);
}

void inverse(double *A, double *I, int N) {

     for (int i = 0; i < N; i++) {
         double factor = A[i*N + i]; 

         for (int j = 0; j < N; j++) { 
            A[i*N+j] = A[i*N+j]/factor;
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
	if(argc == 0) {
		return 0;
	}

	int n = stoi(argv[3]);

	double * A = (double*) malloc (n);
	double * I = (double*) malloc (n);
	matrix_read(argv[1], &A, &I, n);

	clock_t start = clock();

	inverse(A,I,n);

	clock_t finish = clock();
	double time = (double)(finish - start);

	cout << "time: " << time << "\n";

	free(A);
	free(I);

    return 0;
}