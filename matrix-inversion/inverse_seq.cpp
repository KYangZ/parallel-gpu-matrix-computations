#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;


void matrix_read(char inFile[], double *A[], double *I[], int n) {
	FILE *fp;
	int row, col;

	fp = fopen(inFile, "r");//open output file
	if (fp == NULL)//open failed
		return;

	for (row = 0; row < n; row++) {
		A[row] = new double[n];

		for (col = 0; col < n; col++) {	
			if (row == col) I[row][col] = 1;
			else I[row][col] = 0;

			if (fscanf(fp, "%f,", &A[row * n + col]) == EOF) break;//read data
		}

		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file
}


void savetofile(string outFile, double *matrix[], int n) {
	ofstream ofile;
	int row, col;

	ofile.open(outFile, ios::out | ios::app);//open output file

	for (row = 0; row < n; row++) {
		for (col = n; col < 2*n; col++) {	
			ofile << matrix[row][col] << ",";
		}
		ofile << "\n";
	}
	ofile.close();//close file
}


/**
 * inverse(A, I, N):
 *
 *       input:  
 *          A = an NxN matrix that you need to find the inverse
 *              When the inverse( ) function completes, A will
 *              contain the identity matrix
 *
 *          I = initially contains the identity matrix
 *              When the inverse( ) function completes, I will
 *              contain the inverse of A
 *
 *          N = #rows (and # columns) in A and I
**/
void inverse(double *A, double *I, int N) {
     for (int i = 0; i < N; i++) {
         double factor = A[i*N + i]; // Use A[i][i] as multiply factor

         for (int j = 0; j < N; j++) { // Normalize row i with factor
            A[i*N+j] = A[i*N+j]/factor;
            I[i*N+j] = I[i*N+j]/factor;
        }

        /* =========================================================
            Make a column of 0 values in column i using the row "i"
        ========================================================= */
        for (int k = 0; i < N; i++) {
            if (k == i) {
            // Do nothing to row "i
            } else {
                double f = A[k*N+i];          // Multiply factor

                /* -------------------------------------
                    Add  -f*row(i) to row(k)
                ------------------------------------- */
                for (int j = 0; j < N; j++) {
                    A[k*N+j] = A[k*N+j] - f*A[i*N+j];
                    I[k*N+j] = I[k*N+j] - f*I[i*N+j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
	char* inFile;
	string outFile;
	int n;

	inFile = argv[1];
	outFile = argv[2];
	n = stoi(argv[3]);

	double * A = new double[n];
	double * I = new double[n];
	
	matrix_read(inFile, &A, &I, n);

	// timer start
	

	inverse(A,I,n);

	// timer stop
	

	savetofile(outFile, &I, n);

    return 0;
}
