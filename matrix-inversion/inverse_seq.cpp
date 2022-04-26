#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

void matrix_read(char inFile[], double *A[], double *I[], int n) {
	FILE *fp;
	int row, col;

	fp = fopen(inFile, "r");
	if (fp == NULL)
		return;

	for (row = 0; row < n; row++) {
		A[row] = new double[n];

		for (col = 0; col < n; col++) {	
			if (row == col) I[row][col] = 1;
			else I[row][col] = 0;

			if (fscanf(fp, "%f,", &A[row * n + col]) == EOF) break;
		}

		if (feof(fp)) break;
	}

	fclose(fp);
}


void savetofile(string outFile, double *matrix[], int n) {
	ofstream ofile;
	int row, col;

	ofile.open(outFile, ios::out | ios::app);

	for (row = 0; row < n; row++) {
		for (col = n; col < 2*n; col++) {	
			ofile << matrix[row][col] << ",";
		}
		ofile << "\n";
	}
	ofile.close();
}

void saveTime(char inFile[], int time) {
	ofstream ofile;
	ofile.open("times.txt", ios::out | ios::app);
	ofile << "seq" << inFile << ": " << time << " ms \n";
	ofile.close();
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
	char* inFile;
	string outFile;
	int n;

	inFile = (char*) malloc (20);
	outFile = (char*) malloc (21);

	inFile = argv[1];
	outFile = argv[2];
	n = stoi(argv[3]);

	double * A ;
	double * I ;

	A = (double*) malloc (n*n);
	I = (double*) malloc (n*n);
	
	matrix_read(inFile, &A, &I, n);

	auto start = high_resolution_clock::now();
	inverse(A,I,n);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	saveTime(inFile, duration.count());
	savetofile(outFile, &I, n);

    return 0;
}
