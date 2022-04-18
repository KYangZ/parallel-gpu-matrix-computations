/**
 * inverse(A, C, N):
 *
 *       input:  
 *          A = an NxN matrix that you need to find the inverse
 *              When the inverse( ) function completes, A will
 *              contain the identity matrix
 *
 *          C = initially contains the identity matrix
 *              When the inverse( ) function completes, C will
 *              contain the inverse of A
 *
 *          N = #rows (and # columns) in A and C
**/
void inverse(double *A, double *C, int N) {
     for (int i = 0; i < N; i++) {
         double factor = A[i*N + i]; // Use A[i][i] as multiply factor

         for (int j = 0; j < N; j++) { // Normalize row i with factor
            A[i*N+j] = A[i*N+j]/factor;
            C[i*N+j] = C[i*N+j]/factor;
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
                    C[k*N+j] = C[k*N+j] - f*C[i*N+j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
	char* file;
	int n;

	file = argv[1];
	n = argv[2];

	printf("inverse: %f\n", inversion(file, n));
    return 0;
}