#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "io.h"
void print_matrix_seq(float**, int N);
void gen_matrix_seq(float**, int);
void lud_sequential(float**, float**, float**, int);

void print_matrix_seq(float** mat, int N){
    printf("[");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%.2f\t",mat[i][j]);
        }
        printf("\n");
    }
    printf("]\n");
}
void create_matrix_seq(float** mat, int N){
    for(int i=0;i<N;i++){
        mat[i] = new float[N];
    }
}
void delete_matrix_seq(float** mat, int N){
    for(int i=0;i<N;i++){
        delete[] mat[i];
    }
}
void gen_matrix_seq(float** mat, int N){
    for (int i=0;i<N;i++){
		for(int j=0; j < N; j++){
			mat[i][j] = ((rand()%10)+1) ;
		}
	}
}
void flat_to_grid(float** mat, float* tocopy, int N){
    int count = 0;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            mat[i][j] = tocopy[count];
            count++;
        }
    }
}
bool compare_float(float x, float y, float epsilon = 1.0f){
    if(std::fabs(x - y) < epsilon)
        return true; //they are same
    return false; //they are not same
}
void compare_grids(float** mat1, float** mat2, int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(!compare_float(mat1[i][j], mat2[i][j])){
                printf("Not Same\n");
                printf("%f, %f \n", mat1[i][j], mat2[i][j]);
                // return;
            }
        }
    }
    // printf("Passed!\n");
}

void lud_sequential(float** l, float** a, int N){
    //gaussian elimination based 
    for(int j = 0; j<N; j++){
        for(int i=j;i<N;i++){
            if(i==j){
                for(int k=j;k<N;k++){
                    // if(i==0) u[i][k] = a[i][k];
                    if(k==i)l[i][k] = 1;
                    else l[i][k] = 0;
                }
            }else{
                float f = a[i][j]/a[j][j];
                for(int k=j;k<N;k++){
                    a[i][k] = a[i][k]-f*a[j][k];
                }
                l[i][j] = f;
            }
        }
    }
    
}
void simple_verification_seq(int N){
    float** l = new float*[N];
    float** A = new float*[N];
    create_matrix_seq(l, N);
    create_matrix_seq(A, N);
    gen_matrix_seq(A, N);
    print_matrix_seq(A, N);
    lud_sequential(l, A, N);
    print_matrix_seq(l, N);
    print_matrix_seq(A, N);
    delete_matrix_seq(A, N);
    delete_matrix_seq(l, N);
}
void simple_verification_seq(int N, float* flat){
    float** l = new float*[N];
    float** A = new float*[N];
    create_matrix_seq(l, N);
    create_matrix_seq(A, N);
    flat_to_grid(A, flat, N);
    print_matrix_seq(A, N);
    lud_sequential(l, A, N);
    print_matrix_seq(l, N);
    print_matrix_seq(A, N);
    delete_matrix_seq(A, N);
    delete_matrix_seq(l, N);
}
// int main(int argc, char** argv){
//     int N = atoi(argv[1]);
//     srand(42);
    
//     simple_verification_seq(N);
//     return 0;
// }