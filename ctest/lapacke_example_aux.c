#include <stdio.h>
#include "lapacke.h"

void print_matrix_colmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm) {
    printf("%s\n", desc);
    for (lapack_int i = 0; i < m; i++) {
        for (lapack_int j = 0; j < n; j++) {
            printf(" %8.4f", mat[i + j * ldm]);
        }
        printf("\n");
    }
}

void print_matrix_rowmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm) {
    printf("%s\n", desc);
    for (lapack_int i = 0; i < m; i++) {
        for (lapack_int j = 0; j < n; j++) {
            printf(" %8.4f", mat[i * ldm + j]);
        }
        printf("\n");
    }
}

void print_vector(char* desc, lapack_int n, lapack_int* vec) {
    printf("%s\n", desc);
    for (lapack_int i = 0; i < n; i++) {
        printf(" %d", vec[i]);
    }
    printf("\n");
}
