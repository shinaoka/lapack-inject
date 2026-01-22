/* Test for LAPACKE_dgetrf - LU factorization */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lapacke.h"

/* Auxiliary routines prototypes (from lapacke_example_aux.c) */
extern void print_matrix_colmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm);
extern void print_matrix_rowmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm);
extern void print_vector(char* desc, lapack_int n, lapack_int* vec);

#define N 4
#define LDA N

int main() {
    lapack_int m = N, n = N, lda = LDA, info;
    lapack_int ipiv[N];

    /* Test matrix (column-major storage) */
    double a[LDA*N] = {
        6.80, -2.11,  5.66,  5.97,
       -6.05, -3.30,  5.36, -4.44,
       -0.45,  2.58, -2.70,  0.27,
        8.32,  2.71,  4.35, -7.17
    };

    printf("=== LAPACKE_dgetrf Test (Column-Major) ===\n");
    print_matrix_colmajor("Entry Matrix A", m, n, a, lda);

    /* Call LAPACKE_dgetrf */
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);

    if (info > 0) {
        printf("U(%d,%d) is exactly zero - matrix is singular\n", info, info);
        return 1;
    } else if (info < 0) {
        printf("Parameter %d had an illegal value\n", -info);
        return 1;
    }

    printf("LAPACKE_dgetrf completed successfully\n");
    print_matrix_colmajor("LU factorization", m, n, a, lda);
    print_vector("Pivot indices", n, ipiv);

    /* Test row-major interface */
    printf("\n=== LAPACKE_dgetrf Test (Row-Major) ===\n");

    /* Row-major version of same matrix */
    double a_row[LDA*N] = {
        6.80, -6.05, -0.45,  8.32,
       -2.11, -3.30,  2.58,  2.71,
        5.66,  5.36, -2.70,  4.35,
        5.97, -4.44,  0.27, -7.17
    };
    lapack_int ipiv_row[N];

    print_matrix_rowmajor("Entry Matrix A (row-major)", m, n, a_row, lda);

    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a_row, lda, ipiv_row);

    if (info > 0) {
        printf("U(%d,%d) is exactly zero - matrix is singular\n", info, info);
        return 1;
    } else if (info < 0) {
        printf("Parameter %d had an illegal value\n", -info);
        return 1;
    }

    printf("LAPACKE_dgetrf (row-major) completed successfully\n");
    print_matrix_rowmajor("LU factorization (row-major)", m, n, a_row, lda);
    print_vector("Pivot indices (row-major)", n, ipiv_row);

    printf("\nAll DGETRF tests PASSED\n");
    return 0;
}
