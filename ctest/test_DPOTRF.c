/* Test for LAPACKE_dpotrf - Cholesky factorization */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lapacke.h"

/* Auxiliary routines prototypes (from lapacke_example_aux.c) */
extern void print_matrix_colmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm);
extern void print_matrix_rowmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm);

#define N 3
#define LDA N

/* Verify L * L^T = A (for lower triangular Cholesky, column-major) */
int verify_cholesky_lower(int n, double* a_orig, double* l, int lda) {
    double tol = 1e-10;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {  /* Only lower triangular part */
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += l[i + k*lda] * l[j + k*lda];
            }
            if (fabs(sum - a_orig[i + j*lda]) > tol) {
                printf("Verification failed at (%d,%d): got %f, expected %f\n",
                       i, j, sum, a_orig[i + j*lda]);
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    lapack_int n = N, lda = LDA, info;

    /* Symmetric positive definite matrix (column-major) */
    /* A = | 4  2  2 |
           | 2  5  3 |
           | 2  3  6 | */
    double a[LDA*N] = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 3.0,
        2.0, 3.0, 6.0
    };

    /* Save original for verification */
    double a_orig[LDA*N];
    for (int i = 0; i < LDA*N; i++) a_orig[i] = a[i];

    printf("=== LAPACKE_dpotrf Test (Column-Major, Lower) ===\n");
    print_matrix_colmajor("Entry Matrix A (SPD)", n, n, a, lda);

    /* Call LAPACKE_dpotrf with 'L' for lower triangular */
    info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a, lda);

    if (info > 0) {
        printf("The leading minor of order %d is not positive definite\n", info);
        return 1;
    } else if (info < 0) {
        printf("Parameter %d had an illegal value\n", -info);
        return 1;
    }

    printf("LAPACKE_dpotrf completed successfully\n");
    print_matrix_colmajor("Cholesky factor L", n, n, a, lda);

    /* Verify L * L^T = A */
    if (verify_cholesky_lower(n, a_orig, a, lda) != 0) {
        printf("Cholesky verification FAILED\n");
        return 1;
    }
    printf("Cholesky verification PASSED\n");

    /* Test upper triangular */
    printf("\n=== LAPACKE_dpotrf Test (Column-Major, Upper) ===\n");

    double a_upper[LDA*N] = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 3.0,
        2.0, 3.0, 6.0
    };

    print_matrix_colmajor("Entry Matrix A (SPD)", n, n, a_upper, lda);

    info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, a_upper, lda);

    if (info != 0) {
        printf("LAPACKE_dpotrf (upper) failed with info = %d\n", info);
        return 1;
    }

    printf("LAPACKE_dpotrf (upper) completed successfully\n");
    print_matrix_colmajor("Cholesky factor U", n, n, a_upper, lda);

    /* Test row-major interface */
    printf("\n=== LAPACKE_dpotrf Test (Row-Major, Lower) ===\n");

    /* Same matrix in row-major order (symmetric, so same values) */
    double a_row[LDA*N] = {
        4.0, 2.0, 2.0,
        2.0, 5.0, 3.0,
        2.0, 3.0, 6.0
    };

    print_matrix_rowmajor("Entry Matrix A (row-major, SPD)", n, n, a_row, lda);

    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, a_row, lda);

    if (info != 0) {
        printf("LAPACKE_dpotrf (row-major) failed with info = %d\n", info);
        return 1;
    }

    printf("LAPACKE_dpotrf (row-major) completed successfully\n");
    print_matrix_rowmajor("Cholesky factor L (row-major)", n, n, a_row, lda);

    printf("\nAll DPOTRF tests PASSED\n");
    return 0;
}
