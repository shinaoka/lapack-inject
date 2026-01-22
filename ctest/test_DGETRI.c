/* Test for LAPACKE_dgetri - Matrix inverse via LU factorization */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lapacke.h"

/* Auxiliary routines prototypes (from lapacke_example_aux.c) */
extern void print_matrix_colmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm);
extern void print_matrix_rowmajor(char* desc, lapack_int m, lapack_int n, double* mat, lapack_int ldm);
extern void print_vector(char* desc, lapack_int n, lapack_int* vec);

#define N 3
#define LDA N

/* Verify A * A_inv = I (column-major) */
int verify_inverse_colmajor(int n, double* a, double* a_inv, int lda) {
    double tol = 1e-10;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i + k*lda] * a_inv[k + j*lda];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(sum - expected) > tol) {
                printf("Verification failed at (%d,%d): got %f, expected %f\n",
                       i, j, sum, expected);
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    lapack_int n = N, lda = LDA, info;
    lapack_int ipiv[N];

    /* Simple invertible matrix (column-major) */
    double a[LDA*N] = {
        1.0, 0.0, 5.0,
        2.0, 1.0, 6.0,
        3.0, 4.0, 0.0
    };

    /* Save original for verification */
    double a_orig[LDA*N];
    for (int i = 0; i < LDA*N; i++) a_orig[i] = a[i];

    printf("=== LAPACKE_dgetri Test (Column-Major) ===\n");
    print_matrix_colmajor("Entry Matrix A", n, n, a, lda);

    /* Step 1: LU factorization */
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ipiv);
    if (info != 0) {
        printf("LAPACKE_dgetrf failed with info = %d\n", info);
        return 1;
    }
    printf("LAPACKE_dgetrf completed successfully\n");
    print_matrix_colmajor("LU factorization", n, n, a, lda);

    /* Step 2: Compute inverse */
    info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, a, lda, ipiv);
    if (info != 0) {
        printf("LAPACKE_dgetri failed with info = %d\n", info);
        return 1;
    }
    printf("LAPACKE_dgetri completed successfully\n");
    print_matrix_colmajor("Inverse A^{-1}", n, n, a, lda);

    /* Verify A * A_inv = I */
    if (verify_inverse_colmajor(n, a_orig, a, lda) != 0) {
        printf("Inverse verification FAILED\n");
        return 1;
    }
    printf("Inverse verification PASSED\n");

    /* Test row-major interface */
    printf("\n=== LAPACKE_dgetri Test (Row-Major) ===\n");

    /* Row-major version (transpose of column-major) */
    double a_row[LDA*N] = {
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    };

    lapack_int ipiv_row[N];

    print_matrix_rowmajor("Entry Matrix A (row-major)", n, n, a_row, lda);

    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, a_row, lda, ipiv_row);
    if (info != 0) {
        printf("LAPACKE_dgetrf (row-major) failed with info = %d\n", info);
        return 1;
    }
    printf("LAPACKE_dgetrf (row-major) completed successfully\n");

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a_row, lda, ipiv_row);
    if (info != 0) {
        printf("LAPACKE_dgetri (row-major) failed with info = %d\n", info);
        return 1;
    }
    printf("LAPACKE_dgetri (row-major) completed successfully\n");
    print_matrix_rowmajor("Inverse A^{-1} (row-major)", n, n, a_row, lda);

    printf("\nAll DGETRI tests PASSED\n");
    return 0;
}
