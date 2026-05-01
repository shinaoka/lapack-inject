/* Test for LAPACKE_dgesv - linear solve */
#include <math.h>
#include <stdio.h>
#include "lapacke.h"

#define N 2
#define NRHS 1

static int approx_eq(double got, double want) {
    return fabs(got - want) < 1e-10;
}

int main() {
    lapack_int ipiv[N];

    double a_col[N * N] = {
        1.0, 3.0,
        2.0, 4.0
    };
    double b_col[N] = {5.0, 11.0};

    printf("=== LAPACKE_dgesv Test (Column-Major) ===\n");
    lapack_int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, a_col, N, ipiv, b_col, N);
    if (info != 0) {
        printf("LAPACKE_dgesv failed with info = %d\n", info);
        return 1;
    }
    if (!approx_eq(b_col[0], 1.0) || !approx_eq(b_col[1], 2.0)) {
        printf("Column-major solution mismatch: [%f, %f]\n", b_col[0], b_col[1]);
        return 1;
    }

    double a_row[N * N] = {
        1.0, 2.0,
        3.0, 4.0
    };
    double b_row[N] = {5.0, 11.0};

    printf("=== LAPACKE_dgesv Test (Row-Major) ===\n");
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, NRHS, a_row, N, ipiv, b_row, NRHS);
    if (info != 0) {
        printf("LAPACKE_dgesv row-major failed with info = %d\n", info);
        return 1;
    }
    if (!approx_eq(b_row[0], 1.0) || !approx_eq(b_row[1], 2.0)) {
        printf("Row-major solution mismatch: [%f, %f]\n", b_row[0], b_row[1]);
        return 1;
    }

    printf("All DGESV tests PASSED\n");
    return 0;
}
