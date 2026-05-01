#ifndef LAPACK_INJECT_CTEST_LAPACKE_H
#define LAPACK_INJECT_CTEST_LAPACKE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int lapack_int;

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

lapack_int LAPACKE_dgesv(
    int matrix_layout,
    lapack_int n,
    lapack_int nrhs,
    double* a,
    lapack_int lda,
    lapack_int* ipiv,
    double* b,
    lapack_int ldb
);

lapack_int LAPACKE_dgetrf(
    int matrix_layout,
    lapack_int m,
    lapack_int n,
    double* a,
    lapack_int lda,
    lapack_int* ipiv
);

lapack_int LAPACKE_dgetri(
    int matrix_layout,
    lapack_int n,
    double* a,
    lapack_int lda,
    const lapack_int* ipiv
);

lapack_int LAPACKE_dpotrf(
    int matrix_layout,
    char uplo,
    lapack_int n,
    double* a,
    lapack_int lda
);

long long LAPACKE_dgesv_64(
    long long matrix_layout,
    long long n,
    long long nrhs,
    double* a,
    long long lda,
    long long* ipiv,
    double* b,
    long long ldb
);

long long LAPACKE_dgetrf_64(
    long long matrix_layout,
    long long m,
    long long n,
    double* a,
    long long lda,
    long long* ipiv
);

long long LAPACKE_dgetri_64(
    long long matrix_layout,
    long long n,
    double* a,
    long long lda,
    const long long* ipiv
);

long long LAPACKE_dpotrf_64(
    long long matrix_layout,
    char uplo,
    long long n,
    double* a,
    long long lda
);

#ifdef __cplusplus
}
#endif

#endif
