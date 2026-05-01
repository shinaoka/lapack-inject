#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "lapacke.h"

typedef void (*dgesv_fn)(
    const lapack_int*, const lapack_int*, double*, const lapack_int*,
    lapack_int*, double*, const lapack_int*, lapack_int*
);
typedef void (*dgetrf_fn)(
    const lapack_int*, const lapack_int*, double*, const lapack_int*,
    lapack_int*, lapack_int*
);
typedef void (*dgetri_fn)(
    const lapack_int*, double*, const lapack_int*, const lapack_int*,
    double*, const lapack_int*, lapack_int*
);
typedef void (*dpotrf_fn)(
    const char*, const lapack_int*, double*, const lapack_int*, lapack_int*
);

extern int register_dgesv_lp64(dgesv_fn f);
extern int register_dgetrf_lp64(dgetrf_fn f);
extern int register_dgetri_lp64(dgetri_fn f);
extern int register_dpotrf_lp64(dpotrf_fn f);

static void* load_openblas(void) {
    const char* candidates[] = {
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
        "libopenblas.so",
        "libopenblas.so.0",
        NULL,
    };

    for (int i = 0; candidates[i] != NULL; i++) {
        void* handle = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle != NULL) {
            return handle;
        }
    }

    fprintf(stderr, "Failed to load OpenBLAS: %s\n", dlerror());
    exit(1);
}

static void* required_symbol(void* handle, const char* name) {
    void* sym = dlsym(handle, name);
    if (sym == NULL) {
        fprintf(stderr, "Missing OpenBLAS symbol %s\n", name);
        exit(1);
    }
    return sym;
}

__attribute__((constructor))
void init_lapack_inject(void) {
    void* handle = load_openblas();

    register_dgesv_lp64((dgesv_fn)required_symbol(handle, "dgesv_"));
    register_dgetrf_lp64((dgetrf_fn)required_symbol(handle, "dgetrf_"));
    register_dgetri_lp64((dgetri_fn)required_symbol(handle, "dgetri_"));
    register_dpotrf_lp64((dpotrf_fn)required_symbol(handle, "dpotrf_"));
}
