// Initialize lapack-inject with OpenBLAS function pointers
// This is called before Fortran tests run

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Registration functions from lapack-inject
extern void register_dgesv(void* f);
extern void register_dgetrf(void* f);
extern void register_dgetri(void* f);
extern void register_dpotrf(void* f);
extern void register_dgels(void* f);

// Called automatically at program startup
__attribute__((constructor))
void init_lapack_inject(void) {
    // macOS Homebrew path
    void* handle = dlopen("/opt/homebrew/opt/openblas/lib/libopenblas.dylib", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        // Intel Mac path
        handle = dlopen("/usr/local/opt/openblas/lib/libopenblas.dylib", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!handle) {
        // Linux path
        handle = dlopen("libopenblas.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!handle) {
        fprintf(stderr, "Failed to load OpenBLAS: %s\n", dlerror());
        exit(1);
    }

    void* sym;

    sym = dlsym(handle, "dgesv_");
    if (sym) register_dgesv(sym);

    sym = dlsym(handle, "dgetrf_");
    if (sym) register_dgetrf(sym);

    sym = dlsym(handle, "dgetri_");
    if (sym) register_dgetri(sym);

    sym = dlsym(handle, "dpotrf_");
    if (sym) register_dpotrf(sym);

    sym = dlsym(handle, "dgels_");
    if (sym) register_dgels(sym);

    printf("lapack-inject initialized with OpenBLAS backend\n");
}
