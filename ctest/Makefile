# Makefile for lapack-inject ctest
# Uses test files from extern/OpenBLAS/lapack-netlib/LAPACKE/example/

OPENBLAS_LAPACKE = ../extern/OpenBLAS/lapack-netlib/LAPACKE
OPENBLAS_EXAMPLE = $(OPENBLAS_LAPACKE)/example
VPATH = $(OPENBLAS_EXAMPLE)

CC = cc
CFLAGS = -I. -I$(OPENBLAS_LAPACKE)/include

# Detect OS and set OpenBLAS path
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS (Homebrew)
    OPENBLAS_LIB ?= /opt/homebrew/opt/openblas/lib
else
    # Linux
    OPENBLAS_LIB ?= /usr/lib/x86_64-linux-gnu/openblas-pthread
endif

# Link lapack-inject first (provides LAPACKE_* symbols)
# Then OpenBLAS for Fortran LAPACK (dgesv_, etc.) reference
LDFLAGS = -L../target/release -llapack_inject -L$(OPENBLAS_LIB) -lopenblas -lm

.PHONY: all clean test build-trampoline

all: build-trampoline xexample_DGESV_colmajor xexample_DGESV_rowmajor xexample_DGELS_colmajor xexample_DGELS_rowmajor xtest_DGETRF xtest_DGETRI xtest_DPOTRF

build-trampoline:
	cd .. && cargo build --release --features openblas

# Compile sources
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# LAPACKE example executables (from OpenBLAS)
xexample_DGESV_colmajor: example_DGESV_colmajor.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

xexample_DGESV_rowmajor: example_DGESV_rowmajor.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

xexample_DGELS_colmajor: example_DGELS_colmajor.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

xexample_DGELS_rowmajor: example_DGELS_rowmajor.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Custom test executables
xtest_DGETRF: test_DGETRF.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

xtest_DGETRI: test_DGETRI.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

xtest_DPOTRF: test_DPOTRF.o lapacke_example_aux.o
	$(CC) -o $@ $^ $(LDFLAGS)

test: all
	@echo "=== Testing DGESV column-major ==="
	./xexample_DGESV_colmajor
	@echo ""
	@echo "=== Testing DGESV row-major ==="
	./xexample_DGESV_rowmajor
	@echo ""
	@echo "=== Testing DGELS column-major ==="
	./xexample_DGELS_colmajor
	@echo ""
	@echo "=== Testing DGELS row-major ==="
	./xexample_DGELS_rowmajor
	@echo ""
	@echo "=== Testing DGETRF ==="
	./xtest_DGETRF
	@echo ""
	@echo "=== Testing DGETRI ==="
	./xtest_DGETRI
	@echo ""
	@echo "=== Testing DPOTRF ==="
	./xtest_DPOTRF

clean:
	rm -f xexample_* xtest_* *.o
