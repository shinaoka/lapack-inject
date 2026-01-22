# Makefile for Fortran tests of lapack-inject

FC = gfortran
CC = cc
FFLAGS = -O2
CFLAGS = -O2

# Detect OS and set paths
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OPENBLAS_LIB ?= /opt/homebrew/opt/openblas/lib
    DL_LIBS =
else
    OPENBLAS_LIB ?= /usr/lib/x86_64-linux-gnu/openblas-pthread
    DL_LIBS = -ldl
endif

# Link init code, lapack-inject, then OpenBLAS for BLAS
LDFLAGS = -L../target/release -llapack_inject -L$(OPENBLAS_LIB) -lopenblas $(DL_LIBS)

.PHONY: all clean test build-lib

all: build-lib test_dgesv test_dgetrf test_dpotrf

build-lib:
	cd .. && cargo build --release

init_openblas.o: init_openblas.c
	$(CC) $(CFLAGS) -c $< -o $@

test_dgesv: test_dgesv.f90 init_openblas.o
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

test_dgetrf: test_dgetrf.f90 init_openblas.o
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

test_dpotrf: test_dpotrf.f90 init_openblas.o
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

test: all
	@echo "=== Testing DGESV (Fortran) ==="
	./test_dgesv
	@echo ""
	@echo "=== Testing DGETRF (Fortran) ==="
	./test_dgetrf
	@echo ""
	@echo "=== Testing DPOTRF (Fortran) ==="
	./test_dpotrf

clean:
	rm -f test_dgesv test_dgetrf test_dpotrf *.o *.mod
