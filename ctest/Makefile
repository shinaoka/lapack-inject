# Makefile for lapack-inject LAPACKE C tests.

CC = cc
CFLAGS = -I. -O2

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OPENBLAS_LIB ?= /opt/homebrew/opt/openblas/lib
    DL_LIBS =
    RPATH_FLAG = -Wl,-rpath,$(abspath ../target/release)
else
    OPENBLAS_LIB ?= /usr/lib/x86_64-linux-gnu/openblas-pthread
    DL_LIBS = -ldl
    RPATH_FLAG = -Wl,-rpath,$(abspath ../target/release)
endif

LDFLAGS = -L../target/release $(RPATH_FLAG) -llapack_inject -L$(OPENBLAS_LIB) -lopenblas -lm $(DL_LIBS)

.PHONY: all clean test build-lib

all: build-lib xtest_DGESV xtest_DGETRF xtest_DGETRI xtest_DPOTRF

build-lib:
	cd .. && cargo build --release

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

xtest_DGESV: test_DGESV.o init_openblas.o
	$(CC) -o $@ $^ $(LDFLAGS)

xtest_DGETRF: test_DGETRF.o lapacke_example_aux.o init_openblas.o
	$(CC) -o $@ $^ $(LDFLAGS)

xtest_DGETRI: test_DGETRI.o lapacke_example_aux.o init_openblas.o
	$(CC) -o $@ $^ $(LDFLAGS)

xtest_DPOTRF: test_DPOTRF.o lapacke_example_aux.o init_openblas.o
	$(CC) -o $@ $^ $(LDFLAGS)

test: all
	@echo "=== Testing DGESV ==="
	./xtest_DGESV
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
	rm -f xtest_* *.o
