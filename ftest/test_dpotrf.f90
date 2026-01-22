! Test program for DPOTRF (Cholesky factorization)
program test_dpotrf
  implicit none

  integer, parameter :: n = 3
  double precision :: a(n,n)
  integer :: info
  character :: uplo

  ! Symmetric positive definite matrix
  ! A = [[4, 2, 2], [2, 5, 3], [2, 3, 6]]
  a(1,1) = 4.0d0; a(1,2) = 2.0d0; a(1,3) = 2.0d0
  a(2,1) = 2.0d0; a(2,2) = 5.0d0; a(2,3) = 3.0d0
  a(3,1) = 2.0d0; a(3,2) = 3.0d0; a(3,3) = 6.0d0

  uplo = 'L'

  print *, "Testing DPOTRF..."
  print *, "Matrix A (SPD):"
  print *, a(1,:)
  print *, a(2,:)
  print *, a(3,:)

  ! Call DPOTRF (lower triangular)
  call dpotrf(uplo, n, a, n, info)

  if (info == 0) then
    print *, "DPOTRF succeeded!"
    print *, "Cholesky factor L:"
    print *, a(1,:)
    print *, a(2,:)
    print *, a(3,:)
    ! Expected L = [[2, *, *], [1, 2, *], [1, 1, 2]]
  else if (info > 0) then
    print *, "Matrix is not positive definite, leading minor", info
    stop 1
  else
    print *, "DPOTRF failed with info =", info
    stop 1
  end if

  print *, "Test PASSED"
end program test_dpotrf
