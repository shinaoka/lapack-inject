! Test program for DGETRF (LU factorization)
program test_dgetrf
  implicit none

  integer, parameter :: m = 3, n = 3
  double precision :: a(m,n), a_orig(m,n)
  integer :: ipiv(min(m,n)), info

  ! Matrix A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
  a(1,1) = 1.0d0; a(1,2) = 2.0d0; a(1,3) = 3.0d0
  a(2,1) = 4.0d0; a(2,2) = 5.0d0; a(2,3) = 6.0d0
  a(3,1) = 7.0d0; a(3,2) = 8.0d0; a(3,3) = 10.0d0

  a_orig = a

  print *, "Testing DGETRF..."
  print *, "Matrix A:"
  print *, a(1,:)
  print *, a(2,:)
  print *, a(3,:)

  ! Call DGETRF
  call dgetrf(m, n, a, m, ipiv, info)

  if (info == 0) then
    print *, "DGETRF succeeded!"
    print *, "LU factorization:"
    print *, a(1,:)
    print *, a(2,:)
    print *, a(3,:)
    print *, "Pivot:", ipiv
  else if (info > 0) then
    print *, "Matrix is singular, U(", info, ",", info, ") = 0"
    stop 1
  else
    print *, "DGETRF failed with info =", info
    stop 1
  end if

  print *, "Test PASSED"
end program test_dgetrf
