! Test program for DGESV
program test_dgesv
  implicit none

  integer, parameter :: n = 3, nrhs = 1
  double precision :: a(n,n), b(n,nrhs)
  integer :: ipiv(n), info

  ! Matrix A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
  a(1,1) = 1.0d0; a(1,2) = 2.0d0; a(1,3) = 3.0d0
  a(2,1) = 4.0d0; a(2,2) = 5.0d0; a(2,3) = 6.0d0
  a(3,1) = 7.0d0; a(3,2) = 8.0d0; a(3,3) = 10.0d0

  ! Right-hand side b = [1, 2, 3]
  b(1,1) = 1.0d0
  b(2,1) = 2.0d0
  b(3,1) = 3.0d0

  print *, "Testing DGESV..."
  print *, "Matrix A:"
  print *, a(1,:)
  print *, a(2,:)
  print *, a(3,:)
  print *, "RHS b:", b(:,1)

  ! Call DGESV
  call dgesv(n, nrhs, a, n, ipiv, b, n, info)

  if (info == 0) then
    print *, "DGESV succeeded!"
    print *, "Solution x:", b(:,1)
    print *, "Pivot:", ipiv
    ! Expected solution: x = [1, -2, 1] (approximately)
  else
    print *, "DGESV failed with info =", info
    stop 1
  end if

  print *, "Test PASSED"
end program test_dgesv
