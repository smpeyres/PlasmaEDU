import numpy as np

def laplacian_tridiagnonal(N):
  diag = np.zeros(N)
  diag[0] = 1
  for i in range(1, N-1):
    diag[i] = 2.0

    """
    Construct the tridiagonal matrix for the 1D Poisson equation
    with Dirichlet boundary conditions.
    """
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = 2
        if i > 0:
            A[i,i-1] = -1
        if i < n-1:
            A[i,i+1] = -1
    return A


def poisson_tridiagonal(N):
  N = np.size(N)
  phi = np.zeros(N)
  a, diag, c = laplacian_tridiagnonal(N);

  # auxiliary vectors
  l = np.zeros(N)
  u = np.zeros(N)
  y = np.zeros(N)

  # Tridiagonal solver
  l[0] = diag[0]
  y[0] = rhs[0]/l[0]
  for i in range(1,N):
    u[i=1] =


  return

