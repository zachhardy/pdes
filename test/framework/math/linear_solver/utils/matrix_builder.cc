#include "matrix_builder.h"

namespace pdes::test::utils
{
  Matrix<>
  laplace_1d(const size_t n)
  {
    // Build 1D Laplacian:
    // tridiagonal with 2 on diagonal, -1 off-diagonal
    Matrix<> A(n, n);
    for (size_t i = 0; i < n; ++i)
    {
      A(i, i) = 2.0;
      if (i > 0)
        A(i, i - 1) = -1.0;
      if (i < n - 1)
        A(i, i + 1) = -1.0;
    }
    return A;
  }
}
