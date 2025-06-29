#include <gtest/gtest.h>

#include "framework/math/vector.h"
#include "framework/math/linear_solver/gauss_seidel_solver.h"
#include "framework/math/linear_solver/solver_control.h"

#include "test/framework/math/linear_solver/utils/matrix_builder.h"

#include <cmath>
#include <iomanip>

namespace pdes::test
{
  TEST(LinearSolver_Stationary, GaussSeidelConvergesOnLaplace)
  {
    constexpr size_t n = 10;
    const Matrix<> A = utils::laplace_1d(n);
    Vector<> b(n, 1.0); // RHS: all ones
    Vector<> x(n, 0.0); // Initial guess

    SolverControl control(1000, 1.0e-10);
    const GaussSeidelSolver<> solver(&control);
    const auto result = solver.solve(A, b, x);
    EXPECT_TRUE(result.converged);

    // Check that Ax = b
    Vector<> Ax = A.vmult(x);
    for (size_t i = 0; i < n; ++i)
      EXPECT_LT(std::abs(Ax(i) - b(i)), 1e-6);
  }
}
