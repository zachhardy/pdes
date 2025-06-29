#include <gtest/gtest.h>
#include "framework/math/linear_solver/lu_solver.h"
#include "framework/math/vector.h"
#include "framework/math/matrix.h"

namespace pdes::test
{
  TEST(LinearSolver_Direct, LUSolves3x3System)
  {
    using VectorType = Vector<>;
    using MatrixType = Matrix<>;
    using value_type = VectorType::value_type;

    // Matrix A
    MatrixType A(3, 3);
    A(0, 0) = 2;
    A(0, 1) = 3;
    A(0, 2) = 1;
    A(1, 0) = 4;
    A(1, 1) = 7;
    A(1, 2) = 7;
    A(2, 0) = 6;
    A(2, 1) = 18;
    A(2, 2) = 22;

    // Known solution x_true = [1, 1, 1]
    VectorType x_true(3, 1.0);

    // Compute right-hand side: b = A * x_true
    const VectorType b = A * x_true;

    // Solve with LUSolver
    LUSolver<> solver;
    solver.factor(A);

    VectorType x;
    solver.solve(b, x);

    // Check residual
    const value_type residual = (A * x - b).l2_norm();
    EXPECT_LT(residual, 1e-12) << "Residual too high";

    // Check solution is close to [1, 1, 1]
    for (size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(x(i), x_true(i), 1e-12);
  }
} // namespace pdes
