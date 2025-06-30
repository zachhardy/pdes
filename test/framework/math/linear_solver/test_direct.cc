#include <gtest/gtest.h>
#include "test/framework/math/linear_solver/utils/matrix_builder.h"
#include "framework/math/linear_solver/lu_solver.h"


namespace pdes::test
{
  TEST(LinearSolver_Direct, LUSolves3x3System)
  {
    using VectorType = Vector<>;
    using MatrixType = Matrix<>;

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
    const VectorType x_true(3, 1.0);

    // Compute right-hand side: b = A * x_true
    const auto b = A * x_true;

    // Solve with LUSolver
    LUSolver<> solver;
    solver.factor(A);

    VectorType x;
    solver.solve(b, x);
    const auto residual = (A * x - b).l2_norm();

    std::cout << std::setw(18) << std::left
        << "LUSolver"
        << " | final ||r||: " << residual << "\n";

    // Check residual
    EXPECT_LT(residual, 1e-12) << "Residual too high";
    for (size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(x(i), x_true(i), 1e-10);
  }

  TEST(LinearSolver_Direct, LUSolvesLaplace1D)
  {
    using VectorType = Vector<>;

    constexpr size_t n = 10;
    const auto A = utils::laplace_1d(n);

    // Known solution x_true = [1, 1, 1]
    const VectorType x_true(n, 1.0);

    // Compute right-hand side: b = A * x_true
    const auto b = A * x_true;

    // Solve with LUSolver
    LUSolver<> solver;
    solver.factor(A);

    VectorType x;
    solver.solve(b, x);
    const auto residual = (A * x - b).l2_norm();

    std::cout << std::setw(18) << std::left
        << "LUSolver"
        << " | final ||r||: " << residual << "\n";

    // Check residual
    EXPECT_LT(residual, 1e-12) << "Residual too high";
    for (size_t i = 0; i < x.size(); ++i)
      EXPECT_NEAR(x(i), x_true(i), 1e-10);
  }
} // namespace pdes
