#include <gtest/gtest.h>

#include "framework/math/vector.h"
#include "framework/math/linear_solver/cg_solver.h"
#include "framework/math/linear_solver/preconditioner/precondition_jacobi.h"
#include "framework/math/linear_solver/preconditioner/precondition_ssor.h"
#include "framework/math/linear_solver/preconditioner/precondition_ilu.h"
#include "framework/math/linear_solver/solver_control.h"

#include "test/framework/math/linear_solver/utils/matrix_builder.h"
#include "test/framework/math/linear_solver/utils/test_convergence.h"

#include <cmath>
#include <iomanip>

namespace pdes::test
{
  TEST(LinearSolver_Krylov, PCCGConvergesOnLaplace)
  {
    constexpr size_t n = 10;
    const Matrix<> A = utils::laplace_1d(n);

    SolverControl control(1000, 1.0e-10);
    const CGSolver<> solver(&control);

    // No preconditioning
    {
      Vector<> b(n, 1.0); // RHS: all ones
      Vector<> x(n, 0.0); // Initial guess

      const auto result = solver.solve(A, b, x);
      EXPECT_TRUE(result.converged);

      // Check that Ax = b
      Vector<> Ax = A.vmult(x);
      for (size_t i = 0; i < n; ++i)
        EXPECT_LT(std::abs(Ax(i) - b(i)), 1e-6);
    }
    // Jacobi preconditioning
    {
      Vector<> b(n, 1.0); // RHS: all ones
      Vector<> x(n, 0.0); // Initial guess
      const PreconditionJacobi<> M(&A);

      const auto result = solver.solve(A, b, x, M);
      EXPECT_TRUE(result.converged);

      // Check that Ax = b
      Vector<> Ax = A.vmult(x);
      for (size_t i = 0; i < n; ++i)
        EXPECT_LT(std::abs(Ax(i) - b(i)), 1e-6);
    }
    // SSOR preconditioning
    {
      Vector<> b(n, 1.0); // RHS: all ones
      Vector<> x(n, 0.0); // Initial guess
      const PreconditionSSOR<> M(&A);

      const auto result = solver.solve(A, b, x, M);
      EXPECT_TRUE(result.converged);

      // Check that Ax = b
      Vector<> Ax = A.vmult(x);
      for (size_t i = 0; i < n; ++i)
        EXPECT_LT(std::abs(Ax(i) - b(i)), 1e-6);
    }
    // ILU preconditioning
    {
      Vector<> b(n, 1.0); // RHS: all ones
      Vector<> x(n, 0.0); // Initial guess
      const PreconditionILU M(&A);

      const auto result = solver.solve(A, b, x, M);
      EXPECT_TRUE(result.converged);

      // Check that Ax = b
      Vector<> Ax = A.vmult(x);
      for (size_t i = 0; i < n; ++i)
        EXPECT_LT(std::abs(Ax(i) - b(i)), 1e-6);
    }
  }

  TEST(LinearSolver_Krylov, CGComparePreconditioners)
  {
    Matrix<> A = utils::laplace_1d(50);
    Vector<> b(A.m(), 1.0);

    const auto i_identity = utils::test_convergence<CGSolver<>>(
      "Identity", A, b, PreconditionIdentity<>());
    const auto i_jacobi = utils::test_convergence<CGSolver<>>(
      "Jacobi", A, b, PreconditionJacobi<>(&A));
    const auto i_ilu = utils::test_convergence<CGSolver<>>(
      "ILU", A, b, PreconditionILU<>(&A));

    EXPECT_GE(i_identity, i_jacobi);
    EXPECT_GT(i_jacobi, i_ilu);
  }
}
