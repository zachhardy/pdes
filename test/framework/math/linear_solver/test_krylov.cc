#include <gtest/gtest.h>
#include "test/framework/math/linear_solver/utils/matrix_builder.h"
#include "test/framework/math/linear_solver/utils/test_convergence.h"
#include "framework/math/linear_solver/cg_solver.h"
#include "framework/math/linear_solver/bicgstab_solver.h"
#include "framework/math/linear_solver/preconditioner/precondition_jacobi.h"
#include "framework/math/linear_solver/preconditioner/precondition_ssor.h"
#include "framework/math/linear_solver/preconditioner/precondition_ilu.h"
#include "framework/math/linear_solver/solver_control.h"
#include <iomanip>

namespace pdes::test
{
  template<typename SolverType, typename PCType>
  struct SolverPCPair
  {
    using Solver = SolverType;
    using PC = PCType;
  };

  using AllKrylovCombos = ::testing::Types<
    SolverPCPair<CGSolver<>, PreconditionIdentity<>>,
    SolverPCPair<CGSolver<>, PreconditionJacobi<>>,
    SolverPCPair<CGSolver<>, PreconditionSSOR<>>,
    SolverPCPair<CGSolver<>, PreconditionILU<>>,
    SolverPCPair<BiCGStabSolver<>, PreconditionIdentity<>>,
    SolverPCPair<BiCGStabSolver<>, PreconditionJacobi<>>,
    SolverPCPair<BiCGStabSolver<>, PreconditionSSOR<>>,
    SolverPCPair<BiCGStabSolver<>, PreconditionILU<>>
  >;

  // clang-format off
  template <typename T>
  std::string get_type_label()
  {
    if constexpr (std::is_same_v<T, CGSolver<>>) { return "CG"; }
    else if constexpr (std::is_same_v<T, BiCGStabSolver<>>) { return "BiCGStab"; }
    else if constexpr (std::is_same_v<T, PreconditionIdentity<>>) { return "Identity"; }
    else if constexpr (std::is_same_v<T, PreconditionJacobi<>>) { return "Jacobi"; }
    else if constexpr (std::is_same_v<T, PreconditionSSOR<>>) { return "SSOR"; }
    else if constexpr (std::is_same_v<T, PreconditionILU<>>) { return "ILU"; }
    else return "Unknown";
  }

  struct SolverPCName {
    template <typename T>
    static std::string GetName(int) {
      return get_type_label<typename T::Solver>() + "_" + get_type_label<typename T::PC>();
    }
  };

  template<typename Pair>
  class LinearSolver_Krylov : public ::testing::Test {};

  TYPED_TEST_SUITE(LinearSolver_Krylov, AllKrylovCombos, SolverPCName);
  TYPED_TEST(LinearSolver_Krylov, SolvesLaplace1D)
  {
    using Solver = typename TypeParam::Solver;
    using PC = typename TypeParam::PC;

    constexpr size_t n = 100;
    auto A = utils::laplace_1d(n);
    Vector<> b(n, 1.0);

    SolverControl control(1000, 1e-10);
    Solver solver(&control);
    PC M;
    M.build(&A);

    Vector<> x(b.size(), 0.0);
    auto result = solver.solve(A, b, x, M);

    std::cout << std::setw(18) << std::left
        << SolverPCName::GetName<TypeParam>(0)
        << " | iter: " << std::setw(4) << result.iterations
        << " | final ||r||: " << result.residual_norm << "\n";

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 1000);
    EXPECT_LE(result.residual_norm, 1e-8);
  }
}
