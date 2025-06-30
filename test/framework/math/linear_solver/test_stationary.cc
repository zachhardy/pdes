#include <gtest/gtest.h>
#include "test/framework/math/linear_solver/utils/matrix_builder.h"
#include "test/framework/math/linear_solver/utils/test_convergence.h"
#include "framework/math/linear_solver/jacobi_solver.h"
#include "framework/math/linear_solver/gauss_seidel_solver.h"
#include "framework/math/linear_solver/sor_solver.h"
#include "framework/math/linear_solver/ssor_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include <iomanip>

#include "framework/math/linear_solver/preconditioner/precondition_ilu.h"
#include "framework/math/linear_solver/preconditioner/precondition_jacobi.h"
#include "framework/math/linear_solver/preconditioner/precondition_ssor.h"

namespace pdes::test
{
  using AllStationarySolvers = ::testing::Types<
    JacobiSolver<>,
    GaussSeidelSolver<>,
    SORSolver<>,
    SSORSolver<>>;

  // clang-format off
  template <typename T>
  std::string get_type_label()
  {
    if constexpr (std::is_same_v<T, JacobiSolver<>>) { return "Jacobi"; }
    else if constexpr (std::is_same_v<T, GaussSeidelSolver<>>) { return "GaussSeidel"; }
    else if constexpr (std::is_same_v<T, SORSolver<>>) { return "SOR"; }
    else if constexpr (std::is_same_v<T, SSORSolver<>>) { return "SSOR"; }
    else return "Unknown";
  }

  struct SolverName {
    template <typename T>
    static std::string GetName(int) { return get_type_label<T>(); }
  };

  template<typename SolverType>
  class LinearSolver_Stationary : public ::testing::Test {};

  TYPED_TEST_SUITE(LinearSolver_Stationary, AllStationarySolvers, SolverName);
  TYPED_TEST(LinearSolver_Stationary, SolvesLaplace1D)
  {
    using SolverType = TypeParam;

    constexpr size_t n = 10;
    auto A = utils::laplace_1d(n);
    Vector<> b(n, 1.0);

    SolverControl control(1000, 1e-10);
    SolverType solver(&control);

    Vector<> x(b.size(), 0.0);
    auto result = solver.solve(A, b, x);


    std::cout << std::setw(18) << std::left
        << SolverName::GetName<TypeParam>(0)
        << " | iter: " << std::setw(4) << result.iterations
        << " | final ||r||: " << result.residual_norm << "\n";

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 1000);
    EXPECT_LE(result.residual_norm, 1e-8);
  }
}
