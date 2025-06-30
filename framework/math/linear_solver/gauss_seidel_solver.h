#pragma once
#include "framework/math/linear_solver/sor_solver.h"


namespace pdes
{
  /**
   * Gauss-Seidel iterative solver for linear systems Ax = b.
   *
   * The Gauss-Seidel method is a variant of the Jacobi iteration that uses updated
   * values as soon as they are available within each iteration. This often results
   * in faster convergence compared to Jacobi, particularly for diagonally dominant matrices.
   *
   * This class is implemented as a special case of the SORSolver with relaxation
   * factor omega = 1.0.
   *
   * @tparam MatrixType The matrix type to use (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class GaussSeidelSolver final : public SORSolver<MatrixType>
  {
  public:
    using Base = SORSolver<MatrixType>;
    using Result = typename Base::Result;
    using VectorType = typename Base::VectorType;
    using value_type = typename VectorType::value_type;

    /// Constructs an uninitialized Gauss-Seidel solver.
    GaussSeidelSolver() = default;

    /// Constructs a Gauss-Seidel solver with solver control.
    explicit GaussSeidelSolver(SolverControl* control) : Base(control, value_type(1)) {}

    /// Returns the name of the solver.
    std::string name() const override { return "GaussSeidelSolver"; }
  };
}
