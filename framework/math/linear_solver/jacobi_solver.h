#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  /**
   * Jacobi iterative solver for linear systems Ax = b.
   *
   * This is a basic stationary method that iteratively updates the solution vector
   * based on the diagonal components of the system matrix. It is simple to implement
   * and parallelize, though it typically converges slowly and is only effective for
   * diagonally dominant or well-conditioned systems.
   *
   * @tparam MatrixType The matrix type to use (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class JacobiSolver final : public LinearSolver<MatrixType>
  {
  public:
    using value_type = typename LinearSolver<MatrixType>::value_type;
    using VectorType = typename LinearSolver<MatrixType>::VectorType;
    using Result = typename LinearSolver<MatrixType>::Result;


    /// Constructs an uninitialized Jacobi solver.
    JacobiSolver() = default;

    /// Constructs a Jacobi solver with solver control parameters.
    explicit JacobiSolver(SolverControl* control) : LinearSolver<MatrixType>(control) {}

    /// Returns the name of the solver.
    std::string name() const override { return "JacobiSolver"; }

    using LinearSolver<MatrixType>::solve;

    /**
     * Solves Ax = b using Jacobi iteration.
     *
     * @param A System matrix.
     * @param b Right-hand side vector.
     * @param x Solution vector (initial guess and result).
     * @return Solver result with convergence status.
     */
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const Preconditioner<MatrixType>&) const override;
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  typename JacobiSolver<MatrixType>::Result
  JacobiSolver<MatrixType>::solve(const MatrixType& A,
                                  const VectorType& b,
                                  VectorType& x,
                                  const Preconditioner<MatrixType>&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    VectorType x_old(n, value_type(0));
    for (unsigned int iter = 0;; ++iter)
    {
      x_old = x;

      // Compute the updated iterate
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < n; ++i)
      {
        value_type sum = 0;
        const value_type* row = A.begin(i);
        for (size_t j = 0; j < n; ++j)
          if (j != i)
            sum += row[j] * x_old(j);
        x(i) = inv_diag[i] * (b(i) - sum);
      }

      const auto residual_norm = A.residual_norm(x, b);
      this->log_iter(iter, residual_norm);

      if (not control.check(iter, residual_norm))
      {
        // Return result if converged
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }
    }
  }
}

