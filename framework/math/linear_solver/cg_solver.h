#pragma once
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/linear_solver.h"


namespace pdes
{
  /**
   * Conjugate Gradient (CG) solver for symmetric positive-definite systems.
   *
   * This solver implements the classical CG method, an efficient Krylov subspace
   * technique for solving large sparse symmetric positive-definite systems.
   *
   * It supports preconditioning and returns a result struct containing convergence info.
   *
   * @tparam MatrixType The matrix type to use (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class CGSolver final : public LinearSolver<MatrixType>
  {
  public:
    using value_type = typename LinearSolver<MatrixType>::value_type;
    using VectorType = typename LinearSolver<MatrixType>::VectorType;
    using Result = typename LinearSolver<MatrixType>::Result;

    /// Constructs a CG solver with a given convergence controller.
    explicit CGSolver(SolverControl* control) : LinearSolver<MatrixType>(control) {}

    /// Returns the name of the solver.
    std::string name() const override { return "ConjugateGradientSolver"; }

    using LinearSolver<MatrixType>::solve;

    /**
     * Solves Ax = b using the CG method with preconditioner M.
     *
     * @param A Symmetric positive-definite system matrix.
     * @param b Right-hand side vector.
     * @param x Solution vector (initial guess and result).
     * @param M Preconditioner to apply on residuals.
     * @return Solver result with convergence status.
     */
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const Preconditioner<MatrixType>& M) const override;
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  typename CGSolver<MatrixType>::Result
  CGSolver<MatrixType>::solve(const MatrixType& A,
                              const VectorType& b,
                              VectorType& x,
                              const Preconditioner<MatrixType>& M) const
  {
    auto& control = *this->control_;

    const size_t n = b.size();
    VectorType r(n), z(n), p(n), v(n);

    // Initial preconditioned residual:
    // z = M * r = M * (b - A * x)
    A.residual(x, b, r);
    M.vmult(r, z);

    // Initialize search direction
    p = z;

    // Initialize dot product
    auto rz_old = r.dot(z);

    for (unsigned int iter = 0;; ++iter)
    {
      // Compute step-length:
      // v = A*p -> alpha = (r, z)/(p, v)
      A.vmult(p, v);
      const auto alpha = rz_old / p.dot(v);

      // Update solution: x += alpha * p
      x.add(alpha, p);

      // Update preconditioned residual:
      // z = M * r = M * (alpha * v)
      r.add(-alpha, v);
      M.vmult(r, z);

      // Compute new dot product for next iteration
      const auto rz_new = r.dot(z);

      // Check convergence based on residual norm
      const auto residual_norm = r.l2_norm();
      this->log_iter(iter, residual_norm);
      if (not control.check(iter, residual_norm))
      {
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }

      // Update search direction: p = z + beta * p
      const auto beta = rz_new / rz_old;
      p.sadd(beta, value_type(1), z);

      rz_old = rz_new;
    }
  }
}
