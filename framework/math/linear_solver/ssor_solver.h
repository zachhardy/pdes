#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"
#include <stdexcept>
#include <string>

namespace pdes
{
  /**
   * Symmetric Successive Over-Relaxation (SSOR) solver for Ax = b.
   *
   * SSOR performs both a forward and backward Gauss-Seidel sweep each iteration,
   * using a relaxation factor \f$ \omega \in (0, 2) \f$. It can improve convergence
   * behavior over standard SOR when used as a smoother or preconditioner.
   *
   * Requires the matrix to be square and typically performs best on diagonally dominant systems.
   *
   * @tparam MatrixType The matrix type to use (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class SSORSolver final : public LinearSolver<MatrixType>
  {
  public:
    using Base = LinearSolver<MatrixType>;
    using Result = typename Base::Result;
    using VectorType = typename Base::VectorType;
    using value_type = typename VectorType::value_type;

    /// Constructs an SSOR solver with default omega = 1.3.
    explicit SSORSolver(SolverControl* control) : Base(control) {}

    /// Constructs an SSOR solver with specified relaxation factor.
    explicit SSORSolver(SolverControl* control, value_type omega);

    /// Returns the name of the solver.
    std::string name() const override { return "SSORSolver"; }

    using Base::solve;

    /**
     * Solves Ax = b using SSOR iteration.
     *
     * @param A System matrix.
     * @param b Right-hand side vector.
     * @param x Solution vector (initial guess and result).
     * @param M Unused preconditioner (ignored).
     * @return Solver result with convergence status.
     */
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const Preconditioner<VectorType>&) const override;

    /// Relaxation parameter \f$ \omega \in (0, 2) \f$.
    value_type omega_ = value_type(1.3);
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  SSORSolver<MatrixType>::SSORSolver(SolverControl* control, const value_type omega)
    : Base(control),
      omega_(omega)
  {
    if (omega_ <= 0.0 || omega_ >= 2.0)
      throw std::invalid_argument("SSOR relaxation parameter omega must be in (0, 2)");
  }

  template<typename MatrixType>
  typename SSORSolver<MatrixType>::Result
  SSORSolver<MatrixType>::solve(const MatrixType& A,
                                const VectorType& b,
                                VectorType& x,
                                const Preconditioner<VectorType>&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    for (unsigned int iter = 0;; ++iter)
    {
      // Forward sweep
      for (size_t i = 0; i < n; ++i)
      {
        value_type sum = 0;
        const auto row = A.begin(i);
        for (size_t j = 0; j < i; ++j)
          sum += row[j] * x(j);
        for (size_t j = i + 1; j < n; ++j)
          sum += row[j] * x(j);
        x(i) = (1 - omega_) * x(i) + omega_ * inv_diag[i] * (b(i) - sum);
      }

      // Backward sweep
      for (size_t i = n; i-- > 0;)
      {
        value_type sum = 0;
        const auto row = A.begin(i);
        for (size_t j = 0; j < i; ++j)
          sum += row[j] * x(j);
        for (size_t j = i + 1; j < n; ++j)
          sum += row[j] * x(j);
        x(i) = (1 - omega_) * x(i) + omega_ * inv_diag[i] * (b(i) - sum);
      }

      const auto residual_norm = A.residual_norm(x, b);
      this->log_iter(iter, residual_norm);
      if (not control.check(iter, residual_norm))
      {
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }
    }
  }
}
