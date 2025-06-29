#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  /**
   * @brief Successive Over-Relaxation (SOR) solver for linear systems Ax = b.
   *
   * This class implements the SOR method, which generalizes the Gauss-Seidel method
   * by introducing a relaxation parameter \f$ \omega \in (0, 2) \f$ to accelerate
   * convergence. A value of \f$ \omega = 1 \f$ recovers Gauss-Seidel, while values
   * above or below adjust the influence of the current iterate.
   *
   * The solver requires the matrix to be square and generally performs best on
   * diagonally dominant systems.
   *
   * @tparam VectorType The vector type to use (default: Vector<>).
   */
  template<typename VectorType = Vector<>>
  class SORSolver : public LinearSolver<SORSolver<VectorType>, VectorType>
  {
  public:
    using Base = LinearSolver<SORSolver, VectorType>;
    using Result = typename Base::Result;
    using value_type = typename VectorType::value_type;

    /// Constructs a solver with control and default omega = 1.3.
    explicit SORSolver(SolverControl* control) : Base(control) {}

    /// Constructs a solver with specified relaxation factor.
    explicit SORSolver(SolverControl* control, value_type omega);

    /// Returns the name of the solver.
    std::string name() const override { return "SORSolver"; }

    using Base::solve;

    /**
     * Solves Ax = b using SOR iteration.
     *
     * @param A System matrix.
     * @param b Right-hand side vector.
     * @param x Solution vector (initial guess and result).
     * @param M Unused preconditioner (ignored).
     * @return Solver result with convergence status.
     */
    template<typename MatrixType, typename PreconditionerType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const PreconditionerType&) const;

    /// Relaxation parameter \f$ \omega \in (0, 2) \f$.
    value_type omega_ = value_type(1.3);
  };

  /*-------------------- member functions --------------------*/

  template<typename VectorType>
  SORSolver<VectorType>::SORSolver(SolverControl* control, const value_type omega)
    : LinearSolver<SORSolver, VectorType>(control),
      omega_(omega)
  {
    if (omega_ <= 0.0 || omega_ >= 2.0)
      throw std::invalid_argument("SOR relaxation parameter omega must be in (0, 2)");
  }

  template<typename VectorType>
  template<typename MatrixType, typename PreconditionerType>
  typename SORSolver<VectorType>::Result
  SORSolver<VectorType>::solve(const MatrixType& A,
                               const VectorType& b,
                               VectorType& x,
                               const PreconditionerType&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    for (unsigned int iter = 0;; ++iter)
    {
      for (size_t i = 0; i < n; ++i)
      {
        value_type sum = 0;
        const value_type* row = A.begin(i);
        for (size_t j = 0; j < i; ++j)
          sum += row[j] * x(j);
        for (size_t j = i + 1; j < n; ++j)
          sum += row[j] * x(j);
        x(i) = (1.0 - omega_) * x(i) + omega_ * inv_diag[i] * (b(i) - sum);
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
