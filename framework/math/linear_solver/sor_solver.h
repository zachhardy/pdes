#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  /**
   * Successive Over-Relaxation (SOR) iterative solver for Ax = b.
   * Templated on scalar type (default = types::real).
   */
  template<typename VectorType = Vector<>>
  class SORSolver : public LinearSolver<SORSolver<VectorType>, VectorType>
  {
  public:
    using value_type = typename VectorType::value_type;
    using Result = typename LinearSolver<SORSolver, VectorType>::Result;

    explicit SORSolver(SolverControl* control)
      : LinearSolver<SORSolver, VectorType>(control)
    {}

    explicit SORSolver(SolverControl* control, value_type omega);

    std::string name() const override { return "SORSolver"; }

    using LinearSolver<SORSolver, VectorType>::solve;

    template<typename MatrixType, typename PreconditionerType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const PreconditionerType&) const;

    value_type omega_ = value_type(1.3);
  };

  /*-------------------- inline functions --------------------*/

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
